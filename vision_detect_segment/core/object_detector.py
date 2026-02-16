import base64
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import supervision as sv
import torch

try:
    from redis_robot_comm import RedisMessageBroker
except ImportError:
    RedisMessageBroker = None

from ..utils.config import VisionConfig
from ..utils.exceptions import (
    ModelLoadError,
    handle_detection_error,
    handle_model_loading_error,
    handle_redis_error,
)
from ..utils.utils import (
    Timer,
    get_optimal_device,
    setup_logging,
    validate_confidence_threshold,
)
from .detectors.owlv2 import GroundingDinoBackend, Owlv2Backend
from .detectors.yolo import YOLOWorldBackend
from .detectors.yoloe import YOLOEBackend
from .object_segmenter import ObjectSegmenter
from .object_tracker import ObjectTracker

# Handle optional dependencies gracefully for backward compatibility and tests
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    YOLO = None
    YOLO_AVAILABLE = False

try:
    from ultralytics import YOLOE

    YOLOE_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    YOLOE = None
    YOLOE_AVAILABLE = False

try:
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        Owlv2ForObjectDetection,
        Owlv2Processor,
    )

    TRANSFORMERS_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    Owlv2Processor = None
    Owlv2ForObjectDetection = None
    AutoProcessor = None
    AutoModelForZeroShotObjectDetection = None
    TRANSFORMERS_AVAILABLE = False


class ObjectDetector:
    """
    Unified object detection class supporting multiple model backends via Strategy Pattern.

    This class provides a common interface for different object detection models,
    handling model loading, inference, tracking coordination, and result publishing.

    Supports:
    - OWL-V2 (Open-vocabulary detection)
    - Grounding-DINO (Text-guided detection)
    - YOLO-World (Real-time detection)
    - YOLOE (Unified Detection & Segmentation)
    """

    def __init__(
        self,
        device: str,
        model_id: str,
        object_labels: List[List[str]],
        redis_host: str = "localhost",
        redis_port: int = 6379,
        stream_name: str = "detected_objects",
        verbose: bool = False,
        config: Optional[VisionConfig] = None,
        enable_tracking: bool = True,
        publish_during_movement: bool = False,
    ):
        """
        Initialize ObjectDetector.

        Args:
            device: Computation device ("cuda", "cpu", or "auto")
            model_id: Model identifier (e.g., "owlv2", "yoloe-11s")
            object_labels: Nested list of object labels to detect
            redis_host: Redis server host
            redis_port: Redis server port
            stream_name: Redis stream name for publishing detections
            verbose: Enable verbose logging
            config: Optional VisionConfig instance
            enable_tracking: Enable object tracking
            publish_during_movement: Whether to publish detections when robot is moving
        """
        # Private attributes
        self._device = get_optimal_device(prefer_gpu=(device != "cpu"))
        self._model_id = model_id
        self._object_labels = object_labels
        self._current_detections: Optional[sv.Detections] = None
        self._current_labels: Optional[np.ndarray] = None
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._stream_name = stream_name
        self._publish_during_movement = publish_during_movement
        self._verbose = verbose

        # Public configuration
        self._config = config or VisionConfig()

        log_filename = os.path.join("log", f'object_detector_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        # Setup logging
        self._logger = setup_logging(verbose, log_filename)

        try:
            # Factory for backend
            self._backend = self._create_backend(model_id, self._device, object_labels[0])

            with Timer(f"Loading {model_id} model", self._logger):
                self._backend.load_model()

            # Initialize segmenter
            self._segmenter = ObjectSegmenter(device=self._device, verbose=verbose)

            # Initialize tracker
            self._tracker = ObjectTracker(
                model=getattr(self._backend, "model", None),
                model_id=self._model_id,
                enable_tracking=enable_tracking,
                verbose=self._verbose,
            )

            # Initialize Redis broker
            self._setup_redis(redis_host, redis_port)

            if self._verbose:
                self._logger.info(f"ObjectDetector initialized with {model_id} on {self._device}")

        except Exception as e:
            model_error = handle_model_loading_error(model_id, e)
            self._logger.error(str(model_error))
            raise model_error

    def _create_backend(self, model_id: str, device: str, labels: List[str]):
        """
        Factory method to create the appropriate backend.

        Args:
            model_id: Identifier of the model
            device: Computing device
            labels: List of labels

        Returns:
            An instance of DetectionBackend
        """
        if model_id == "owlv2":
            return Owlv2Backend(model_id, device, labels)
        elif model_id == "grounding_dino":
            return GroundingDinoBackend(model_id, device, labels)
        elif model_id == "yolo-world":
            return YOLOWorldBackend(model_id, device, labels)
        elif "yoloe" in model_id.lower():
            return YOLOEBackend(model_id, device, labels)
        else:
            raise ModelLoadError(model_id, f"Unsupported model ID: {model_id}")

    def _setup_redis(self, host: str, port: int) -> None:
        """
        Initialize Redis connection.

        Args:
            host: Redis host
            port: Redis port
        """
        if RedisMessageBroker is None:
            if self._verbose:
                self._logger.warning("RedisMessageBroker not available")
            self._redis_broker = None
            return

        try:
            self._redis_broker = RedisMessageBroker(host, port)
            self._redis_broker.client.xtrim(self._stream_name, maxlen=100, approximate=True)
        except Exception as e:
            redis_error = handle_redis_error("connection", host, port, e)
            if self._verbose:
                self._logger.warning(f"Redis initialization failed: {redis_error}")
            self._redis_broker = None

    def detect_objects(
        self, image: np.ndarray, confidence_threshold: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run object detection or tracking on the given image.

        Args:
            image: Input image as numpy array in RGB format
            confidence_threshold: Minimum confidence for detections (overrides config)
            metadata: Optional metadata containing robot state (e.g., is_moving)

        Returns:
            List of detected objects as dictionaries
        """
        if confidence_threshold is not None:
            validate_confidence_threshold(confidence_threshold)

        threshold = confidence_threshold or self._config.model.confidence_threshold

        try:
            with Timer("Object detection", self._logger if self._verbose else None):
                # Delegate detection to backend strategy
                detected_objects = self._backend.detect(image, threshold)

                # Post-process for tracking if enabled
                track_ids = self._handle_tracking(detected_objects)

                # Apply label stabilization based on history
                detected_objects = self._apply_label_stabilization(detected_objects, track_ids)

                # Add segmentation masks
                detected_objects = self._handle_segmentation(detected_objects, image)

                # Update internal state for annotation tools
                self._update_supervision_state(detected_objects, track_ids)

                # Conditionally publish based on movement
                if self._should_publish_detections(metadata):
                    self._publish_detections(detected_objects, self._model_id)
                elif self._verbose:
                    self._logger.info("Skipping detection publishing - robot is moving")

                return detected_objects
        except Exception as e:
            detection_error = handle_detection_error(e, image.shape, self._model_id)
            if self._verbose:
                self._logger.error(str(detection_error))
            return []

    def _handle_tracking(self, detected_objects: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Handle tracking logic for both built-in and external trackers."""
        if not self._tracker or not self._tracker.enable_tracking:
            return None

        track_ids = None
        if not self._backend.supports_tracking:
            # Use ByteTrack for models without built-in tracking
            boxes = np.array(
                [
                    [obj["bbox"]["x_min"], obj["bbox"]["y_min"], obj["bbox"]["x_max"], obj["bbox"]["y_max"]]
                    for obj in detected_objects
                ]
            )
            scores = np.array([obj["confidence"] for obj in detected_objects])
            if len(boxes) > 0:
                detections = sv.Detections(xyxy=boxes, confidence=scores, class_id=np.zeros(len(boxes), dtype=int))
                tracked_detections = self._tracker.update_with_detections(detections)
                track_ids = tracked_detections.tracker_id

                for i, obj in enumerate(detected_objects):
                    if i < len(track_ids):
                        obj["track_id"] = int(track_ids[i])
        else:
            # Backend handled tracking, extract track IDs
            track_ids = np.array([obj["track_id"] for obj in detected_objects if "track_id" in obj])
            if len(track_ids) == 0:
                track_ids = None

        return track_ids

    def _handle_segmentation(self, detected_objects: List[Dict[str, Any]], image: np.ndarray) -> List[Dict[str, Any]]:
        """Coordinate segmentation between backend and external segmenter."""
        if not self._config.enable_segmentation:
            return detected_objects

        if self._backend.supports_segmentation:
            # Process masks already provided by backend
            for obj in detected_objects:
                if "mask_8u" in obj:
                    obj["mask_data"] = self._serialize_mask(obj["mask_8u"])
                    obj["mask_shape"] = list(obj["mask_8u"].shape)
                    obj["mask_dtype"] = str(obj["mask_8u"].dtype)
                    obj["has_mask"] = True
                    del obj["mask_8u"]
        else:
            # Use external segmenter (SAM/FastSAM)
            if detected_objects:
                boxes = torch.tensor(
                    [
                        [obj["bbox"]["x_min"], obj["bbox"]["y_min"], obj["bbox"]["x_max"], obj["bbox"]["y_max"]]
                        for obj in detected_objects
                    ]
                )
                detected_objects = self._add_segmentation(detected_objects, image, boxes)

        return detected_objects

    def _update_supervision_state(self, objects: List[Dict[str, Any]], track_ids: Optional[np.ndarray]) -> None:
        """Update sv.Detections and labels for annotation."""
        if not objects:
            self._current_detections = None
            self._current_labels = None
            return

        try:
            xyxy = np.array(
                [[obj["bbox"]["x_min"], obj["bbox"]["y_min"], obj["bbox"]["x_max"], obj["bbox"]["y_max"]] for obj in objects]
            )
            confidence = np.array([obj.get("confidence", 0.0) for obj in objects])
            class_id = np.zeros(len(objects), dtype=int)

            detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
            if track_ids is not None and len(track_ids) == len(objects):
                detections.tracker_id = track_ids

            self._current_detections = detections
            self._current_labels = np.array([obj["label"] for obj in objects])
        except (KeyError, TypeError):
            pass

    def _apply_label_stabilization(
        self, detected_objects: List[Dict[str, Any]], track_ids: Optional[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Apply label stabilization based on tracking history."""
        if not self._tracker or not self._tracker.enable_tracking or track_ids is None:
            return detected_objects

        current_labels = [obj["label"] for obj in detected_objects]
        stabilized_labels = self._tracker.update_label_history(track_ids, current_labels)

        for obj, stabilized_label in zip(detected_objects, stabilized_labels):
            obj["label"] = stabilized_label

        lost_tracks = self._tracker.detect_lost_tracks(track_ids)
        if lost_tracks:
            self._tracker.cleanup_lost_tracks(lost_tracks)

        return detected_objects

    def _add_segmentation(self, objects: List[Dict[str, Any]], image: np.ndarray, boxes: torch.Tensor) -> List[Dict[str, Any]]:
        """Add segmentation masks using the external segmenter."""
        if self._segmenter.get_segmenter() is None:
            return objects

        for i, (obj, box) in enumerate(zip(objects, boxes)):
            try:
                mask_8u, _ = self._segmenter.segment_box_in_image(box, image)
                if mask_8u is not None:
                    obj["mask_data"] = self._serialize_mask(mask_8u)
                    obj["has_mask"] = True
                    obj["mask_shape"] = list(mask_8u.shape)
                    obj["mask_dtype"] = str(mask_8u.dtype)
            except Exception as e:
                if self._verbose:
                    self._logger.warning(f"Segmentation failed for object {i}: {e}")
        return objects

    @staticmethod
    def _serialize_mask(mask: np.ndarray) -> str:
        """Serialize numpy mask to base64 string."""
        return base64.b64encode(mask.tobytes()).decode("utf-8")

    def _should_publish_detections(self, metadata: Optional[Dict[str, Any]]) -> bool:
        """
        Determine if detections should be published based on robot state.

        Args:
            metadata: Image metadata potentially containing robot movement state

        Returns:
            bool: True if detections should be published
        """
        if self._publish_during_movement or metadata is None:
            return True

        is_moving = (
            metadata.get("is_moving", False) or metadata.get("robot_moving", False) or metadata.get("camera_moving", False)
        )

        if is_moving and self._verbose:
            self._logger.debug("Robot movement detected - skipping detection publishing")
            return False
        return True

    def _publish_detections(self, objects: List[Dict[str, Any]], method: str) -> None:
        """Publish detections to Redis."""
        if not objects or self._redis_broker is None:
            return

        # Clean object dicts of large or non-serializable objects before publishing
        clean_objects = []
        for obj in objects:
            clean_obj = obj.copy()
            if "results" in clean_obj:
                del clean_obj["results"]
            clean_objects.append(clean_obj)

        metadata = {
            "timestamp": time.time(),
            "object_count": len(objects),
            "detection_method": method,
            "model_id": self._model_id,
        }

        try:
            self._redis_broker.publish_objects(clean_objects, metadata, maxlen=500)
        except Exception as e:
            if self._verbose:
                self._logger.warning(f"Redis publish failed: {e}")

    # Public API methods
    def add_label(self, label: str) -> None:
        """Add a new detectable object label."""
        self._object_labels[0].append(label.lower())
        self._backend.add_label(label)

    def get_detections(self) -> Optional[sv.Detections]:
        """Get current supervision detections."""
        return self._current_detections

    def get_label_texts(self) -> Optional[np.ndarray]:
        """Get current detection labels (with track IDs if available)."""
        if self._current_detections is None or self._current_labels is None:
            return None

        labels = []
        has_tracking = hasattr(self._current_detections, "tracker_id") and self._current_detections.tracker_id is not None

        for i, label in enumerate(self._current_labels):
            conf = self._current_detections.confidence[i] if self._current_detections.confidence is not None else None
            text = f"{label}"
            if has_tracking and self._current_detections.tracker_id[i] is not None:
                text += f" #{int(self._current_detections.tracker_id[i])}"
            if conf is not None:
                text += f" ({conf:.2f})"
            labels.append(text)

        return np.array(labels)

    def get_object_labels(self) -> List[List[str]]:
        """Get current object labels."""
        return self._object_labels

    def get_device(self) -> str:
        """Get current device."""
        return self._device

    def get_model_id(self) -> str:
        """Get current model ID."""
        return self._model_id

    @property
    def _processed_labels(self):
        """Legacy property for backward compatibility in tests."""
        return getattr(self._backend, "processed_labels", None)

    @_processed_labels.setter
    def _processed_labels(self, value):
        """Legacy setter for backward compatibility in tests."""
        if hasattr(self._backend, "processed_labels"):
            self._backend.processed_labels = value

    def set_publish_during_movement(self, enable: bool) -> None:
        """Set publishing during movement flag."""
        self._publish_during_movement = enable

    # Backward compatibility for tests and old code
    def _load_model(self, model_id: str):
        return self._backend.load_model()

    def _validate_model_availability(self):
        pass

    @staticmethod
    def _preprocess_labels(labels: List[List[str]], model_id: str) -> Union[List[List[str]], str]:
        if model_id == "grounding_dino":
            flat_labels = [label.lower() for label in labels[0]]
            return ". ".join(flat_labels) + "."
        return labels

    @staticmethod
    def _create_object_dicts(results: Dict, labels: Union[List[str], np.ndarray]) -> List[Dict]:
        detected_objects = []
        for i, (box, score) in enumerate(zip(results["boxes"], results["scores"])):
            x_min, y_min, x_max, y_max = map(int, box)
            label = labels[i] if isinstance(labels, (list, np.ndarray)) else labels
            detected_objects.append(
                {
                    "label": str(label),
                    "confidence": float(score),
                    "bbox": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
                    "has_mask": False,
                }
            )
        return detected_objects

    @staticmethod
    def _convert_labels_to_class_ids(labels: List[str]) -> np.ndarray:
        return np.array([hash(label.lower()) % 100 for label in labels])

    def _create_supervision_detections(self, results, objects, track_ids=None):
        self._update_supervision_state(objects, track_ids)

    def _create_supervision_detections_from_results(self, results, labels, track_ids=None):
        objects = [
            {
                "bbox": {"x_min": int(box[0]), "y_min": int(box[1]), "x_max": int(box[2]), "y_max": int(box[3])},
                "confidence": float(score),
                "label": labels[i],
            }
            for i, (box, score) in enumerate(zip(results["boxes"], results["scores"]))
        ]
        self._update_supervision_state(objects, track_ids)

    def _detect_transformer_based(self, image: np.ndarray, threshold: float) -> List[Dict]:
        """Legacy method for backward compatibility in tests."""
        return self._backend.detect(image, threshold)

    def _detect_yolo(self, image: np.ndarray, threshold: float) -> List[Dict]:
        """Legacy method for backward compatibility in tests."""
        return self._backend.detect(image, threshold)

    def _detect_yoloe(self, image: np.ndarray, threshold: float) -> List[Dict]:
        """Legacy method for backward compatibility in tests."""
        return self._backend.detect(image, threshold)

    # Deprecated
    def detections(self):
        return self.get_detections()

    def label_texts(self):
        return self.get_label_texts()

    def object_labels(self):
        return self.get_object_labels()
