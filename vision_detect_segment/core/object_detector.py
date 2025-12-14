import numpy as np
import time
import base64
import os
from typing import List, Dict, Optional, Tuple, Union
import torch
import supervision as sv
from datetime import datetime

from .object_segmenter import ObjectSegmenter
from .object_tracker import ObjectTracker

from redis_robot_comm import RedisMessageBroker

from ..utils.config import VisionConfig, MODEL_CONFIGS
from ..utils.exceptions import (
    ModelLoadError,
    DependencyError,
    handle_model_loading_error,
    handle_detection_error,
    handle_redis_error,
)
from ..utils.utils import setup_logging, validate_model_requirements, get_optimal_device, Timer, validate_confidence_threshold

# Handle optional dependencies gracefully
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except (ModuleNotFoundError, ImportError) as e:
    print(f"YOLO not available: {e}")
    YOLO_AVAILABLE = False

try:
    from ultralytics import YOLOE

    YOLOE_AVAILABLE = True
except (ModuleNotFoundError, ImportError) as e:
    print(f"YOLOE not available: {e}")
    YOLOE_AVAILABLE = False

try:
    from transformers import Owlv2Processor, Owlv2ForObjectDetection, AutoProcessor, AutoModelForZeroShotObjectDetection

    TRANSFORMERS_AVAILABLE = True
except (ModuleNotFoundError, ImportError) as e:
    print(f"Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False


class ObjectDetector:
    """
    Unified object detection class supporting multiple model backends.

    Supports:
    - OWL-V2 (Open-vocabulary detection)
    - Grounding-DINO (Text-guided detection)
    - YOLO-World (Real-time detection)
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
    ):
        """
        Initialize ObjectDetector.

        Args:
            device: Computation device ("cuda" or "cpu")
            model_id: Model identifier (owlv2, grounding_dino, yolo-world)
            object_labels: Nested list of object labels to detect
            redis_host: Redis server host
            redis_port: Redis server port
            stream_name: Redis stream name for publishing detections
            verbose: Enable verbose logging
            config: Optional VisionConfig instance
            enable_tracking:
        """
        # Private attributes
        self._device = get_optimal_device(prefer_gpu=(device != "cpu"))
        self._model_id = model_id
        self._object_labels = object_labels
        self._processed_labels = None
        self._model = None
        self._processor = None
        self._current_detections = None
        self._current_labels = None
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._stream_name = stream_name

        # Public configuration
        self.verbose = verbose
        self._config = config or VisionConfig()

        log_filename = os.path.join("log", f'object_detector_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        # Setup logging
        self._logger = setup_logging(verbose, log_filename)

        # Validate dependencies and load model
        try:
            self._validate_model_availability()
            validate_model_requirements(model_id)

            # Preprocess labels based on model requirements
            self._processed_labels = ObjectDetector._preprocess_labels(object_labels, model_id)

            # Load model with timing
            with Timer(f"Loading {model_id} model", self._logger):
                self._model, self._processor = self._load_model(model_id)

            # Initialize segmenter
            self._segmenter = ObjectSegmenter(device=self._device, verbose=verbose)

            self._tracker = ObjectTracker(
                model=self._model, model_id=self._model_id, enable_tracking=enable_tracking, verbose=self.verbose
            )

            # Initialize Redis broker
            try:
                self._redis_broker = RedisMessageBroker(redis_host, redis_port)
            except Exception as e:
                redis_error = handle_redis_error("connection", redis_host, redis_port, e)
                if verbose:
                    self._logger.warning(f"Redis initialization failed: {redis_error}")
                self._redis_broker = None

            if verbose:
                self._logger.info(f"ObjectDetector initialized with {model_id} on {self._device}")

        except Exception as e:
            model_error = handle_model_loading_error(model_id, e)
            self._logger.error(str(model_error))
            raise model_error

    def _validate_model_availability(self):
        """Check if required dependencies are available for the selected model."""
        if "yoloe" in self._model_id.lower() and not YOLOE_AVAILABLE:
            raise DependencyError(
                "ultralytics (with YOLOE support)",
                f"model {self._model_id}",
                "Install with: pip install -U ultralytics>=8.3.0",
            )
        elif self._model_id == "yolo-world" and not YOLO_AVAILABLE:
            raise DependencyError("ultralytics", f"model {self._model_id}", "Install with: pip install ultralytics")
        elif self._model_id in ["owlv2", "grounding_dino"] and not TRANSFORMERS_AVAILABLE:
            raise DependencyError("transformers", f"model {self._model_id}", "Install with: pip install transformers")
        elif self._model_id not in MODEL_CONFIGS:
            available = list(MODEL_CONFIGS.keys())
            raise ModelLoadError(self._model_id, f"Unsupported model. Available: {available}")

    @staticmethod
    def _preprocess_labels(labels: List[List[str]], model_id: str) -> Union[List[List[str]], str]:
        """Preprocess labels based on model requirements."""
        if model_id == "grounding_dino":
            # Grounding DINO needs lowercase labels joined with periods
            flat_labels = [label.lower() for label in labels[0]]
            return ". ".join(flat_labels) + "."
        return labels

    def detect_objects(self, image: np.ndarray, confidence_threshold: Optional[float] = None) -> List[Dict]:
        """
        Run object detection or tracking on the given image.
        Returns a list of detections, optionally with track IDs.

        Args:
            image: Input image as numpy array in RGB format
            confidence_threshold: Minimum confidence for detections

        Returns:
            List of detected objects as dictionaries
        """
        if confidence_threshold is not None:
            validate_confidence_threshold(confidence_threshold)

        threshold = confidence_threshold or self._config.model.confidence_threshold

        try:
            with Timer("Object detection", self._logger if self.verbose else None):
                if "yoloe" in self._model_id.lower():
                    return self._detect_yoloe(image, threshold)
                elif self._model_id == "yolo-world":
                    return self._detect_yolo(image, threshold)
                else:
                    return self._detect_transformer_based(image, threshold)
        except Exception as e:
            detection_error = handle_detection_error(e, image.shape, self._model_id)
            print(detection_error)
            if self.verbose:
                self._logger.error(str(detection_error))
            return []

    def _detect_yolo(self, image: np.ndarray, threshold: float) -> List[Dict]:
        """Run YOLO-World detection with label stabilization."""
        if self._tracker and self._tracker._use_yolo_tracker:
            results = self._tracker.track(image, threshold)
        else:
            results = self._model.predict(image, conf=threshold, max_det=20, verbose=False)

        detected_objects = []
        boxes = results[0].boxes

        if boxes is None:
            return detected_objects

        for i, box in enumerate(boxes):
            cls = int(boxes.cls[i])
            class_name = results[0].names[cls]
            confidence = float(boxes.conf[i])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            obj_dict = {
                "label": class_name,
                "confidence": confidence,
                "bbox": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2},
                "has_mask": False,
            }
            detected_objects.append(obj_dict)

        # Extract track IDs - FIXED: Handle both tensor and numpy array
        track_ids = None
        if hasattr(results[0].boxes, "id") and results[0].boxes.id is not None:
            raw_ids = results[0].boxes.id

            # Convert to numpy array if it's a tensor
            if isinstance(raw_ids, torch.Tensor):
                track_ids = raw_ids.cpu().numpy().astype(int)
            elif isinstance(raw_ids, np.ndarray):
                track_ids = raw_ids.astype(int)
            else:
                # Try to convert to numpy array
                track_ids = np.array(raw_ids, dtype=int)

            if self.verbose:
                print(f"object_detector track_ids: {track_ids}")

            # Add track IDs to objects
            for i, obj in enumerate(detected_objects):
                if i < len(track_ids):
                    obj["track_id"] = int(track_ids[i])

        # ✅ Apply label stabilization
        detected_objects = self._apply_label_stabilization(detected_objects, track_ids)

        # Update supervision detections
        self._create_supervision_detections(results, detected_objects, track_ids)

        # Publish to Redis
        self._publish_detections(detected_objects, "yolo-world")

        return detected_objects

    def _detect_yoloe(self, image: np.ndarray, threshold: float) -> List[Dict]:
        """
        Run YOLOE detection and segmentation.

        YOLOE performs both detection and segmentation in one pass.
        It supports tracking and has built-in segmentation capabilities.
        """
        # YOLOE supports tracking similar to YOLO-World
        if self._tracker and self._tracker._use_yolo_tracker:
            results = self._tracker.track(image, threshold)
        else:
            results = self._model.predict(image, conf=threshold, max_det=20, verbose=False)

        detected_objects = []
        boxes = results[0].boxes

        if boxes is None:
            return detected_objects

        # Extract detection information
        for i, box in enumerate(boxes):
            cls = int(boxes.cls[i])
            class_name = results[0].names[cls]
            confidence = float(boxes.conf[i])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            obj_dict = {
                "label": class_name,
                "confidence": confidence,
                "bbox": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2},
                "has_mask": False,
            }

            # Add track ID if available
            if hasattr(results[0].boxes, "id") and results[0].boxes.id is not None:
                track_id = int(results[0].boxes.id[i])
                obj_dict["track_id"] = track_id

            detected_objects.append(obj_dict)

        # Extract track IDs
        track_ids = None
        if hasattr(results[0].boxes, "id") and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        # ✅ Apply label stabilization
        detected_objects = self._apply_label_stabilization(detected_objects, track_ids)

        # Handle segmentation masks (YOLOE has built-in segmentation)
        if hasattr(results[0], "masks") and results[0].masks is not None:
            masks_data = results[0].masks.data

            for i, obj in enumerate(detected_objects):
                if i < len(masks_data):
                    mask = masks_data[i].cpu().numpy()
                    mask_8u = (mask * 255).astype(np.uint8)

                    obj["mask_data"] = self._serialize_mask(mask_8u)
                    obj["has_mask"] = True
                    obj["mask_shape"] = list(mask_8u.shape)
                    obj["mask_dtype"] = str(mask_8u.dtype)

                    if self.verbose:
                        self._logger.debug(f"Added segmentation mask for {obj['label']}")

        # Create supervision detections
        self._create_supervision_detections(results, detected_objects, track_ids)

        # Publish to Redis
        self._publish_detections(detected_objects, self._model_id)

        return detected_objects

    def _detect_transformer_based(self, image: np.ndarray, threshold: float) -> List[Dict]:
        """Run OWL-V2 or Grounding-DINO detection with label stabilization."""
        h, w = image.shape[:2]

        # Prepare inputs
        inputs = self._processor(images=image, text=self._processed_labels, return_tensors="pt").to(self._device)

        # print("_detect_transformer_based:", inputs)

        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process results
        if self._model_id == "owlv2":
            results = self._processor.post_process_object_detection(
                outputs=outputs, target_sizes=[(h, w)], threshold=threshold
            )
            labels = self._extract_owlv2_labels(results)
        else:  # grounding_dino
            results = self._processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=threshold, text_threshold=0.3, target_sizes=[(h, w)]
            )
            labels = results[0]["labels"]

        boxes = results[0]["boxes"]
        scores = results[0]["scores"]

        track_ids = None

        # ✅ Tracking (optional)
        if self._tracker and self._tracker.enable_tracking:
            try:
                detections = sv.Detections(
                    xyxy=boxes.cpu().numpy(), confidence=scores.cpu().numpy(), class_id=np.zeros(len(boxes), dtype=int)
                )
                tracked_detections = self._tracker.update_with_detections(detections)
                track_ids = tracked_detections.tracker_id

            except Exception as e:
                if self.verbose:
                    self._logger.warning(f"Tracking update failed: {e}")

        # Convert to object dictionaries
        detected_objects = ObjectDetector._create_object_dicts(results[0], labels)

        # Attach track IDs
        if track_ids is not None:
            for i, obj in enumerate(detected_objects):
                if i < len(track_ids):
                    obj["track_id"] = int(track_ids[i])

        # ✅ Apply label stabilization
        detected_objects = self._apply_label_stabilization(detected_objects, track_ids)

        # Create supervision detections
        self._create_supervision_detections_from_results(results[0], [obj["label"] for obj in detected_objects], track_ids)

        # Add segmentation if available
        detected_objects = self._add_segmentation(detected_objects, image, results[0]["boxes"])

        # Publish to Redis
        self._publish_detections(detected_objects, self._model_id)

        return detected_objects

    def _apply_label_stabilization(self, detected_objects: List[Dict], track_ids: Optional[np.ndarray]) -> List[Dict]:
        """
        Apply label stabilization based on tracking history.

        Args:
            detected_objects: List of detected object dictionaries
            track_ids: Array of track IDs corresponding to detections

        Returns:
            List of detected objects with stabilized labels
        """
        if not self._tracker or not self._tracker.enable_tracking or track_ids is None:
            return detected_objects

        # Extract current labels
        current_labels = [obj["label"] for obj in detected_objects]

        # Get stabilized labels from tracker
        stabilized_labels = self._tracker.update_label_history(track_ids, current_labels)

        # Update detected objects with stabilized labels
        for obj, stabilized_label in zip(detected_objects, stabilized_labels):
            obj["label"] = stabilized_label

        # Clean up lost tracks
        if len(track_ids) > 0:
            lost_tracks = self._tracker.detect_lost_tracks(track_ids)
            if lost_tracks:
                self._tracker.cleanup_lost_tracks(lost_tracks)

        return detected_objects

    @staticmethod
    def _create_object_dicts(results: Dict, labels: Union[List[str], np.ndarray]) -> List[Dict]:
        """Create standardized object dictionaries from detection results."""
        detected_objects = []

        for i, (box, score) in enumerate(zip(results["boxes"], results["scores"])):
            x_min, y_min, x_max, y_max = map(int, box)
            label = labels[i] if isinstance(labels, (list, np.ndarray)) else labels

            obj_dict = {
                "label": str(label),
                "confidence": float(score),
                "bbox": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
                "has_mask": False,
            }
            detected_objects.append(obj_dict)

        return detected_objects

    def _add_segmentation(self, objects: List[Dict], image: np.ndarray, boxes: torch.Tensor) -> List[Dict]:
        """Add segmentation masks to detected objects."""
        if self._segmenter.segmenter() is None:
            return objects

        masks = []
        for i, (obj, box) in enumerate(zip(objects, boxes)):
            try:
                mask_8u, mask_binary = self._segmenter.segment_box_in_image(box, image)
                if mask_8u is not None:
                    # Serialize mask for Redis
                    obj["mask_data"] = ObjectDetector._serialize_mask(mask_8u)
                    obj["has_mask"] = True
                    obj["mask_shape"] = list(mask_8u.shape)  # [height, width]
                    obj["mask_dtype"] = str(mask_8u.dtype)  # 'uint8'
                    masks.append(mask_binary)
                else:
                    masks.append(None)
            except Exception as e:
                if self.verbose:
                    self._logger.warning(f"Segmentation failed for object {i}: {e}")
                masks.append(None)

        # Store masks for supervision
        if self._current_detections is not None:
            self._current_detections.mask = [m for m in masks if m is not None]

        return objects

    @staticmethod
    def _serialize_mask(mask: np.ndarray) -> str:
        """Serialize numpy mask to base64 string."""
        return base64.b64encode(mask.tobytes()).decode("utf-8")

    def _create_supervision_detections(self, results, objects: List[Dict], track_ids: Optional[np.ndarray] = None):
        """Create supervision detections from YOLO results."""
        if not results[0].boxes:
            self._current_detections = None
            return

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)

        if track_ids is not None:
            detections.tracker_id = track_ids  # 🔥 wichtig: Track-IDs übernehmen

        self._current_detections = detections
        # self._current_detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
        self._current_labels = np.array([objects[i]["label"] for i in range(len(objects))])

    def _create_supervision_detections_from_results(self, results: Dict, labels, track_ids: Optional[np.ndarray] = None):
        """Create supervision detections from transformer results."""
        xyxy = results["boxes"].cpu().detach().numpy()
        conf = results["scores"].cpu().detach().numpy()

        # Create class IDs
        if self._model_id == "owlv2":
            cls = results["labels"].cpu().detach().numpy()
        else:  # grounding_dino
            cls = ObjectDetector._convert_labels_to_class_ids(labels)

        detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
        if track_ids is not None:
            detections.tracker_id = track_ids
        self._current_detections = detections
        # self._current_detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
        self._current_labels = np.array([str(label) for label in labels])

    def _extract_owlv2_labels(self, results) -> np.ndarray:
        """Extract text labels for OWL-V2 results."""
        labels = []
        for label_idx in results[0]["labels"]:
            label_text = self._object_labels[0][label_idx.item()]
            labels.append(label_text)
        return np.array(labels)

    @staticmethod
    def _convert_labels_to_class_ids(labels: List[str]) -> np.ndarray:
        """Convert string labels to class IDs for Grounding-DINO."""
        class_ids = []
        for label in labels:
            try:
                # This is a simplified approach - you might need to adjust
                # based on your specific label structure
                class_id = hash(label.lower()) % 100  # Simple hash-based ID
                class_ids.append(class_id)
            except Exception:
                class_ids.append(0)  # Default class ID
        return np.array(class_ids)

    def _publish_detections(self, objects: List[Dict], method: str):
        """Publish detections to Redis."""
        if not objects or self._redis_broker is None:
            return

        metadata = {
            "timestamp": time.time(),
            "object_count": len(objects),
            "detection_method": method,
            "model_id": self._model_id,
        }

        try:
            self._redis_broker.publish_objects(objects, metadata)
        except Exception as e:
            redis_error = handle_redis_error("publish", self._redis_host, self._redis_port, e)
            if self.verbose:
                self._logger.warning(str(redis_error))

    def _load_model(self, model_id: str) -> Tuple[any, any]:
        """Load the specified model and processor."""
        if "yoloe" in model_id.lower():
            return self._load_yoloe_model()
        elif model_id == "yolo-world":
            return self._load_yolo_model()
        elif model_id == "owlv2":
            return self._load_owlv2_model()
        elif model_id == "grounding_dino":
            return self._load_grounding_dino_model()
        else:
            raise ModelLoadError(model_id, "Unknown model type")

    def _load_yolo_model(self):
        """Load YOLO-World model."""
        model_config = MODEL_CONFIGS["yolo-world"]
        model_path = model_config.model_params["model_path"]
        model = YOLO(model_path)
        model.set_classes(self._object_labels[0])
        return model, None

    def _load_yoloe_model(self):
        """Load YOLOE model."""
        model_config = MODEL_CONFIGS[self._model_id]
        model_path = model_config.model_params["model_path"]

        # Initialize YOLOE model
        model = YOLOE(model_path)

        # Check if prompt-free variant
        is_prompt_free = model_config.model_params.get("is_prompt_free", False)

        if not is_prompt_free and model_config.model_params.get("supports_prompts", True):
            # Set classes for prompted models (text prompts)
            # This allows open-vocabulary detection
            model.set_classes(self._object_labels[0], model.get_text_pe(self._object_labels[0]))
            if self.verbose:
                self._logger.info(f"YOLOE model loaded with {len(self._object_labels[0])} text prompts")
        else:
            # Prompt-free models use internal vocabulary (1200+ classes)
            if self.verbose:
                self._logger.info("YOLOE prompt-free model loaded with internal vocabulary")

        return model, None

    def _load_owlv2_model(self):
        """Load OWL-V2 model."""
        model_config = MODEL_CONFIGS["owlv2"]
        model_path = model_config.model_params["model_path"]
        processor = Owlv2Processor.from_pretrained(model_path)
        model = Owlv2ForObjectDetection.from_pretrained(model_path).to(self._device)
        return model, processor

    def _load_grounding_dino_model(self):
        """Load Grounding-DINO model."""
        model_config = MODEL_CONFIGS["grounding_dino"]
        model_path = model_config.model_params["model_path"]
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(self._device)
        return model, processor

    # Public API methods
    def add_label(self, label: str):
        """Add a new detectable object label."""
        self._object_labels[0].append(label.lower())

        if "yoloe" in self._model_id.lower():
            # Check if it's a prompt-free variant
            model_config = MODEL_CONFIGS.get(self._model_id)
            if model_config and not model_config.model_params.get("is_prompt_free", False):
                # Re-set classes for prompted YOLOE models
                self._model.set_classes(self._object_labels[0], self._model.get_text_pe(self._object_labels[0]))
                if self.verbose:
                    self._logger.info(f"Updated YOLOE classes with new label: {label}")
        elif self._model_id == "grounding_dino":
            self._processed_labels = ObjectDetector._preprocess_labels(self._object_labels, self._model_id)
        elif self._model_id == "yolo-world":
            self._model.set_classes(self._object_labels[0])

    def get_detections(self) -> Optional[sv.Detections]:
        """Get current supervision detections."""
        return self._current_detections

    # def get_label_texts(self) -> Optional[np.ndarray]:
    #     """Get current detection labels."""
    #     return self._current_labels

    def get_label_texts(self) -> Optional[np.ndarray]:
        """Get current detection labels (with track IDs if available)."""
        if self._current_detections is None or self._current_labels is None:
            return None

        labels = []
        has_tracking = hasattr(self._current_detections, "tracker_id") and self._current_detections.tracker_id is not None

        for i, label in enumerate(self._current_labels):
            conf = None
            if hasattr(self._current_detections, "confidence") and self._current_detections.confidence is not None:
                conf = self._current_detections.confidence[i]

            text = str(label)

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
        """Get the current computation device."""
        return self._device

    def get_model_id(self) -> str:
        """Get the current model identifier."""
        return self._model_id

    # Backward compatibility methods (deprecated)
    def detections(self) -> Optional[sv.Detections]:
        """Deprecated: use get_detections() instead."""
        return self.get_detections()

    def label_texts(self) -> Optional[np.ndarray]:
        """Deprecated: use get_label_texts() instead."""
        return self.get_label_texts()

    def object_labels(self) -> List[List[str]]:
        """Deprecated: use get_object_labels() instead."""
        return self.get_object_labels()
