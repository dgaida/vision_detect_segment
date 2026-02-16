import copy
import gc
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import supervision as sv
from redis_robot_comm import RedisImageStreamer, RedisLabelManager

from ..utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from ..utils.config import VisionConfig, get_default_config
from ..utils.exceptions import (
    DetectionError,
)
from ..utils.redis_helpers import redis_operation
from ..utils.retry import retry_with_backoff
from ..utils.utils import (
    Timer,
    clear_gpu_cache,
    get_optimal_device,
    resize_image,
    setup_logging,
    validate_image,
)
from .object_detector import ObjectDetector


class VisualCortex:
    """
    A class for handling object detection and segmentation in a robot's workspace.

    This class integrates object detection models and provides functionality for
    annotating images, detecting objects, and managing the visual processing pipeline.
    It communicates with other components via Redis streams.
    """

    def __init__(
        self,
        objdetect_model_id: str,
        device: str = "auto",
        stream_name: str = "robot_camera",
        annotated_stream_name: str = "annotated_camera",
        publish_annotated: bool = True,
        verbose: bool = False,
        config: Optional[VisionConfig] = None,
        publish_detections_during_movement: bool = False,
    ):
        """
        Initialize the VisualCortex.

        Args:
            objdetect_model_id: Identifier for the object detection model
            device: Device to use ("auto", "cuda", "cpu")
            stream_name: Redis stream name for input image data
            annotated_stream_name: Redis stream name for annotated images
            publish_annotated: Whether to publish annotated images to Redis
            verbose: Enable verbose logging
            config: Optional VisionConfig instance
            publish_detections_during_movement: Whether to publish detections when robot moves
        """
        # Private attributes
        self._objdetect_model_id = objdetect_model_id
        self._device = get_optimal_device(prefer_gpu=(device != "cpu"))
        self._stream_name = stream_name
        self._annotated_stream_name = annotated_stream_name
        self._publish_annotated = publish_annotated
        self._img_work: Optional[np.ndarray] = None
        self._annotated_frame: Optional[np.ndarray] = None
        self._detected_objects: List[Dict[str, Any]] = []
        self._processed_frames = 0
        self._publish_detections_during_movement = publish_detections_during_movement
        self._verbose = verbose
        self._streamer: Optional[RedisImageStreamer] = None
        self._annotated_streamer: Optional[RedisImageStreamer] = None

        # Config initialization
        self._config = config or get_default_config(objdetect_model_id)

        # Setup logging
        self._logger = setup_logging(verbose)

        # Reliability patterns
        self._redis_circuit_breaker = CircuitBreaker(
            failure_threshold=self._config.redis.retry_attempts, recovery_timeout=60.0, expected_exception=Exception
        )

        try:
            if verbose:
                self._logger.info(f"Initializing VisualCortex on device: {self._device}")

            # Initialize components
            with Timer("Initializing VisualCortex", self._logger if verbose else None):
                self._initialize_redis_streamers()
                self._setup_annotators()
                self._initialize_object_detector()

            # Add label manager
            self._label_manager = RedisLabelManager()
            self._publish_current_labels()

            # Start label monitoring thread
            self._label_monitor_thread: Optional[threading.Thread] = None
            self._label_monitor_stop = threading.Event()
            self._start_label_monitoring()

            if verbose:
                self._logger.info("VisualCortex initialization completed successfully")

        except Exception as e:
            error_msg = f"VisualCortex initialization failed: {e}"
            self._logger.error(error_msg)
            raise DetectionError(error_msg)

    def _initialize_redis_streamers(self) -> None:
        """Initialize both input and annotated image streamers with robust connection testing."""
        # Input image streamer
        with redis_operation(
            "initialization",
            self._config.redis.host,
            self._config.redis.port,
            self._logger,
            raise_on_error=self._config.redis.fail_on_error,
        ):
            self._streamer = RedisImageStreamer(
                host=self._config.redis.host,
                port=self._config.redis.port,
                password=self._config.redis.password,
                ssl=self._config.redis.ssl,
                stream_name=self._stream_name,
            )
            self._streamer.client.ping()
            if self._verbose:
                self._logger.info(f"✓ Initialized input streamer: {self._stream_name}")

        # Annotated image streamer
        if self._publish_annotated:
            try:
                self._annotated_streamer = RedisImageStreamer(
                    host=self._config.redis.host,
                    port=self._config.redis.port,
                    password=self._config.redis.password,
                    ssl=self._config.redis.ssl,
                    stream_name=self._annotated_stream_name,
                )
                self._annotated_streamer.client.ping()
                if self._verbose:
                    self._logger.info(f"✓ Initialized annotated streamer: {self._annotated_stream_name}")
            except Exception as e:
                if self._verbose:
                    self._logger.warning(f"Annotated streamer initialization failed: {e}")
                self._annotated_streamer = None
        else:
            self._annotated_streamer = None

    def _setup_annotators(self) -> None:
        """Initialize supervision library annotators for image visualization."""
        try:
            annotation_config = self._config.annotation
            self._label_annotator = sv.LabelAnnotator(
                text_position=sv.Position.BOTTOM_CENTER,
                text_scale=annotation_config.text_scale,
                text_padding=annotation_config.text_padding,
            )
            self._corner_annotator = sv.BoxCornerAnnotator(thickness=annotation_config.box_thickness)
            self._halo_annotator = sv.HaloAnnotator()
        except Exception as e:
            self._logger.warning(f"Annotation setup failed: {e}")
            self._label_annotator = sv.LabelAnnotator()
            self._corner_annotator = sv.BoxCornerAnnotator()
            self._halo_annotator = sv.HaloAnnotator()

    def _initialize_object_detector(self) -> None:
        """Initialize the underlying ObjectDetector backend."""
        self._object_detector = ObjectDetector(
            device=self._device,
            model_id=self._objdetect_model_id,
            object_labels=self._config.get_object_labels(),
            verbose=self._verbose,
            config=self._config,
            publish_during_movement=self._publish_detections_during_movement,
            redis_host=self._config.redis.host,
            redis_port=self._config.redis.port,
        )

    @retry_with_backoff(max_attempts=3, exceptions=(Exception,))
    def detect_objects_from_redis(self) -> bool:
        """
        Manually trigger object detection from latest image in Redis.

        Returns:
            bool: True if detection was successful, False otherwise
        """
        if self._streamer is None:
            return False

        try:
            result = self._redis_circuit_breaker.call(self._streamer.get_latest_image)
            if not result:
                return False

            image, metadata = result
            self.process_image_callback(image, metadata)
            return True
        except CircuitBreakerOpenError as e:
            if self._verbose:
                self._logger.warning(f"Circuit breaker open: {e}")
            return False
        except Exception as e:
            if self._verbose:
                self._logger.error(f"Error in manual detection: {e}")
            return False

    def process_image_callback(
        self, image: np.ndarray, metadata: Dict[str, Any], image_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Process incoming images from Redis stream.

        Args:
            image: Input image in BGR format
            metadata: Image metadata from Redis
            image_info: Optional image dimension info
        """
        try:
            validate_image(image)
            self._img_work = image

            # Convert to RGB for models
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detection
            with Timer("Object detection", self._logger if self._verbose else None):
                detected_objects = self._object_detector.detect_objects(image_rgb, metadata=metadata)

            # Annotation
            with Timer("Annotation", self._logger if self._verbose else None):
                self._create_annotated_frame(detected_objects)

            # Publishing
            if self._publish_annotated and self._annotated_frame is not None:
                self._publish_annotated_frame(metadata, metadata.get("frame_id", self._processed_frames))

            # State update
            self._detected_objects = detected_objects
            self._processed_frames += 1

        except Exception as e:
            if self._verbose:
                self._logger.error(f"Image processing failed: {e}")

    def _publish_annotated_frame(self, original_metadata: Dict[str, Any], frame_id: int) -> None:
        """Publish annotated frame to Redis."""
        if self._annotated_streamer is None or self._annotated_frame is None:
            return

        try:
            annotated_metadata = {
                **original_metadata,
                "frame_id": frame_id,
                "annotated": True,
                "detection_count": len(self._detected_objects),
                "model_id": self._objdetect_model_id,
            }
            self._redis_circuit_breaker.call(
                self._annotated_streamer.publish_image,
                self._annotated_frame,
                metadata=annotated_metadata,
                compress_jpeg=True,
                quality=85,
                maxlen=10,
            )
        except Exception as e:
            if self._verbose:
                self._logger.warning(f"Failed to publish annotated frame: {e}")

    def cleanup(self, force: bool = False) -> None:
        """
        Cleanup resources, stop threads and clear memory.

        Args:
            force: Whether to forcefully stop threads if they don't terminate gracefully.
        """
        if self._label_monitor_thread and self._label_monitor_thread.is_alive():
            self._label_monitor_stop.set()
            self._label_monitor_thread.join(timeout=2.0)
            if self._label_monitor_thread.is_alive() and force:
                self._logger.warning("Label monitor thread did not stop gracefully")

        self._img_work = None
        self._annotated_frame = None
        self._detected_objects = []

        if self._device == "cuda":
            clear_gpu_cache()

        gc.collect()
        if self._verbose:
            self._logger.info("VisualCortex resources cleaned up")

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup(force=True)
        except Exception:
            pass

    # Public API methods
    def get_object_labels(self) -> List[List[str]]:
        """Get current object labels."""
        return self._config.get_object_labels()

    def get_processed_frames_count(self) -> int:
        """Get processed frames count."""
        return self._processed_frames

    def get_device(self) -> str:
        """Get the current computation device."""
        return self._device

    def clear_cache(self) -> None:
        """Clear GPU cache and reset internal state."""
        clear_gpu_cache()

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        try:
            from ..utils.utils import get_memory_usage

            return get_memory_usage()
        except Exception:
            return {}

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "processed_frames": self._processed_frames,
            "device": self._device,
            "model_id": self._objdetect_model_id,
            "has_current_image": self._img_work is not None,
            "current_detections_count": len(self._detected_objects),
            "redis_available": self._streamer is not None,
            "annotated_streamer_available": self._annotated_streamer is not None,
            "annotated_publishing_enabled": self._publish_annotated,
            "detector_available": self._object_detector is not None,
        }

    def get_current_image(self, resize: bool = True) -> Optional[np.ndarray]:
        """Get current raw image, optionally resized."""
        if self._img_work is None:
            return None
        if resize and self._img_work.shape[0] < 640:
            resized, _, _ = resize_image(self._img_work, scale_factor=self._config.annotation.resize_scale_factor)
            return resized
        return self._img_work

    def get_annotated_image(self) -> Optional[np.ndarray]:
        return self._annotated_frame

    def get_detected_objects(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._detected_objects)

    def add_detectable_object(self, object_name: str) -> None:
        """Add a new detectable object and sync with Redis."""
        self._object_detector.add_label(object_name)
        self._publish_current_labels()

    def _create_annotated_frame(self, detected_objects: List[Dict[str, Any]]) -> None:
        """Apply all annotations to the current working image."""
        if self._img_work is None:
            return

        detections = self._object_detector.get_detections()
        if not detected_objects or detections is None:
            self._annotated_frame, _, _ = resize_image(
                self._img_work.copy(), scale_factor=self._config.annotation.resize_scale_factor
            )
            return

        annotated_frame = self._img_work.copy()

        # Halo for masks
        if hasattr(detections, "mask") and detections.mask is not None and len(detections.mask) > 0:
            annotated_frame = self._halo_annotator.annotate(scene=annotated_frame, detections=detections)

        # Resize for display
        resized_frame, scale_x, scale_y = resize_image(
            annotated_frame, scale_factor=self._config.annotation.resize_scale_factor
        )

        # Coordinate scaling
        scaled_xyxy = detections.xyxy.astype(np.float64) * [scale_x, scale_y, scale_x, scale_y]
        scaled_detections = sv.Detections(
            xyxy=scaled_xyxy,
            confidence=detections.confidence,
            class_id=detections.class_id,
            tracker_id=getattr(detections, "tracker_id", None),
        )

        # Boxes and Labels
        if self._config.annotation.show_labels:
            resized_frame = self._corner_annotator.annotate(scene=resized_frame, detections=scaled_detections)
            labels = self._object_detector.get_label_texts()
            if labels is not None:
                resized_frame = self._label_annotator.annotate(
                    scene=resized_frame, detections=scaled_detections, labels=labels
                )

        self._annotated_frame = resized_frame

    def _publish_current_labels(self) -> None:
        """Sync current labels with Redis."""
        try:
            labels = self._config.get_object_labels()[0]
            self._label_manager.publish_labels(labels, {"model_id": self._objdetect_model_id})
        except Exception as e:
            if self._verbose:
                self._logger.error(f"Failed to publish labels: {e}")

    def _start_label_monitoring(self) -> None:
        """Start thread to monitor label changes from Redis."""

        def monitor_labels():
            last_check = 0
            while not self._label_monitor_stop.is_set():
                if time.time() - last_check > 1.0:
                    try:
                        new_labels = self._label_manager.get_latest_labels(timeout_seconds=1.0)
                        if new_labels:
                            current = self._config.get_object_labels()[0]
                            if set(new_labels) != set(current):
                                self._update_detector_labels(new_labels)
                    except Exception:
                        pass
                    last_check = time.time()
                time.sleep(0.1)

        self._label_monitor_thread = threading.Thread(target=monitor_labels, daemon=True)
        self._label_monitor_thread.start()

    def _update_detector_labels(self, labels: List[str]) -> None:
        self._config.set_object_labels(labels)
        self._object_detector.add_label(labels[-1])  # Simplified for now

    # Properties and compatibility
    @property
    def current_image(self):
        return self.get_current_image()

    @property
    def annotated_image(self):
        return self.get_annotated_image()

    @property
    def detected_objects(self):
        return self.get_detected_objects()

    @property
    def object_labels(self):
        return self.get_object_labels()

    @property
    def processed_frames(self):
        return self._processed_frames

    def img_work(self, resize: bool = True) -> Optional[np.ndarray]:
        """Legacy method for backward compatibility."""
        return self.get_current_image(resize)

    def annotated_frame(self) -> Optional[np.ndarray]:
        """Legacy method for backward compatibility."""
        return self.get_annotated_image()
