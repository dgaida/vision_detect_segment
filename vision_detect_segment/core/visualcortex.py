import numpy as np
from typing import List, Optional, Dict, Any
import torch
import supervision as sv
import cv2
import copy
import threading
import time

from .object_detector import ObjectDetector
from ..utils.config import VisionConfig, get_default_config
from ..utils.exceptions import ImageProcessingError, DetectionError, handle_redis_error, handle_detection_error
from ..utils.utils import (
    setup_logging,
    get_optimal_device,
    Timer,
    validate_image,
    resize_image,
    format_detection_results,
    clear_gpu_cache,
)

from redis_robot_comm import RedisImageStreamer
from redis_robot_comm import RedisLabelManager


class VisualCortex:
    """
    A class for handling object detection and segmentation in a robot's workspace.

    This class integrates object detection models and provides functionality for
    annotating images, detecting objects, and managing the visual processing pipeline.
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
        """
        # Private attributes
        self._objdetect_model_id = objdetect_model_id
        self._device = get_optimal_device(prefer_gpu=(device != "cpu"))
        self._stream_name = stream_name
        self._annotated_stream_name = annotated_stream_name
        self._publish_annotated = publish_annotated
        self._img_work = None
        self._annotated_frame = None
        self._detected_objects = []
        self._processed_frames = 0
        self._label_annotator = None
        self._corner_annotator = None
        self._halo_annotator = None
        self._object_detector = None
        self._streamer = None
        self._annotated_streamer = None

        # Public configuration
        self.verbose = verbose
        self._config = config or get_default_config(objdetect_model_id)

        # Setup logging
        self._logger = setup_logging(verbose)

        try:
            if verbose:
                self._logger.info(f"Using device: {self._device}")
                self._logger.info(f"PyTorch version: {torch.__version__}")
                self._logger.info(f"CUDA available: {torch.cuda.is_available()}")

            # Initialize components
            with Timer("Initializing VisualCortex", self._logger if verbose else None):
                self._initialize_redis_streamer()
                self._initialize_annotated_streamer()
                self._setup_annotators()
                self._initialize_object_detector()

            if verbose:
                self._logger.info("VisualCortex initialization completed successfully")

        except Exception as e:
            error_msg = f"VisualCortex initialization failed: {e}"
            self._logger.error(error_msg)
            raise DetectionError(error_msg)

        # Add label manager for publishing detectable objects
        self._label_manager = RedisLabelManager()

        # Publish initial labels
        self._publish_current_labels()

        # Start label monitoring thread
        self._label_monitor_thread = None
        self._label_monitor_stop = threading.Event()
        if config and hasattr(config, "enable_label_monitoring") and config.enable_label_monitoring:
            self._start_label_monitoring()

    def _initialize_redis_streamer(self):
        """Initialize Redis image streamer for input images."""
        try:
            self._streamer = RedisImageStreamer(stream_name=self._stream_name)
            if self.verbose:
                self._logger.info(f"Initialized input streamer: {self._stream_name}")
        except Exception as e:
            redis_error = handle_redis_error("initialization", self._config.redis.host, self._config.redis.port, e)
            if self.verbose:
                self._logger.warning(f"Redis streamer initialization failed: {redis_error}")
            self._streamer = None

    def _initialize_annotated_streamer(self):
        """Initialize Redis image streamer for annotated images."""
        if not self._publish_annotated:
            self._annotated_streamer = None
            return

        try:
            self._annotated_streamer = RedisImageStreamer(stream_name=self._annotated_stream_name)
            if self.verbose:
                self._logger.info(f"Initialized annotated streamer: {self._annotated_stream_name}")
        except Exception as e:
            redis_error = handle_redis_error("initialization", self._config.redis.host, self._config.redis.port, e)
            if self.verbose:
                self._logger.warning(f"Annotated streamer initialization failed: {redis_error}")
            self._annotated_streamer = None

    def _setup_annotators(self):
        """Initialize supervision library annotators."""
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
            # Create minimal annotators as fallback
            self._label_annotator = sv.LabelAnnotator()
            self._corner_annotator = sv.BoxCornerAnnotator()
            self._halo_annotator = sv.HaloAnnotator()

    def _initialize_object_detector(self):
        """Initialize the object detection model."""
        object_labels = self._config.get_object_labels()

        self._object_detector = ObjectDetector(
            device=self._device,
            model_id=self._objdetect_model_id,
            object_labels=object_labels,
            verbose=self.verbose,
            config=self._config,
        )

    def detect_objects_from_redis(self) -> bool:
        """
        Manually trigger object detection from latest Redis image.

        Returns:
            bool: True if detection was successful, False otherwise
        """
        if self._streamer is None:
            if self.verbose:
                self._logger.error("Redis streamer not available")
            return False

        try:
            result = self._streamer.get_latest_image()
            if not result:
                if self.verbose:
                    self._logger.info("No image available from Redis")
                return False

            image, metadata = result

            self.process_image_callback(image, metadata, None)
            return True

        except ImageProcessingError as e:
            if self.verbose:
                self._logger.error(f"Image processing error: {e}")
            return False
        except Exception as e:
            if self.verbose:
                self._logger.error(f"Error in manual detection: {e}")
            return False

    def process_image_callback(self, image: np.ndarray, metadata: Dict[str, Any], image_info: Optional[Dict[str, Any]] = None):
        """
        Process incoming images from Redis stream.

        Args:
            image: Input image
            metadata: Image metadata from Redis
            image_info: Optional image dimension info
        """
        workspace_id = metadata.get("workspace_id", "unknown")
        frame_id = metadata.get("frame_id", self._processed_frames)

        try:
            # Validate input image
            validate_image(image)

            if self.verbose and image_info:
                width, height = image_info["width"], image_info["height"]
                self._logger.info(f"Processing frame {frame_id}: {width}x{height} from {workspace_id}")

            # Store current image
            self._img_work = image

            # Convert to RGB for detection models
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Run detection with timing
            with Timer("Object detection", self._logger if self.verbose else None):
                detected_objects = self._run_detection(image_rgb)

            # Create annotated frame with timing
            with Timer("Annotation", self._logger if self.verbose else None):
                self._create_annotated_frame(detected_objects)

            # Publish annotated frame to Redis
            if self._publish_annotated and self._annotated_frame is not None:
                self._publish_annotated_frame(metadata, frame_id)

            # Update state
            self._detected_objects = detected_objects
            self._processed_frames += 1

            if self.verbose:
                if detected_objects:
                    self._logger.info(f"Found {len(detected_objects)} objects")
                    if self.verbose:  # Extra detail in verbose mode
                        summary = format_detection_results(detected_objects, max_items=5)
                        self._logger.debug(summary)
                else:
                    self._logger.info("No objects detected")

        except ImageProcessingError as e:
            if self.verbose:
                self._logger.error(f"Image processing failed: {e}")
        except Exception as e:
            detection_error = handle_detection_error(e, image.shape, self._objdetect_model_id)
            if self.verbose:
                self._logger.error(str(detection_error))

    def _publish_annotated_frame(self, original_metadata: Dict[str, Any], frame_id: int):
        """
        Publish annotated frame to Redis.

        Args:
            original_metadata: Metadata from the original image
            frame_id: Frame identifier
        """
        if self._annotated_streamer is None:
            return

        try:
            # Create metadata for annotated frame
            annotated_metadata = {
                **original_metadata,
                "frame_id": frame_id,
                "annotated": True,
                "detection_count": len(self._detected_objects),
                "model_id": self._objdetect_model_id,
            }

            # Publish to Redis
            with Timer("Publishing annotated frame", self._logger if self.verbose else None):
                stream_id = self._annotated_streamer.publish_image(
                    self._annotated_frame, metadata=annotated_metadata, compress_jpeg=True, quality=85, maxlen=10
                )

            if self.verbose:
                self._logger.debug(f"Published annotated frame {frame_id} to Redis: {stream_id}")

        except Exception as e:
            redis_error = handle_redis_error("publish", self._config.redis.host, self._config.redis.port, e)
            if self.verbose:
                self._logger.warning(f"Failed to publish annotated frame: {redis_error}")

    def add_detectable_object(self, object_name: str):
        """
        Add a new object type to the detectable objects list and publish to Redis.

        Args:
            object_name: Name of the object to add
        """
        if self._object_detector:
            self._object_detector.add_label(object_name)

            # Publish updated labels to Redis
            self._publish_current_labels()

            if self.verbose:
                self._logger.info(f"Added new detectable object: {object_name}")

    def enable_annotated_publishing(self, enable: bool = True):
        """
        Enable or disable publishing of annotated frames to Redis.

        Args:
            enable: Whether to enable annotated frame publishing
        """
        self._publish_annotated = enable
        if enable and self._annotated_streamer is None:
            self._initialize_annotated_streamer()
        if self.verbose:
            status = "enabled" if enable else "disabled"
            self._logger.info(f"Annotated frame publishing {status}")

    # Properties with proper encapsulation
    def get_current_image(self, resize: bool = True) -> Optional[np.ndarray]:
        """
        Get current raw image.

        Args:
            resize: Whether to resize small images

        Returns:
            Current image or None if no image available
        """
        if self._img_work is None:
            return None

        if resize and self._img_work.shape[0] < 640:
            try:
                resized, _, _ = resize_image(self._img_work, scale_factor=self._config.annotation.resize_scale_factor)
                return resized
            except ImageProcessingError as e:
                if self.verbose:
                    self._logger.warning(f"Image resize failed: {e}")
                return self._img_work

        return self._img_work

    def get_annotated_image(self) -> Optional[np.ndarray]:
        """Get current annotated image."""
        return self._annotated_frame

    def get_detected_objects(self) -> List[Dict]:
        """Get list of detected objects (returns copy to prevent external modification)."""
        return copy.deepcopy(self._detected_objects)

    def get_object_labels(self) -> List[List[str]]:
        """Get list of detectable object labels."""
        if self._object_detector:
            return self._object_detector.get_object_labels()
        return [[]]

    def get_processed_frames_count(self) -> int:
        """Get number of processed frames."""
        return self._processed_frames

    def get_device(self) -> str:
        """Get current computation device."""
        return self._device

    def get_annotated_stream_name(self) -> str:
        """Get the name of the annotated image stream."""
        return self._annotated_stream_name

    def is_annotated_publishing_enabled(self) -> bool:
        """Check if annotated frame publishing is enabled."""
        return self._publish_annotated and self._annotated_streamer is not None

    # Private methods
    def _run_detection(self, image_rgb: np.ndarray) -> List[Dict]:
        """
        Run object detection on RGB image.

        Args:
            image_rgb: Input image in RGB format (required for detection models)
        """
        if image_rgb is None or self._object_detector is None:
            return []

        try:
            return self._object_detector.detect_objects(image_rgb)
        except Exception as e:
            if self.verbose:
                self._logger.error(f"Detection failed: {e}")
            return []

    def _create_annotated_frame(self, detected_objects: List[Dict]):
        """Create annotated version of current image."""
        if self._img_work is None:
            self._annotated_frame = None
            return

        try:
            if not detected_objects or self._object_detector is None:
                # No detections, just resize the image
                self._annotated_frame, _, _ = resize_image(
                    self._img_work.copy(), scale_factor=self._config.annotation.resize_scale_factor
                )
                return

            detections = self._object_detector.get_detections()
            if detections is None:
                self._annotated_frame = self._img_work.copy()
                return

            # Start with base image
            annotated_frame = self._img_work.copy()

            # Apply halo annotation if masks are available
            if hasattr(detections, "mask") and detections.mask is not None and len(detections.mask) > 0:
                try:
                    annotated_frame = self._halo_annotator.annotate(scene=annotated_frame, detections=detections)
                except Exception as e:
                    if self.verbose:
                        self._logger.warning(f"Halo annotation failed: {e}")

            # Resize for display
            resized_frame, scale_x, scale_y = resize_image(
                annotated_frame, scale_factor=self._config.annotation.resize_scale_factor
            )

            # Scale detection coordinates for display
            scaled_detections = self._scale_detections(detections, scale_x, scale_y)

            # Add bounding boxes
            if self._config.annotation.show_labels:
                try:
                    resized_frame = self._corner_annotator.annotate(scene=resized_frame, detections=scaled_detections)
                except Exception as e:
                    if self.verbose:
                        self._logger.warning(f"Corner annotation failed: {e}")

            # Add labels
            if self._config.annotation.show_labels:
                try:
                    labels = self._object_detector.get_label_texts()
                    if labels is not None:
                        resized_frame = self._label_annotator.annotate(
                            scene=resized_frame, detections=scaled_detections, labels=labels
                        )
                except Exception as e:
                    if self.verbose:
                        self._logger.warning(f"Label annotation failed: {e}")

            self._annotated_frame = resized_frame

        except Exception as e:
            if self.verbose:
                self._logger.error(f"Annotation creation failed: {e}")
            # Fallback to original image
            try:
                self._annotated_frame, _, _ = resize_image(
                    self._img_work.copy(), scale_factor=self._config.annotation.resize_scale_factor
                )
            except ImageProcessingError:
                self._annotated_frame = self._img_work.copy()

    def _scale_detections(self, detections: sv.Detections, scale_x: float, scale_y: float) -> sv.Detections:
        """Scale detection coordinates for resized image."""
        try:
            scaled_xyxy = copy.deepcopy(detections.xyxy)
            scaled_xyxy[:, [0, 2]] *= scale_x  # Scale x-coordinates
            scaled_xyxy[:, [1, 3]] *= scale_y  # Scale y-coordinates

            return sv.Detections(xyxy=scaled_xyxy, confidence=detections.confidence, class_id=detections.class_id)
        except Exception as e:
            if self.verbose:
                self._logger.warning(f"Detection scaling failed: {e}")
            return detections

    def _publish_current_labels(self):
        """
        Publish current detectable object labels to Redis.
        Should be called on initialization and whenever labels change.
        """
        try:
            object_labels = self._config.get_object_labels()

            if object_labels and len(object_labels) > 0:
                # Flatten nested list structure
                labels_flat = object_labels[0] if isinstance(object_labels[0], list) else object_labels

                metadata = {
                    "model_id": self._objdetect_model_id,
                    "source": "vision_detect_segment",
                    "label_count": len(labels_flat),
                }

                self._label_manager.publish_labels(labels_flat, metadata)

                if self.verbose:
                    self._logger.info(f"Published {len(labels_flat)} labels to Redis")

        except Exception as e:
            if self.verbose:
                self._logger.error(f"Failed to publish labels: {e}")

    def _start_label_monitoring(self):
        """
        Start a background thread that monitors Redis for label updates
        and updates the object detector accordingly.
        """

        def monitor_labels():
            """Background thread function to check for label updates."""
            if self.verbose:
                self._logger.info("Starting label monitoring thread")

            last_check_time = time.time()
            check_interval = 1.0  # Check every second

            while not self._label_monitor_stop.is_set():
                try:
                    current_time = time.time()

                    # Only check periodically
                    if current_time - last_check_time >= check_interval:
                        # Get latest labels from Redis
                        new_labels = self._label_manager.get_latest_labels(timeout_seconds=5.0)

                        if new_labels is not None:
                            # Get current labels
                            current_labels = self.get_object_labels()
                            current_labels_flat = current_labels[0] if current_labels else []

                            # Check if labels have changed
                            if set(new_labels) != set(current_labels_flat):
                                if self.verbose:
                                    self._logger.info("Label update detected, updating detector")

                                # Update detector with new labels
                                self._update_detector_labels(new_labels)

                        last_check_time = current_time

                    # Sleep briefly to avoid busy waiting
                    time.sleep(0.1)

                except Exception as e:
                    if self.verbose:
                        self._logger.error(f"Error in label monitoring: {e}")
                    time.sleep(1.0)  # Wait longer on error

            if self.verbose:
                self._logger.info("Label monitoring thread stopped")

        self._label_monitor_thread = threading.Thread(target=monitor_labels, daemon=True)
        self._label_monitor_thread.start()

    def _update_detector_labels(self, new_labels: list):
        """
        Update the object detector with new labels.

        Args:
            new_labels: List of new label strings
        """
        try:
            # Update config
            self._config.set_object_labels(new_labels)

            # Update detector
            if self._object_detector:
                # Clear existing labels and add new ones
                for label in new_labels:
                    self._object_detector.add_label(label)

                if self.verbose:
                    self._logger.info(f"Updated detector with {len(new_labels)} labels")

        except Exception as e:
            if self.verbose:
                self._logger.error(f"Failed to update detector labels: {e}")

    # Utility methods
    def clear_cache(self):
        """Clear GPU cache and reset internal state."""
        clear_gpu_cache()
        if self.verbose:
            self._logger.info("GPU cache cleared")

    def cleanup(self):
        """
        Cleanup method to stop background threads.
        """
        if self._label_monitor_thread:
            self._label_monitor_stop.set()
            self._label_monitor_thread.join(timeout=2.0)

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        try:
            from .utils import get_memory_usage

            return get_memory_usage()
        except Exception as e:
            if self.verbose:
                self._logger.warning(f"Could not get memory usage: {e}")
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

    # Property compatibility methods (for backward compatibility)
    @property
    def current_image(self) -> Optional[np.ndarray]:
        """Get current raw image."""
        return self.get_current_image()

    @property
    def annotated_image(self) -> Optional[np.ndarray]:
        """Get current annotated image."""
        return self.get_annotated_image()

    @property
    def detected_objects(self) -> List[Dict]:
        """Get list of detected objects."""
        return self.get_detected_objects()

    @property
    def object_labels(self) -> List[List[str]]:
        """Get list of detectable object labels."""
        return self.get_object_labels()

    @property
    def processed_frames(self) -> int:
        """Get number of processed frames."""
        return self._processed_frames

    # Deprecated methods for backward compatibility
    def img_work(self, resize: bool = True) -> Optional[np.ndarray]:
        """Deprecated: use get_current_image() instead."""
        return self.get_current_image(resize)

    def annotated_frame(self) -> Optional[np.ndarray]:
        """Deprecated: use get_annotated_image() instead."""
        return self.get_annotated_image()
