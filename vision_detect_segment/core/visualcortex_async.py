"""
vision_detect_segment/core/visualcortex_async.py

Async-enabled version of VisualCortex with background processing,
Redis streaming, and proper queue management.
"""

import numpy as np
from typing import List, Optional, Dict, Any
import supervision as sv
import cv2
import copy
import threading
import time

from .async_processor import AsyncProcessor, ProcessingResult
from .object_detector import ObjectDetector
from ..utils.config import VisionConfig, get_default_config
from ..utils.exceptions import (
    DetectionError,
)
from ..utils.utils import (
    setup_logging,
    get_optimal_device,
    validate_image,
    resize_image,
    clear_gpu_cache,
)

from redis_robot_comm import RedisImageStreamer, RedisLabelManager


class VisualCortexAsync:
    """
    Async-enabled VisualCortex with background processing and Redis integration.

    Features:
    - Asynchronous image processing pipeline
    - Background Redis polling
    - Multi-threaded detection
    - Backpressure handling
    - Real-time statistics
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
        # Async-specific parameters
        num_workers: int = 2,
        max_queue_size: int = 100,
        enable_backpressure: bool = True,
        redis_poll_interval: float = 0.01,  # 100 Hz polling
    ):
        """
        Initialize async-enabled VisualCortex.

        Args:
            objdetect_model_id: Detection model identifier
            device: Computation device
            stream_name: Redis input stream
            annotated_stream_name: Redis output stream
            publish_annotated: Whether to publish annotated frames
            verbose: Enable verbose logging
            config: Optional configuration
            num_workers: Number of processing workers
            max_queue_size: Maximum processing queue size
            enable_backpressure: Enable adaptive backpressure
            redis_poll_interval: Redis polling interval in seconds
        """
        # Basic configuration
        self._objdetect_model_id = objdetect_model_id
        self._device = get_optimal_device(prefer_gpu=(device != "cpu"))
        self._stream_name = stream_name
        self._annotated_stream_name = annotated_stream_name
        self._publish_annotated = publish_annotated
        self._redis_poll_interval = redis_poll_interval

        # Async-specific settings
        self._num_workers = num_workers
        self._max_queue_size = max_queue_size
        self._enable_backpressure = enable_backpressure

        # State management
        self._img_work = None
        self._annotated_frame = None
        self._detected_objects = []
        self._latest_result: Optional[ProcessingResult] = None
        self._result_lock = threading.Lock()

        # Configuration
        self.verbose = verbose
        self._config = config or get_default_config(objdetect_model_id)
        self._logger = setup_logging(verbose)

        # Initialize components
        self._initialize_components()

        # Async processor
        self._processor = AsyncProcessor(
            process_func=self._process_image_sync,
            num_workers=num_workers,
            max_queue_size=max_queue_size,
            enable_backpressure=enable_backpressure,
            logger=self._logger,
        )

        # Background threads
        self._redis_thread: Optional[threading.Thread] = None
        self._result_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Label management
        self._label_manager = RedisLabelManager()
        self._publish_current_labels()

        if verbose:
            self._logger.info(
                f"VisualCortexAsync initialized: "
                f"workers={num_workers}, queue={max_queue_size}, "
                f"backpressure={enable_backpressure}"
            )

    def _initialize_components(self):
        """Initialize detection and streaming components."""
        try:
            # Redis streamers
            self._streamer = RedisImageStreamer(stream_name=self._stream_name)
            self._annotated_streamer = (
                RedisImageStreamer(stream_name=self._annotated_stream_name) if self._publish_annotated else None
            )

            # Annotators
            self._setup_annotators()

            # Object detector
            object_labels = self._config.get_object_labels()
            self._object_detector = ObjectDetector(
                device=self._device,
                model_id=self._objdetect_model_id,
                object_labels=object_labels,
                verbose=self.verbose,
                config=self._config,
            )

            if self.verbose:
                self._logger.info("Components initialized successfully")

        except Exception as e:
            self._logger.error(f"Component initialization failed: {e}")
            raise DetectionError(f"Initialization failed: {e}")

    def _setup_annotators(self):
        """Initialize supervision annotators."""
        try:
            ann_cfg = self._config.annotation
            self._label_annotator = sv.LabelAnnotator(
                text_position=sv.Position.BOTTOM_CENTER,
                text_scale=ann_cfg.text_scale,
                text_padding=ann_cfg.text_padding,
            )
            self._corner_annotator = sv.BoxCornerAnnotator(thickness=ann_cfg.box_thickness)
            self._halo_annotator = sv.HaloAnnotator()
        except Exception as e:
            self._logger.warning(f"Annotator setup failed: {e}")
            # Fallback to defaults
            self._label_annotator = sv.LabelAnnotator()
            self._corner_annotator = sv.BoxCornerAnnotator()
            self._halo_annotator = sv.HaloAnnotator()

    def start(self):
        """Start async processing pipeline."""
        if self._redis_thread and self._redis_thread.is_alive():
            self._logger.warning("Pipeline already running")
            return

        self._stop_event.clear()

        # Start processor workers
        self._processor.start()

        # Start Redis polling thread
        self._redis_thread = threading.Thread(target=self._redis_polling_loop, name="RedisPoller", daemon=True)
        self._redis_thread.start()

        # Start result handling thread
        self._result_thread = threading.Thread(target=self._result_handling_loop, name="ResultHandler", daemon=True)
        self._result_thread.start()

        if self.verbose:
            self._logger.info("Async pipeline started")

    def stop(self, timeout: float = 5.0):
        """
        Stop async processing pipeline.

        Args:
            timeout: Maximum wait time for graceful shutdown
        """
        self._logger.info("Stopping async pipeline...")
        self._stop_event.set()

        # Stop processor
        self._processor.stop(timeout=timeout)

        # Wait for threads
        if self._redis_thread:
            self._redis_thread.join(timeout=timeout)
        if self._result_thread:
            self._result_thread.join(timeout=timeout)

        if self.verbose:
            self._logger.info("Async pipeline stopped")

    def _redis_polling_loop(self):
        """Background thread for polling Redis stream."""
        self._logger.debug("Redis polling loop started")

        consecutive_failures = 0
        max_failures = 10

        while not self._stop_event.is_set():
            try:
                # Get latest image from Redis
                result = self._streamer.get_latest_image()

                if result:
                    image, metadata = result

                    # Submit for processing
                    priority = metadata.get("priority", 0)
                    accepted = self._processor.submit_task(
                        image=image,
                        metadata=metadata,
                        priority=priority,
                    )

                    if not accepted and self.verbose:
                        self._logger.debug("Frame dropped due to backpressure")

                    consecutive_failures = 0
                else:
                    # No image available - short sleep
                    time.sleep(self._redis_poll_interval * 2)

            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    self._logger.error(f"Redis polling failed {consecutive_failures} times: {e}")
                    break

                time.sleep(self._redis_poll_interval * 10)

            # Rate limiting
            time.sleep(self._redis_poll_interval)

        self._logger.debug("Redis polling loop stopped")

    def _result_handling_loop(self):
        """Background thread for handling processing results."""
        self._logger.debug("Result handling loop started")

        while not self._stop_event.is_set():
            try:
                # Get result from processor
                result = self._processor.get_result(timeout=0.1)

                if result:
                    self._handle_result(result)

            except Exception as e:
                self._logger.error(f"Result handling error: {e}")
                time.sleep(0.1)

        self._logger.debug("Result handling loop stopped")

    def _handle_result(self, result: ProcessingResult):
        """Handle a processing result."""
        try:
            # Update latest result
            with self._result_lock:
                self._latest_result = result
                self._detected_objects = result.detections
                self._annotated_frame = result.annotated_image

            # Publish annotated frame if available
            if self._publish_annotated and self._annotated_streamer and result.annotated_image is not None:

                metadata = {
                    **result.metadata,
                    "detection_count": len(result.detections),
                    "processing_time": result.processing_time,
                    "task_id": result.task_id,
                }

                self._annotated_streamer.publish_image(
                    result.annotated_image, metadata=metadata, compress_jpeg=True, quality=85, maxlen=10
                )

            if self.verbose and len(result.detections) > 0:
                self._logger.info(
                    f"Processed {result.task_id}: " f"{len(result.detections)} objects in " f"{result.processing_time:.3f}s"
                )

        except Exception as e:
            self._logger.error(f"Failed to handle result: {e}")

    def _process_image_sync(self, image: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous processing function called by async workers.

        Args:
            image: Input image
            metadata: Image metadata

        Returns:
            Dict with detection results and annotated image
        """
        try:
            # Validate image
            validate_image(image)

            # Store current image
            self._img_work = image

            # Convert to RGB for detection
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Run detection
            detected_objects = self._object_detector.detect_objects(image_rgb)

            # Create annotated frame
            annotated_frame = self._create_annotated_frame(image, detected_objects)

            return {
                "objects": detected_objects,
                "annotated_image": annotated_frame,
            }

        except Exception as e:
            self._logger.error(f"Image processing failed: {e}")
            return {
                "objects": [],
                "annotated_image": None,
                "error": str(e),
            }

    def _create_annotated_frame(self, image: np.ndarray, detected_objects: List[Dict]) -> Optional[np.ndarray]:
        """Create annotated visualization."""
        if image is None:
            return None

        try:
            if not detected_objects or self._object_detector is None:
                # No detections - just resize
                annotated, _, _ = resize_image(image.copy(), scale_factor=self._config.annotation.resize_scale_factor)
                return annotated

            detections = self._object_detector.get_detections()
            if detections is None:
                return image.copy()

            # Start with base image
            annotated = image.copy()

            # Apply halo for masks
            if hasattr(detections, "mask") and detections.mask is not None and len(detections.mask) > 0:
                try:
                    annotated = self._halo_annotator.annotate(scene=annotated, detections=detections)
                except Exception as e:
                    if self.verbose:
                        self._logger.warning(f"Halo annotation failed: {e}")

            # Resize for display
            resized, scale_x, scale_y = resize_image(annotated, scale_factor=self._config.annotation.resize_scale_factor)

            # Scale detections
            scaled_detections = self._scale_detections(detections, scale_x, scale_y)

            # Add bounding boxes
            if self._config.annotation.show_labels:
                try:
                    resized = self._corner_annotator.annotate(scene=resized, detections=scaled_detections)
                except Exception as e:
                    if self.verbose:
                        self._logger.warning(f"Corner annotation failed: {e}")

            # Add labels
            if self._config.annotation.show_labels:
                try:
                    labels = self._object_detector.get_label_texts()
                    if labels is not None:
                        resized = self._label_annotator.annotate(scene=resized, detections=scaled_detections, labels=labels)
                except Exception as e:
                    if self.verbose:
                        self._logger.warning(f"Label annotation failed: {e}")

            return resized

        except Exception as e:
            self._logger.error(f"Annotation failed: {e}")
            try:
                fallback, _, _ = resize_image(image.copy(), scale_factor=self._config.annotation.resize_scale_factor)
                return fallback
            except Exception as e2:
                print(e2)
                return image.copy()

    def _scale_detections(self, detections: sv.Detections, scale_x: float, scale_y: float) -> sv.Detections:
        """Scale detection coordinates."""
        try:
            scaled_xyxy = copy.deepcopy(detections.xyxy)
            scaled_xyxy[:, [0, 2]] *= scale_x
            scaled_xyxy[:, [1, 3]] *= scale_y

            return sv.Detections(xyxy=scaled_xyxy, confidence=detections.confidence, class_id=detections.class_id)
        except Exception as e:
            if self.verbose:
                self._logger.warning(f"Detection scaling failed: {e}")
            return detections

    def _publish_current_labels(self):
        """Publish current object labels to Redis."""
        try:
            object_labels = self._config.get_object_labels()
            if object_labels and len(object_labels) > 0:
                labels_flat = object_labels[0] if isinstance(object_labels[0], list) else object_labels

                metadata = {
                    "model_id": self._objdetect_model_id,
                    "source": "vision_detect_segment_async",
                    "label_count": len(labels_flat),
                }

                self._label_manager.publish_labels(labels_flat, metadata)

                if self.verbose:
                    self._logger.info(f"Published {len(labels_flat)} labels")
        except Exception as e:
            self._logger.error(f"Failed to publish labels: {e}")

    # Public API

    def get_detected_objects(self) -> List[Dict]:
        """Get latest detected objects (thread-safe)."""
        with self._result_lock:
            return copy.deepcopy(self._detected_objects)

    def get_annotated_image(self) -> Optional[np.ndarray]:
        """Get latest annotated image (thread-safe)."""
        with self._result_lock:
            if self._annotated_frame is not None:
                return self._annotated_frame.copy()
        return None

    def get_latest_result(self) -> Optional[ProcessingResult]:
        """Get latest processing result (thread-safe)."""
        with self._result_lock:
            return self._latest_result

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = self._processor.get_stats()

        with self._result_lock:
            stats["current_detections"] = len(self._detected_objects)
            stats["has_annotated_frame"] = self._annotated_frame is not None

        stats["model_id"] = self._objdetect_model_id
        stats["device"] = self._device
        stats["num_workers"] = self._num_workers

        return stats

    def add_detectable_object(self, object_name: str):
        """Add new detectable object label."""
        if self._object_detector:
            self._object_detector.add_label(object_name)
            self._publish_current_labels()

            if self.verbose:
                self._logger.info(f"Added detectable object: {object_name}")

    def clear_cache(self):
        """Clear GPU cache."""
        clear_gpu_cache()
        if self.verbose:
            self._logger.info("GPU cache cleared")

    def cleanup(self):
        """Cleanup resources."""
        self.stop()
        self.clear_cache()
