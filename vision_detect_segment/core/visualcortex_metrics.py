"""
vision_detect_segment/core/visualcortex_metrics.py

Metrics integration for VisualCortex class.
Adds comprehensive monitoring and observability.
"""

from typing import Dict, Any, Optional
import time

from ..utils.metrics import MetricsRegistry, get_default_registry


class VisionMetrics:
    """
    Metrics collector for vision detection system.

    Provides comprehensive metrics for:
    - Frame processing
    - Detection performance
    - Redis operations
    - Object tracking
    - Model inference
    - Errors and failures
    """

    def __init__(self, registry: Optional[MetricsRegistry] = None, model_id: str = "unknown"):
        """
        Initialize metrics collector.

        Args:
            registry: Optional custom metrics registry
            model_id: Model identifier for labeling
        """
        self.registry = registry or get_default_registry()
        self.model_id = model_id

        # Frame processing metrics
        self.frames_processed = self.registry.register_counter(
            "frames_processed_total", "Total number of frames processed", labels=["model", "status"]
        )

        self.frames_dropped = self.registry.register_counter(
            "frames_dropped_total", "Total number of frames dropped due to backpressure", labels=["model", "reason"]
        )

        # Detection metrics
        self.detections_total = self.registry.register_counter(
            "detections_total", "Total number of objects detected", labels=["model", "class"]
        )

        self.detection_latency = self.registry.register_histogram(
            "detection_latency_seconds",
            "Time taken for object detection",
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            labels=["model", "stage"],
        )

        self.detection_confidence = self.registry.register_histogram(
            "detection_confidence",
            "Confidence scores of detections",
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
            labels=["model", "class"],
        )

        # Redis metrics
        self.redis_operations = self.registry.register_counter(
            "redis_operations_total", "Total Redis operations", labels=["operation", "status"]
        )

        self.redis_errors = self.registry.register_counter(
            "redis_errors_total", "Total Redis errors", labels=["operation", "error_type"]
        )

        self.redis_latency = self.registry.register_histogram(
            "redis_operation_duration_seconds",
            "Duration of Redis operations",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
            labels=["operation"],
        )

        # Tracking metrics
        self.active_tracks = self.registry.register_gauge(
            "active_tracks", "Number of currently active object tracks", labels=["model"]
        )

        self.track_lifetime = self.registry.register_summary(
            "track_lifetime_seconds", "Lifetime of object tracks", quantiles=[0.5, 0.9, 0.95, 0.99], labels=["model", "class"]
        )

        self.tracks_created = self.registry.register_counter(
            "tracks_created_total", "Total number of tracks created", labels=["model"]
        )

        self.tracks_lost = self.registry.register_counter(
            "tracks_lost_total", "Total number of tracks lost", labels=["model", "reason"]
        )

        # Model inference metrics
        self.model_inference_time = self.registry.register_histogram(
            "model_inference_duration_seconds",
            "Model inference duration",
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
            labels=["model", "device"],
        )

        self.model_batch_size = self.registry.register_histogram(
            "model_batch_size", "Batch size for model inference", buckets=[1, 2, 4, 8, 16, 32, 64], labels=["model"]
        )

        # Segmentation metrics
        self.segmentation_operations = self.registry.register_counter(
            "segmentation_operations_total", "Total segmentation operations", labels=["model", "status"]
        )

        self.segmentation_time = self.registry.register_histogram(
            "segmentation_duration_seconds",
            "Segmentation duration",
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            labels=["segmenter"],
        )

        # System metrics
        self.memory_usage = self.registry.register_gauge("memory_usage_bytes", "Memory usage", labels=["type"])

        self.gpu_memory_usage = self.registry.register_gauge("gpu_memory_usage_bytes", "GPU memory usage", labels=["device"])

        # Error metrics
        self.errors_total = self.registry.register_counter(
            "errors_total", "Total errors encountered", labels=["component", "error_type"]
        )

        # Processing pipeline metrics
        self.pipeline_stage_duration = self.registry.register_histogram(
            "pipeline_stage_duration_seconds",
            "Duration of pipeline stages",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            labels=["stage"],
        )

        # Queue metrics (for async processing)
        self.queue_size = self.registry.register_gauge("queue_size", "Current queue size", labels=["queue_type"])

        self.queue_wait_time = self.registry.register_histogram(
            "queue_wait_duration_seconds",
            "Time items spend in queue",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            labels=["queue_type"],
        )

        # Annotation metrics
        self.annotations_created = self.registry.register_counter(
            "annotations_created_total", "Total annotations created", labels=["type"]
        )

        self.annotation_errors = self.registry.register_counter(
            "annotation_errors_total", "Total annotation errors", labels=["type", "error"]
        )

    def record_frame_processed(self, success: bool = True):
        """Record a processed frame."""
        status = "success" if success else "failure"
        self.frames_processed.inc(labels={"model": self.model_id, "status": status})

    def record_frame_dropped(self, reason: str = "backpressure"):
        """Record a dropped frame."""
        self.frames_dropped.inc(labels={"model": self.model_id, "reason": reason})

    def record_detection(self, class_name: str, confidence: float):
        """Record a detection."""
        self.detections_total.inc(labels={"model": self.model_id, "class": class_name})
        self.detection_confidence.observe(confidence, labels={"model": self.model_id, "class": class_name})

    def record_detection_latency(self, duration: float, stage: str = "total"):
        """Record detection latency."""
        self.detection_latency.observe(duration, labels={"model": self.model_id, "stage": stage})

    def record_redis_operation(self, operation: str, duration: float, success: bool = True):
        """Record a Redis operation."""
        status = "success" if success else "failure"
        self.redis_operations.inc(labels={"operation": operation, "status": status})

        if success:
            self.redis_latency.observe(duration, labels={"operation": operation})

    def record_redis_error(self, operation: str, error_type: str):
        """Record a Redis error."""
        self.redis_errors.inc(labels={"operation": operation, "error_type": error_type})

    def update_active_tracks(self, count: int):
        """Update active tracks gauge."""
        self.active_tracks.set(count, labels={"model": self.model_id})

    def record_track_created(self):
        """Record a new track creation."""
        self.tracks_created.inc(labels={"model": self.model_id})

    def record_track_lost(self, class_name: str, lifetime: float, reason: str = "lost"):
        """Record a lost track."""
        self.tracks_lost.inc(labels={"model": self.model_id, "reason": reason})
        self.track_lifetime.observe(lifetime, labels={"model": self.model_id, "class": class_name})

    def record_model_inference(self, duration: float, device: str, batch_size: int = 1):
        """Record model inference metrics."""
        self.model_inference_time.observe(duration, labels={"model": self.model_id, "device": device})
        self.model_batch_size.observe(batch_size, labels={"model": self.model_id})

    def record_segmentation(self, duration: float, segmenter: str, success: bool = True):
        """Record segmentation operation."""
        status = "success" if success else "failure"
        self.segmentation_operations.inc(labels={"model": segmenter, "status": status})

        if success:
            self.segmentation_time.observe(duration, labels={"segmenter": segmenter})

    def update_memory_usage(self, rss_bytes: float, vms_bytes: float):
        """Update memory usage metrics."""
        self.memory_usage.set(rss_bytes, labels={"type": "rss"})
        self.memory_usage.set(vms_bytes, labels={"type": "vms"})

    def update_gpu_memory(self, device: str, used_bytes: float):
        """Update GPU memory usage."""
        self.gpu_memory_usage.set(used_bytes, labels={"device": device})

    def record_error(self, component: str, error_type: str):
        """Record an error."""
        self.errors_total.inc(labels={"component": component, "error_type": error_type})

    def record_pipeline_stage(self, stage: str, duration: float):
        """Record pipeline stage duration."""
        self.pipeline_stage_duration.observe(duration, labels={"stage": stage})

    def update_queue_size(self, queue_type: str, size: int):
        """Update queue size."""
        self.queue_size.set(size, labels={"queue_type": queue_type})

    def record_queue_wait(self, queue_type: str, duration: float):
        """Record queue wait time."""
        self.queue_wait_time.observe(duration, labels={"queue_type": queue_type})

    def record_annotation(self, annotation_type: str, success: bool = True, error: str = ""):
        """Record annotation creation."""
        if success:
            self.annotations_created.inc(labels={"type": annotation_type})
        else:
            self.annotation_errors.inc(labels={"type": annotation_type, "error": error})

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "frames_processed": self.frames_processed.get(labels={"model": self.model_id, "status": "success"}),
            "frames_failed": self.frames_processed.get(labels={"model": self.model_id, "status": "failure"}),
            "frames_dropped": self.frames_dropped.get(labels={"model": self.model_id, "reason": "backpressure"}),
            "active_tracks": self.active_tracks.get(labels={"model": self.model_id}),
            "detection_stats": self.detection_latency.get_statistics(labels={"model": self.model_id, "stage": "total"}),
            "redis_operations": {
                "publish_success": self.redis_operations.get(labels={"operation": "publish", "status": "success"}),
                "publish_failure": self.redis_operations.get(labels={"operation": "publish", "status": "failure"}),
            },
        }


class MetricsTimer:
    """Context manager for timing operations and recording metrics."""

    def __init__(self, metrics: VisionMetrics, metric_callback: callable, **kwargs):
        """
        Initialize metrics timer.

        Args:
            metrics: VisionMetrics instance
            metric_callback: Callback to record the metric (receives duration and kwargs)
            **kwargs: Additional arguments to pass to callback
        """
        self.metrics = metrics
        self.metric_callback = metric_callback
        self.kwargs = kwargs
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time

        # Call the metric callback with duration
        if exc_type is None:
            # Success
            self.metric_callback(duration, **self.kwargs)
        else:
            # Failure - you might want to handle this differently
            self.kwargs["success"] = False
            if hasattr(self.metric_callback, "__name__"):
                # Try to record as error
                if "error" not in self.kwargs:
                    self.kwargs["error"] = exc_type.__name__

        return False  # Don't suppress exceptions


def with_metrics(metrics_func: str):
    """
    Decorator for automatically recording method execution metrics.

    Args:
        metrics_func: Name of the VisionMetrics method to call

    Example:
        @with_metrics("record_detection_latency")
        def detect_objects(self, image):
            ...
    """

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = None
            error = None

            try:
                result = func(self, *args, **kwargs)
                return result
            except Exception as e:
                error = e
                raise
            finally:
                duration = time.time() - start_time

                # Call the metrics method
                if hasattr(self, "_metrics") and hasattr(self._metrics, metrics_func):
                    metric_method = getattr(self._metrics, metrics_func)

                    try:
                        # Try to call with duration and success status
                        metric_method(duration=duration, success=(error is None))
                    except TypeError:
                        # Fall back to just duration
                        try:
                            metric_method(duration)
                        except Exception:
                            pass  # Ignore metrics errors

        return wrapper

    return decorator
