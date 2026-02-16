"""
vision_detect_segment/utils/metrics_exporter.py

HTTP server for exporting metrics to Prometheus and other monitoring systems.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from .metrics import MetricsRegistry, get_default_registry


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for serving metrics."""

    registry: Optional[MetricsRegistry] = None

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/metrics":
            self.serve_prometheus_metrics()
        elif self.path == "/metrics.json":
            self.serve_json_metrics()
        elif self.path == "/health":
            self.serve_health()
        else:
            self.send_error(404)

    def serve_prometheus_metrics(self):
        """Serve metrics in Prometheus format."""
        try:
            registry = self.registry or get_default_registry()
            metrics_text = registry.export_prometheus()

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(metrics_text.encode("utf-8"))
        except Exception as e:
            self.send_error(500, f"Error exporting metrics: {e}")

    def serve_json_metrics(self):
        """Serve metrics in JSON format."""
        try:
            registry = self.registry or get_default_registry()
            metrics_json = registry.export_json()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(metrics_json.encode("utf-8"))
        except Exception as e:
            self.send_error(500, f"Error exporting metrics: {e}")

    def serve_health(self):
        """Serve health check endpoint."""
        health = {"status": "healthy", "timestamp": threading.current_thread().name}

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(health).encode("utf-8"))

    def log_message(self, format, *args):
        """Override to suppress request logging."""
        pass  # Comment this out to enable request logging


class MetricsExporter:
    """
    HTTP server for exporting metrics.

    Runs in a background thread and serves metrics on an HTTP port.
    """

    def __init__(self, port: int = 9090, registry: Optional[MetricsRegistry] = None, address: str = "localhost"):
        """
        Initialize metrics exporter.

        Args:
            port: Port to serve metrics on
            registry: Metrics registry to export
            address: Address to bind to
        """
        self.port = port
        self.address = address
        self.registry = registry or get_default_registry()

        # Set registry in handler class
        MetricsHandler.registry = self.registry

        self.server = HTTPServer((address, port), MetricsHandler)
        self.thread: Optional[threading.Thread] = None
        self.running = False

    def start(self):
        """Start the metrics server in a background thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_server, name="MetricsExporter", daemon=True)
        self.thread.start()

        print(f"✓ Metrics server started on http://{self.address}:{self.port}")
        print(f"  Prometheus: http://{self.address}:{self.port}/metrics")
        print(f"  JSON:       http://{self.address}:{self.port}/metrics.json")
        print(f"  Health:     http://{self.address}:{self.port}/health")

    def _run_server(self):
        """Run the HTTP server."""
        try:
            self.server.serve_forever()
        except Exception as e:
            print(f"Metrics server error: {e}")
        finally:
            self.running = False

    def stop(self, timeout: float = 5.0):
        """Stop the metrics server."""
        if not self.running:
            return

        print("Stopping metrics server...")
        self.server.shutdown()

        if self.thread:
            self.thread.join(timeout=timeout)

        self.running = False
        print("✓ Metrics server stopped")


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

"""
Example 1: Basic metrics usage in VisualCortex
"""


def example_visualcortex_integration():
    from vision_detect_segment import VisualCortex
    from vision_detect_segment.core.visualcortex_metrics import VisionMetrics

    # Initialize VisualCortex with metrics
    cortex = VisualCortex("owlv2", device="auto")

    # Initialize metrics collector
    metrics = VisionMetrics(model_id="owlv2")
    cortex._metrics = metrics  # Attach to cortex

    # Start metrics exporter
    exporter = MetricsExporter(port=9090)
    exporter.start()

    # Process frames (metrics are automatically recorded)
    while True:
        success = cortex.detect_objects_from_redis()

        if success:
            # Record frame processing
            metrics.record_frame_processed(success=True)

            # Record detections
            detections = cortex.get_detected_objects()
            for det in detections:
                metrics.record_detection(class_name=det["label"], confidence=det["confidence"])

        # Update memory metrics periodically
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            mem = process.memory_info()
            metrics.update_memory_usage(mem.rss, mem.vms)
        except Exception as e:
            print(e)
            # pass


"""
Example 2: Manual metrics recording
"""


def example_manual_metrics():
    from vision_detect_segment.utils.metrics import get_default_registry

    # Get default registry
    registry = get_default_registry()

    # Create custom metrics
    requests = registry.register_counter("api_requests_total", "Total API requests", labels=["endpoint", "status"])

    latency = registry.register_histogram("api_latency_seconds", "API request latency", labels=["endpoint"])

    # Record metrics
    requests.inc(labels={"endpoint": "/detect", "status": "success"})
    latency.observe(0.15, labels={"endpoint": "/detect"})

    # Export metrics
    prometheus_format = registry.export_prometheus()
    print(prometheus_format)

    json_format = registry.export_json()
    print(json_format)


"""
Example 3: Using MetricsTimer context manager
"""


def example_metrics_timer():
    from vision_detect_segment.core.visualcortex_metrics import (
        MetricsTimer,
        VisionMetrics,
    )

    metrics = VisionMetrics(model_id="owlv2")

    # Time a detection operation
    with MetricsTimer(metrics, metrics.record_detection_latency, stage="preprocessing"):
        # Your detection code here
        pass

    # Time Redis operation
    with MetricsTimer(metrics, metrics.record_redis_operation, operation="publish", success=True):
        # Redis publish code
        pass


"""
Example 4: Complete integration with async processing
"""


def example_async_with_metrics():
    from vision_detect_segment.core.visualcortex_async import VisualCortexAsync
    from vision_detect_segment.core.visualcortex_metrics import VisionMetrics
    from vision_detect_segment.utils.metrics_exporter import MetricsExporter

    # Initialize async cortex
    cortex = VisualCortexAsync("owlv2", num_workers=2, max_queue_size=100)

    # Initialize metrics
    metrics = VisionMetrics(model_id="owlv2")
    cortex._metrics = metrics

    # Start metrics exporter
    exporter = MetricsExporter(port=9090)
    exporter.start()

    # Start processing
    cortex.start()

    # Monitor metrics in background thread
    import threading
    import time

    def update_metrics():
        while True:
            stats = cortex.get_stats()

            # Update queue metrics
            metrics.update_queue_size("input", stats.get("queue_size", 0))
            metrics.update_active_tracks(stats.get("current_detections", 0))

            # Record throughput
            fps = stats.get("throughput_fps", 0)
            if fps > 0:
                metrics.record_detection_latency(1.0 / fps, stage="total")

            time.sleep(5)

    monitor_thread = threading.Thread(target=update_metrics, daemon=True)
    monitor_thread.start()

    # Let it run
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cortex.stop()
        exporter.stop()


"""
Example 5: Custom metrics for specific use case
"""


def example_custom_metrics():
    from vision_detect_segment.utils.metrics import get_default_registry

    registry = get_default_registry()

    # Robot-specific metrics
    robot_positions = registry.register_gauge("robot_position", "Robot position coordinates", labels=["axis", "robot_id"])

    gripper_operations = registry.register_counter(
        "gripper_operations_total", "Total gripper open/close operations", labels=["robot_id", "operation"]
    )

    pick_success_rate = registry.register_histogram(
        "pick_success_rate",
        "Success rate of pick operations",
        buckets=[0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0],
        labels=["object_type"],
    )

    # Record custom metrics
    robot_positions.set(125.5, labels={"axis": "x", "robot_id": "robot1"})
    gripper_operations.inc(labels={"robot_id": "robot1", "operation": "close"})
    pick_success_rate.observe(0.95, labels={"object_type": "cube"})


"""
Example 6: Integration with visualization dashboard
"""


def example_metrics_dashboard():
    """
    Create a simple dashboard to visualize metrics.
    This could be extended to a full web dashboard.
    """
    import time

    from vision_detect_segment.utils.metrics import get_default_registry

    registry = get_default_registry()

    # Collect metrics periodically
    while True:
        collected = registry.collect_all()

        # Print summary to console (could be sent to dashboard)
        print("\n" + "=" * 70)
        print("METRICS DASHBOARD")
        print("=" * 70)

        for metric_name, metric_data in collected.items():
            print(f"\n{metric_name}:")
            print(f"  Type: {metric_data['type']}")

            # Show some samples
            samples = metric_data.get("samples", [])[:5]
            for sample in samples:
                labels = sample.get("labels", {})
                value = sample.get("value", 0)
                print(f"    {labels} = {value}")

        time.sleep(5)


"""
Example 7: Alerting based on metrics
"""


def example_metrics_alerting():
    from vision_detect_segment.core.visualcortex_metrics import VisionMetrics

    metrics = VisionMetrics(model_id="owlv2")

    # Define alert thresholds
    ALERT_THRESHOLDS = {"max_detection_latency": 1.0, "max_redis_errors": 10, "min_fps": 5.0}  # seconds

    def check_alerts():
        summary = metrics.get_summary()
        alerts = []

        # Check detection latency
        det_stats = summary.get("detection_stats", {})
        avg_latency = det_stats.get("avg", 0)
        if avg_latency > ALERT_THRESHOLDS["max_detection_latency"]:
            alerts.append(
                f"High detection latency: {avg_latency:.3f}s " f"(threshold: {ALERT_THRESHOLDS['max_detection_latency']}s)"
            )

        # Check Redis errors
        redis_stats = summary.get("redis_operations", {})
        failures = redis_stats.get("publish_failure", 0)
        if failures > ALERT_THRESHOLDS["max_redis_errors"]:
            alerts.append(
                f"High Redis error rate: {failures} failures " f"(threshold: {ALERT_THRESHOLDS['max_redis_errors']})"
            )

        return alerts

    # Check alerts periodically
    import time

    while True:
        alerts = check_alerts()
        if alerts:
            print("\n⚠️  ALERTS:")
            for alert in alerts:
                print(f"  - {alert}")

        time.sleep(10)


if __name__ == "__main__":
    # Run example
    print("Starting metrics exporter example...")

    # Start exporter
    exporter = MetricsExporter(port=9090)
    exporter.start()

    # Simulate some metrics
    import random
    import time

    from vision_detect_segment.utils.metrics import get_default_registry

    registry = get_default_registry()

    counter = registry.register_counter("test_counter", "Test counter")
    histogram = registry.register_histogram("test_latency", "Test latency")
    gauge = registry.register_gauge("test_gauge", "Test gauge")

    print("\nGenerating sample metrics...")
    print("Visit http://localhost:9090/metrics to see Prometheus format")
    print("Visit http://localhost:9090/metrics.json to see JSON format")
    print("\nPress Ctrl+C to stop\n")

    try:
        while True:
            # Generate random metrics
            counter.inc()
            histogram.observe(random.uniform(0.01, 1.0))
            gauge.set(random.uniform(0, 100))

            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
        exporter.stop()
