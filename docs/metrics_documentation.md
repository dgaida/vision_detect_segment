# Metrics and Monitoring System

## Overview

The `vision_detect_segment` package includes a comprehensive metrics and monitoring system that provides Prometheus-style metrics for production observability. This system enables you to:

- Track system performance and health
- Monitor detection accuracy and latency
- Identify bottlenecks and errors
- Set up alerting based on thresholds
- Export metrics to monitoring systems like Prometheus, Grafana, or Datadog

## Architecture

The metrics system consists of four main components:

1. **Metric Types**: Counter, Gauge, Histogram, Summary
2. **MetricsRegistry**: Central registry for all metrics
3. **VisionMetrics**: Pre-configured metrics for vision detection
4. **MetricsExporter**: HTTP server for exposing metrics

## Metric Types

### Counter
Monotonically increasing counter for counting events.

```python
from vision_detect_segment.utils.metrics import Counter

# Create counter
frames_processed = Counter(
    "frames_processed_total",
    "Total frames processed",
    labels=["model", "status"]
)

# Increment counter
frames_processed.inc(labels={"model": "owlv2", "status": "success"})

# Get current value
count = frames_processed.get(labels={"model": "owlv2", "status": "success"})
```

### Gauge
Value that can go up or down (e.g., active connections, memory usage).

```python
from vision_detect_segment.utils.metrics import Gauge

# Create gauge
active_tracks = Gauge(
    "active_tracks",
    "Number of active object tracks",
    labels=["model"]
)

# Set gauge value
active_tracks.set(15, labels={"model": "owlv2"})

# Increment/decrement
active_tracks.inc(1, labels={"model": "owlv2"})
active_tracks.dec(1, labels={"model": "owlv2"})
```

### Histogram
Samples observations and counts them in configurable buckets (e.g., latency, request sizes).

```python
from vision_detect_segment.utils.metrics import Histogram

# Create histogram with custom buckets
detection_latency = Histogram(
    "detection_latency_seconds",
    "Detection latency",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    labels=["model"]
)

# Observe values
detection_latency.observe(0.15, labels={"model": "owlv2"})

# Get statistics
stats = detection_latency.get_statistics(labels={"model": "owlv2"})
# Returns: {"count": N, "sum": X, "avg": Y}
```

### Summary
Similar to Histogram but calculates quantiles over a sliding time window.

```python
from vision_detect_segment.utils.metrics import Summary

# Create summary
response_time = Summary(
    "response_time_seconds",
    "Response time",
    max_age=600.0,  # 10 minutes
    quantiles=[0.5, 0.9, 0.99],
    labels=["endpoint"]
)

# Observe values
response_time.observe(0.12, labels={"endpoint": "/detect"})

# Get quantiles
p50 = response_time.get_quantile(0.5, labels={"endpoint": "/detect"})
p99 = response_time.get_quantile(0.99, labels={"endpoint": "/detect"})
```

## Quick Start

### 1. Basic Usage

```python
from vision_detect_segment import VisualCortex
from vision_detect_segment.core.visualcortex_metrics import VisionMetrics
from vision_detect_segment.utils.metrics_exporter import MetricsExporter

# Initialize VisualCortex
cortex = VisualCortex("owlv2", device="auto")

# Initialize metrics
metrics = VisionMetrics(model_id="owlv2")
cortex._metrics = metrics

# Start metrics HTTP server
exporter = MetricsExporter(port=9090)
exporter.start()

# Process frames (metrics are automatically recorded)
while True:
    success = cortex.detect_objects_from_redis()

    if success:
        metrics.record_frame_processed(success=True)

        detections = cortex.get_detected_objects()
        for det in detections:
            metrics.record_detection(
                class_name=det["label"],
                confidence=det["confidence"]
            )

# Metrics available at:
# - http://localhost:9090/metrics (Prometheus format)
# - http://localhost:9090/metrics.json (JSON format)
# - http://localhost:9090/health (Health check)
```

### 2. View Metrics

**Prometheus format:**
```bash
curl http://localhost:9090/metrics
```

**JSON format:**
```bash
curl http://localhost:9090/metrics.json | jq
```

## Available Metrics

### Frame Processing Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `frames_processed_total` | Counter | Total frames processed | model, status |
| `frames_dropped_total` | Counter | Frames dropped due to backpressure | model, reason |

### Detection Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `detections_total` | Counter | Total objects detected | model, class |
| `detection_latency_seconds` | Histogram | Detection latency | model, stage |
| `detection_confidence` | Histogram | Detection confidence scores | model, class |

### Redis Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `redis_operations_total` | Counter | Total Redis operations | operation, status |
| `redis_errors_total` | Counter | Redis errors | operation, error_type |
| `redis_operation_duration_seconds` | Histogram | Redis operation duration | operation |

### Tracking Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `active_tracks` | Gauge | Currently active tracks | model |
| `track_lifetime_seconds` | Summary | Track lifetime | model, class |
| `tracks_created_total` | Counter | Tracks created | model |
| `tracks_lost_total` | Counter | Tracks lost | model, reason |

### Model Inference Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `model_inference_duration_seconds` | Histogram | Model inference time | model, device |
| `model_batch_size` | Histogram | Inference batch size | model |

### System Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `memory_usage_bytes` | Gauge | Memory usage | type |
| `gpu_memory_usage_bytes` | Gauge | GPU memory usage | device |
| `errors_total` | Counter | Total errors | component, error_type |

### Queue Metrics (Async)

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `queue_size` | Gauge | Current queue size | queue_type |
| `queue_wait_duration_seconds` | Histogram | Queue wait time | queue_type |

## Advanced Usage

### Custom Metrics

```python
from vision_detect_segment.utils.metrics import get_default_registry

registry = get_default_registry()

# Create custom counter
api_requests = registry.register_counter(
    "api_requests_total",
    "Total API requests",
    labels=["endpoint", "method", "status"]
)

# Create custom histogram
processing_time = registry.register_histogram(
    "processing_time_seconds",
    "Processing time",
    buckets=[0.001, 0.01, 0.1, 1.0, 10.0],
    labels=["operation"]
)

# Record metrics
api_requests.inc(labels={
    "endpoint": "/detect",
    "method": "POST",
    "status": "200"
})

processing_time.observe(0.15, labels={"operation": "preprocessing"})
```

### Metrics Timer

Use the `MetricsTimer` context manager to automatically time operations:

```python
from vision_detect_segment.core.visualcortex_metrics import (
    VisionMetrics,
    MetricsTimer
)

metrics = VisionMetrics(model_id="owlv2")

# Time a detection operation
with MetricsTimer(
    metrics,
    metrics.record_detection_latency,
    stage="inference"
):
    # Your detection code here
    results = model.detect(image)

# Latency is automatically recorded
```

### Memory Monitoring

```python
import psutil
import os

def update_memory_metrics(metrics):
    process = psutil.Process(os.getpid())
    mem = process.memory_info()

    # Update memory metrics
    metrics.update_memory_usage(mem.rss, mem.vms)

    # GPU memory (if available)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated()
            metrics.update_gpu_memory("cuda:0", gpu_mem)
    except:
        pass

# Call periodically
import threading
def memory_monitor():
    while True:
        update_memory_metrics(metrics)
        time.sleep(5)

threading.Thread(target=memory_monitor, daemon=True).start()
```

### Alerting

```python
def check_alerts(metrics):
    """Check metrics and generate alerts."""
    summary = metrics.get_summary()
    alerts = []

    # Check detection latency
    det_stats = summary.get("detection_stats", {})
    avg_latency = det_stats.get("avg", 0)
    if avg_latency > 1.0:  # 1 second threshold
        alerts.append(f"High detection latency: {avg_latency:.3f}s")

    # Check Redis errors
    redis_ops = summary.get("redis_operations", {})
    failures = redis_ops.get("publish_failure", 0)
    if failures > 10:
        alerts.append(f"High Redis error rate: {failures} failures")

    # Send alerts
    for alert in alerts:
        print(f"⚠️  ALERT: {alert}")
        # Could send to Slack, PagerDuty, etc.

    return alerts

# Run periodically
import time
while True:
    check_alerts(metrics)
    time.sleep(30)
```

## Integration with Monitoring Systems

### Prometheus

1. **Configure Prometheus to scrape metrics:**

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'vision_detect_segment'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 15s
```

2. **Start Prometheus:**

```bash
prometheus --config.file=prometheus.yml
```

3. **Query metrics in Prometheus:**

```promql
# Average detection latency
rate(vision_detect_segment_detection_latency_seconds_sum[5m]) /
rate(vision_detect_segment_detection_latency_seconds_count[5m])

# Frames per second
rate(vision_detect_segment_frames_processed_total[1m])

# Error rate
rate(vision_detect_segment_errors_total[5m])
```

### Grafana

1. **Add Prometheus as data source**
2. **Import dashboard or create custom panels:**

```json
{
  "title": "Vision Detection Dashboard",
  "panels": [
    {
      "title": "Frames Per Second",
      "targets": [
        {
          "expr": "rate(vision_detect_segment_frames_processed_total{status=\"success\"}[1m])"
        }
      ]
    },
    {
      "title": "Detection Latency (p95)",
      "targets": [
        {
          "expr": "histogram_quantile(0.95, rate(vision_detect_segment_detection_latency_seconds_bucket[5m]))"
        }
      ]
    }
  ]
}
```

### Custom Exporter

Create a custom exporter for your monitoring system:

```python
import requests
import json

def export_to_datadog(metrics_url, api_key):
    """Export metrics to Datadog."""
    # Get metrics in JSON format
    response = requests.get(f"{metrics_url}/metrics.json")
    metrics = response.json()

    # Convert to Datadog format
    datadog_metrics = []
    for metric_name, metric_data in metrics.items():
        for sample in metric_data.get("samples", []):
            datadog_metrics.append({
                "metric": metric_name,
                "points": [[sample["timestamp"], sample["value"]]],
                "tags": [f"{k}:{v}" for k, v in sample["labels"].items()]
            })

    # Send to Datadog
    headers = {"Content-Type": "application/json"}
    requests.post(
        f"https://api.datadoghq.com/api/v1/series?api_key={api_key}",
        headers=headers,
        data=json.dumps({"series": datadog_metrics})
    )

# Run periodically
import time
while True:
    export_to_datadog("http://localhost:9090", "your-api-key")
    time.sleep(60)
```

## Best Practices

### 1. Label Cardinality

Avoid high-cardinality labels (labels with many unique values):

❌ **Bad:**
```python
# User ID as label (potentially millions of values)
counter.inc(labels={"user_id": "user_12345"})
```

✅ **Good:**
```python
# Use aggregated labels
counter.inc(labels={"user_type": "premium"})
```

### 2. Metric Naming

Follow Prometheus naming conventions:

- Use `_total` suffix for counters
- Use `_seconds` suffix for time measurements
- Use `_bytes` suffix for sizes
- Use snake_case

### 3. Buckets for Histograms

Choose appropriate buckets based on your use case:

```python
# For latency (milliseconds to seconds)
latency_buckets = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]

# For sizes (bytes to megabytes)
size_buckets = [1024, 10240, 102400, 1048576, 10485760]

# For percentages
percentage_buckets = [0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]
```

### 4. Performance

- Metrics collection should be lightweight
- Use gauges for values that change frequently
- Use summaries instead of histograms for high-volume metrics
- Consider sampling for very high-frequency events

### 5. Documentation

Document your custom metrics:

```python
robot_position = registry.register_gauge(
    "robot_position_mm",
    "Robot end-effector position in millimeters. "
    "Labels: axis (x/y/z), robot_id. "
    "Updated every 100ms during operation.",
    labels=["axis", "robot_id"]
)
```

## Troubleshooting

### Metrics Not Appearing

1. **Check exporter is running:**
```bash
curl http://localhost:9090/health
```

2. **Verify metric registration:**
```python
from vision_detect_segment.utils.metrics import get_default_registry
registry = get_default_registry()
print(registry.collect_all())
```

3. **Check for label mismatches:**
```python
# Ensure labels match when incrementing
counter.inc(labels={"model": "owlv2"})  # Must match registration
```

### High Memory Usage

If metrics consume too much memory:

1. **Reduce histogram bucket count**
2. **Use summaries instead of histograms**
3. **Decrease summary max_age**
4. **Clear old metrics periodically**

### Prometheus Scrape Errors

Check Prometheus logs:
```bash
tail -f /var/log/prometheus/prometheus.log
```

Common issues:
- Firewall blocking port 9090
- Incorrect target configuration
- Metrics endpoint not responding

## Examples

See `vision_detect_segment/utils/metrics_exporter.py` for complete examples including:

- Basic metrics usage
- Async processing with metrics
- Custom metrics for robotics
- Alerting based on thresholds
- Integration with monitoring dashboards

## API Reference

### MetricsRegistry

```python
registry = MetricsRegistry(namespace="my_app")

# Register metrics
counter = registry.register_counter(name, description, labels)
gauge = registry.register_gauge(name, description, labels)
histogram = registry.register_histogram(name, description, buckets, labels)
summary = registry.register_summary(name, description, max_age, quantiles, labels)

# Export
prometheus_text = registry.export_prometheus()
json_text = registry.export_json()
collected = registry.collect_all()
```

### VisionMetrics

```python
metrics = VisionMetrics(registry, model_id)

# Recording methods
metrics.record_frame_processed(success)
metrics.record_detection(class_name, confidence)
metrics.record_detection_latency(duration, stage)
metrics.record_redis_operation(operation, duration, success)
metrics.record_model_inference(duration, device, batch_size)
metrics.update_active_tracks(count)
metrics.update_memory_usage(rss_bytes, vms_bytes)

# Summary
summary = metrics.get_summary()
```

### MetricsExporter

```python
exporter = MetricsExporter(port, registry, address)
exporter.start()  # Start HTTP server
exporter.stop()   # Stop HTTP server
```

## License

MIT License - See LICENSE file for details.
