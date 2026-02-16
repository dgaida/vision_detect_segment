"""
vision_detect_segment/utils/metrics.py

Comprehensive metrics collection and monitoring system.
Provides Prometheus-style metrics for production observability.
"""

import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MetricType(Enum):
    """Types of metrics supported."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """Container for a metric value with metadata."""

    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    """
    Prometheus-style Counter metric.
    Monotonically increasing counter for events.
    """

    def __init__(self, name: str, description: str = "", labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment counter by amount."""
        label_key = self._make_label_key(labels)
        with self._lock:
            self._values[label_key] += amount

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        label_key = self._make_label_key(labels)
        with self._lock:
            return self._values[label_key]

    def reset(self, labels: Optional[Dict[str, str]] = None):
        """Reset counter to zero."""
        label_key = self._make_label_key(labels)
        with self._lock:
            self._values[label_key] = 0.0

    def _make_label_key(self, labels: Optional[Dict[str, str]]) -> tuple:
        """Create hashable key from labels."""
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    def collect(self) -> Dict[str, Any]:
        """Collect metric data for export."""
        with self._lock:
            samples = []
            for label_key, value in self._values.items():
                labels = dict(label_key) if label_key else {}
                samples.append({"labels": labels, "value": value, "timestamp": time.time()})

            return {"name": self.name, "type": MetricType.COUNTER.value, "description": self.description, "samples": samples}


class Gauge:
    """
    Prometheus-style Gauge metric.
    Value that can go up or down.
    """

    def __init__(self, name: str, description: str = "", labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def set(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Set gauge to specific value."""
        label_key = self._make_label_key(labels)
        with self._lock:
            self._values[label_key] = value

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment gauge."""
        label_key = self._make_label_key(labels)
        with self._lock:
            self._values[label_key] += amount

    def dec(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Decrement gauge."""
        label_key = self._make_label_key(labels)
        with self._lock:
            self._values[label_key] -= amount

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value."""
        label_key = self._make_label_key(labels)
        with self._lock:
            return self._values[label_key]

    def _make_label_key(self, labels: Optional[Dict[str, str]]) -> tuple:
        """Create hashable key from labels."""
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    def collect(self) -> Dict[str, Any]:
        """Collect metric data for export."""
        with self._lock:
            samples = []
            for label_key, value in self._values.items():
                labels = dict(label_key) if label_key else {}
                samples.append({"labels": labels, "value": value, "timestamp": time.time()})

            return {"name": self.name, "type": MetricType.GAUGE.value, "description": self.description, "samples": samples}


class Histogram:
    """
    Prometheus-style Histogram metric.
    Samples observations and counts them in configurable buckets.
    """

    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(
        self, name: str, description: str = "", buckets: Optional[List[float]] = None, labels: Optional[List[str]] = None
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)

        self._buckets: Dict[tuple, List[int]] = defaultdict(lambda: [0] * len(self.buckets))
        self._sum: Dict[tuple, float] = defaultdict(float)
        self._count: Dict[tuple, int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value."""
        label_key = self._make_label_key(labels)

        with self._lock:
            # Update sum and count
            self._sum[label_key] += value
            self._count[label_key] += 1

            # Update buckets
            for i, bucket in enumerate(self.buckets):
                if value <= bucket:
                    self._buckets[label_key][i] += 1

    def get_statistics(self, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get histogram statistics."""
        label_key = self._make_label_key(labels)

        with self._lock:
            count = self._count[label_key]
            if count == 0:
                return {"count": 0, "sum": 0.0, "avg": 0.0}

            return {"count": count, "sum": self._sum[label_key], "avg": self._sum[label_key] / count}

    def _make_label_key(self, labels: Optional[Dict[str, str]]) -> tuple:
        """Create hashable key from labels."""
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    def collect(self) -> Dict[str, Any]:
        """Collect metric data for export."""
        with self._lock:
            samples = []

            for label_key in self._sum.keys():
                labels = dict(label_key) if label_key else {}

                # Bucket samples
                cumulative = 0
                for i, bucket in enumerate(self.buckets):
                    cumulative += self._buckets[label_key][i]
                    bucket_labels = {**labels, "le": str(bucket)}
                    samples.append({"labels": bucket_labels, "value": cumulative, "timestamp": time.time()})

                # +Inf bucket
                inf_labels = {**labels, "le": "+Inf"}
                samples.append({"labels": inf_labels, "value": self._count[label_key], "timestamp": time.time()})

                # Sum and count
                samples.append({"labels": {**labels, "_type": "sum"}, "value": self._sum[label_key], "timestamp": time.time()})
                samples.append(
                    {"labels": {**labels, "_type": "count"}, "value": self._count[label_key], "timestamp": time.time()}
                )

            return {
                "name": self.name,
                "type": MetricType.HISTOGRAM.value,
                "description": self.description,
                "samples": samples,
                "buckets": self.buckets,
            }


class Summary:
    """
    Prometheus-style Summary metric.
    Tracks size and sum of events, can calculate quantiles.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        max_age: float = 600.0,  # 10 minutes
        quantiles: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.max_age = max_age
        self.quantiles = quantiles or [0.5, 0.9, 0.99]

        self._observations: Dict[tuple, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._sum: Dict[tuple, float] = defaultdict(float)
        self._count: Dict[tuple, int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value."""
        label_key = self._make_label_key(labels)
        timestamp = time.time()

        with self._lock:
            self._observations[label_key].append((value, timestamp))
            self._sum[label_key] += value
            self._count[label_key] += 1

            # Clean old observations
            self._clean_old_observations(label_key, timestamp)

    def _clean_old_observations(self, label_key: tuple, current_time: float):
        """Remove observations older than max_age."""
        cutoff_time = current_time - self.max_age
        obs = self._observations[label_key]

        while obs and obs[0][1] < cutoff_time:
            obs.popleft()

    def get_quantile(self, quantile: float, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Calculate quantile value."""
        label_key = self._make_label_key(labels)

        with self._lock:
            obs = self._observations[label_key]
            if not obs:
                return None

            values = sorted([v for v, _ in obs])
            index = int(len(values) * quantile)
            return values[min(index, len(values) - 1)]

    def get_statistics(self, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get summary statistics."""
        label_key = self._make_label_key(labels)

        with self._lock:
            count = len(self._observations[label_key])
            if count == 0:
                return {"count": 0, "sum": 0.0, "avg": 0.0}

            values = [v for v, _ in self._observations[label_key]]
            stats = {"count": count, "sum": sum(values), "avg": sum(values) / count, "min": min(values), "max": max(values)}

            # Add quantiles
            for q in self.quantiles:
                stats[f"p{int(q*100)}"] = self.get_quantile(q, labels)

            return stats

    def _make_label_key(self, labels: Optional[Dict[str, str]]) -> tuple:
        """Create hashable key from labels."""
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    def collect(self) -> Dict[str, Any]:
        """Collect metric data for export."""
        with self._lock:
            samples = []

            for label_key in self._sum.keys():
                labels = dict(label_key) if label_key else {}
                obs = self._observations[label_key]

                if obs:
                    # Quantile samples
                    for quantile in self.quantiles:
                        q_value = self.get_quantile(quantile, labels)
                        if q_value is not None:
                            q_labels = {**labels, "quantile": str(quantile)}
                            samples.append({"labels": q_labels, "value": q_value, "timestamp": time.time()})

                # Sum and count
                samples.append(
                    {"labels": {**labels, "_type": "sum"}, "value": sum(v for v, _ in obs), "timestamp": time.time()}
                )
                samples.append({"labels": {**labels, "_type": "count"}, "value": len(obs), "timestamp": time.time()})

            return {
                "name": self.name,
                "type": MetricType.SUMMARY.value,
                "description": self.description,
                "samples": samples,
                "quantiles": self.quantiles,
            }


class MetricsRegistry:
    """
    Central registry for all metrics.
    Provides collection and export functionality.
    """

    def __init__(self, namespace: str = "vision_detect_segment"):
        self.namespace = namespace
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def register_counter(self, name: str, description: str = "", labels: Optional[List[str]] = None) -> Counter:
        """Register and return a Counter metric."""
        full_name = f"{self.namespace}_{name}"
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Counter(full_name, description, labels)
            return self._metrics[full_name]

    def register_gauge(self, name: str, description: str = "", labels: Optional[List[str]] = None) -> Gauge:
        """Register and return a Gauge metric."""
        full_name = f"{self.namespace}_{name}"
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Gauge(full_name, description, labels)
            return self._metrics[full_name]

    def register_histogram(
        self, name: str, description: str = "", buckets: Optional[List[float]] = None, labels: Optional[List[str]] = None
    ) -> Histogram:
        """Register and return a Histogram metric."""
        full_name = f"{self.namespace}_{name}"
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Histogram(full_name, description, buckets, labels)
            return self._metrics[full_name]

    def register_summary(
        self,
        name: str,
        description: str = "",
        max_age: float = 600.0,
        quantiles: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
    ) -> Summary:
        """Register and return a Summary metric."""
        full_name = f"{self.namespace}_{name}"
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Summary(full_name, description, max_age, quantiles, labels)
            return self._metrics[full_name]

    def get_metric(self, name: str) -> Optional[Any]:
        """Get a registered metric by name."""
        full_name = f"{self.namespace}_{name}"
        with self._lock:
            return self._metrics.get(full_name)

    def collect_all(self) -> Dict[str, Any]:
        """Collect all metrics for export."""
        with self._lock:
            collected = {}
            for name, metric in self._metrics.items():
                collected[name] = metric.collect()
            return collected

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        collected = self.collect_all()

        for metric_data in collected.values():
            name = metric_data["name"]
            metric_type = metric_data["type"]
            description = metric_data["description"]

            # HELP and TYPE lines
            if description:
                lines.append(f"# HELP {name} {description}")
            lines.append(f"# TYPE {name} {metric_type}")

            # Samples
            for sample in metric_data["samples"]:
                labels = sample["labels"]
                value = sample["value"]

                if labels:
                    label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                    lines.append(f"{name}{{{label_str}}} {value}")
                else:
                    lines.append(f"{name} {value}")

            lines.append("")  # Empty line between metrics

        return "\n".join(lines)

    def export_json(self) -> str:
        """Export metrics in JSON format."""
        return json.dumps(self.collect_all(), indent=2)

    def reset_all(self):
        """Reset all counters (useful for testing)."""
        with self._lock:
            for metric in self._metrics.values():
                if isinstance(metric, Counter):
                    metric.reset()


# Global registry instance
_default_registry = MetricsRegistry()


def get_default_registry() -> MetricsRegistry:
    """Get the default global metrics registry."""
    return _default_registry
