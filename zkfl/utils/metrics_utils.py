"""
Metrics utilities for ZKFL framework.

This module provides comprehensive metrics collection, analysis,
and reporting for federated learning experiments.
"""

import time
import math
import statistics
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..core.exceptions import MetricsError


@dataclass
class MetricPoint:
    """Individual metric measurement."""

    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metric measurements."""

    name: str
    points: List[MetricPoint] = field(default_factory=list)
    aggregation_window: Optional[timedelta] = None

    def add_point(self, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Add a new metric point."""
        point = MetricPoint(
            timestamp=datetime.now(), value=value, metadata=metadata or {}
        )
        self.points.append(point)

        # Apply window if specified
        if self.aggregation_window:
            cutoff_time = datetime.now() - self.aggregation_window
            self.points = [p for p in self.points if p.timestamp >= cutoff_time]

    def get_recent_values(self, duration: timedelta) -> List[float]:
        """Get values from the last duration."""
        cutoff_time = datetime.now() - duration
        return [p.value for p in self.points if p.timestamp >= cutoff_time]

    def get_average(self, duration: Optional[timedelta] = None) -> Optional[float]:
        """Get average value over duration."""
        if duration:
            values = self.get_recent_values(duration)
        else:
            values = [p.value for p in self.points]

        return statistics.mean(values) if values else None

    def get_latest(self) -> Optional[float]:
        """Get latest metric value."""
        return self.points[-1].value if self.points else None


class MetricsUtils:
    """Comprehensive metrics collection and analysis."""

    def __init__(self):
        self.metric_series: Dict[str, MetricSeries] = {}
        self.custom_aggregators: Dict[str, Callable] = {}
        self.performance_counters: Dict[str, int] = defaultdict(int)
        self.timing_data: Dict[str, List[float]] = defaultdict(list)

    def create_metric_series(
        self, name: str, aggregation_window: Optional[timedelta] = None
    ) -> MetricSeries:
        """
        Create a new metric series.

        Args:
            name: Metric name
            aggregation_window: Optional time window for aggregation

        Returns:
            Created metric series
        """
        if name in self.metric_series:
            raise MetricsError(f"Metric series '{name}' already exists")

        series = MetricSeries(name=name, aggregation_window=aggregation_window)
        self.metric_series[name] = series
        return series

    def record_metric(
        self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            metadata: Optional metadata
        """
        if name not in self.metric_series:
            self.create_metric_series(name)

        self.metric_series[name].add_point(value, metadata)

    def get_metric_summary(
        self, name: str, duration: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive summary of a metric.

        Args:
            name: Metric name
            duration: Optional time window

        Returns:
            Metric summary statistics
        """
        if name not in self.metric_series:
            raise MetricsError(f"Metric series '{name}' not found")

        series = self.metric_series[name]

        if duration:
            values = series.get_recent_values(duration)
        else:
            values = [p.value for p in series.points]

        if not values:
            return {"name": name, "count": 0}

        return {
            "name": name,
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "latest": values[-1],
            "first": values[0],
        }

    def compute_federated_metrics(
        self,
        client_metrics: List[Dict[str, float]],
        aggregation_weights: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Compute federated aggregation of client metrics.

        Args:
            client_metrics: List of client metric dictionaries
            aggregation_weights: Optional weights for aggregation

        Returns:
            Aggregated metrics
        """
        if not client_metrics:
            return {}

        if aggregation_weights is None:
            aggregation_weights = [1.0] * len(client_metrics)

        if len(aggregation_weights) != len(client_metrics):
            raise MetricsError("Number of weights must match number of clients")

        # Normalize weights
        total_weight = sum(aggregation_weights)
        if total_weight == 0:
            raise MetricsError("Total weight cannot be zero")

        normalized_weights = [w / total_weight for w in aggregation_weights]

        # Get all metric names
        all_metrics = set()
        for metrics in client_metrics:
            all_metrics.update(metrics.keys())

        aggregated = {}

        for metric_name in all_metrics:
            weighted_sum = 0.0
            total_weight_for_metric = 0.0

            for client_metrics_dict, weight in zip(client_metrics, normalized_weights):
                if metric_name in client_metrics_dict:
                    weighted_sum += client_metrics_dict[metric_name] * weight
                    total_weight_for_metric += weight

            if total_weight_for_metric > 0:
                aggregated[metric_name] = weighted_sum / total_weight_for_metric

        return aggregated

    def track_convergence(
        self,
        metric_name: str,
        values: List[float],
        window_size: int = 10,
        tolerance: float = 1e-4,
    ) -> Dict[str, Any]:
        """
        Analyze convergence of a metric.

        Args:
            metric_name: Name of the metric
            values: Sequence of metric values
            window_size: Window size for convergence analysis
            tolerance: Tolerance for convergence detection

        Returns:
            Convergence analysis results
        """
        if len(values) < window_size:
            return {
                "metric_name": metric_name,
                "converged": False,
                "reason": "Insufficient data points",
                "stability_score": 0.0,
            }

        # Calculate moving average and variance
        recent_values = values[-window_size:]
        mean_value = statistics.mean(recent_values)
        variance = statistics.variance(recent_values) if len(recent_values) > 1 else 0.0

        # Check for convergence
        converged = variance < tolerance

        # Calculate stability score (inverse of coefficient of variation)
        cv = math.sqrt(variance) / abs(mean_value) if mean_value != 0 else float("inf")
        stability_score = 1.0 / (1.0 + cv)

        # Trend analysis
        if len(values) >= 2:
            recent_trend = (values[-1] - values[-min(5, len(values))]) / min(
                5, len(values)
            )
        else:
            recent_trend = 0.0

        return {
            "metric_name": metric_name,
            "converged": converged,
            "stability_score": stability_score,
            "variance": variance,
            "mean": mean_value,
            "recent_trend": recent_trend,
            "coefficient_of_variation": cv,
        }

    def calculate_privacy_metrics(
        self, epsilon_values: List[float], delta_values: List[float]
    ) -> Dict[str, float]:
        """
        Calculate privacy budget metrics.

        Args:
            epsilon_values: List of epsilon values
            delta_values: List of delta values

        Returns:
            Privacy metrics
        """
        if not epsilon_values or not delta_values:
            return {}

        # Total privacy budget consumed
        total_epsilon = sum(epsilon_values)
        total_delta = sum(delta_values)

        # Privacy budget efficiency metrics
        avg_epsilon = statistics.mean(epsilon_values)
        avg_delta = statistics.mean(delta_values)

        epsilon_variance = (
            statistics.variance(epsilon_values) if len(epsilon_values) > 1 else 0.0
        )
        delta_variance = (
            statistics.variance(delta_values) if len(delta_values) > 1 else 0.0
        )

        return {
            "total_epsilon": total_epsilon,
            "total_delta": total_delta,
            "avg_epsilon": avg_epsilon,
            "avg_delta": avg_delta,
            "epsilon_variance": epsilon_variance,
            "delta_variance": delta_variance,
            "privacy_rounds": len(epsilon_values),
        }

    def measure_communication_overhead(
        self, message_sizes: List[int], round_times: List[float]
    ) -> Dict[str, Any]:
        """
        Measure communication overhead in federated learning.

        Args:
            message_sizes: List of message sizes in bytes
            round_times: List of round completion times

        Returns:
            Communication metrics
        """
        if not message_sizes or not round_times:
            return {}

        # Basic statistics
        total_bytes = sum(message_sizes)
        avg_message_size = statistics.mean(message_sizes)
        max_message_size = max(message_sizes)
        min_message_size = min(message_sizes)

        avg_round_time = statistics.mean(round_times)

        # Throughput metrics
        total_time = sum(round_times)
        throughput_bytes_per_sec = total_bytes / total_time if total_time > 0 else 0
        throughput_messages_per_sec = (
            len(message_sizes) / total_time if total_time > 0 else 0
        )

        # Efficiency metrics
        size_variance = (
            statistics.variance(message_sizes) if len(message_sizes) > 1 else 0
        )
        time_variance = statistics.variance(round_times) if len(round_times) > 1 else 0

        return {
            "total_bytes": total_bytes,
            "total_messages": len(message_sizes),
            "avg_message_size": avg_message_size,
            "max_message_size": max_message_size,
            "min_message_size": min_message_size,
            "message_size_variance": size_variance,
            "avg_round_time": avg_round_time,
            "round_time_variance": time_variance,
            "throughput_bytes_per_sec": throughput_bytes_per_sec,
            "throughput_messages_per_sec": throughput_messages_per_sec,
            "total_time": total_time,
        }

    def analyze_model_performance(
        self,
        training_metrics: Dict[str, List[float]],
        validation_metrics: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """
        Analyze model performance across training.

        Args:
            training_metrics: Training metrics over time
            validation_metrics: Validation metrics over time

        Returns:
            Performance analysis
        """
        analysis = {}

        # Analyze each metric type
        for metric_name in training_metrics.keys():
            train_values = training_metrics[metric_name]
            val_values = validation_metrics.get(metric_name, [])

            metric_analysis = {
                "train_final": train_values[-1] if train_values else None,
                "train_best": (
                    min(train_values)
                    if "loss" in metric_name.lower()
                    else max(train_values)
                ),
                "train_improvement": self._calculate_improvement(train_values),
                "train_stability": self._calculate_stability(train_values),
            }

            if val_values:
                metric_analysis.update(
                    {
                        "val_final": val_values[-1],
                        "val_best": (
                            min(val_values)
                            if "loss" in metric_name.lower()
                            else max(val_values)
                        ),
                        "val_improvement": self._calculate_improvement(val_values),
                        "val_stability": self._calculate_stability(val_values),
                        "overfitting_score": self._calculate_overfitting_score(
                            train_values, val_values
                        ),
                    }
                )

            analysis[metric_name] = metric_analysis

        return analysis

    def _calculate_improvement(self, values: List[float]) -> float:
        """Calculate improvement from first to last value."""
        if len(values) < 2:
            return 0.0

        first_val = values[0]
        last_val = values[-1]

        if first_val == 0:
            return 0.0

        return (last_val - first_val) / abs(first_val)

    def _calculate_stability(self, values: List[float]) -> float:
        """Calculate stability score for a metric series."""
        if len(values) < 2:
            return 1.0

        # Use coefficient of variation as instability measure
        mean_val = statistics.mean(values)
        if mean_val == 0:
            return 0.0

        std_val = statistics.stdev(values)
        cv = std_val / abs(mean_val)

        # Convert to stability score (higher is more stable)
        return 1.0 / (1.0 + cv)

    def _calculate_overfitting_score(
        self, train_values: List[float], val_values: List[float]
    ) -> float:
        """Calculate overfitting score based on train/validation gap."""
        if len(train_values) != len(val_values) or len(train_values) == 0:
            return 0.0

        # Calculate recent performance gap
        recent_window = min(5, len(train_values))
        recent_train = statistics.mean(train_values[-recent_window:])
        recent_val = statistics.mean(val_values[-recent_window:])

        # Gap score (higher means more overfitting)
        if recent_val == 0:
            return 0.0

        gap_ratio = abs(recent_train - recent_val) / abs(recent_val)
        return min(gap_ratio, 10.0)  # Cap at 10.0

    def create_performance_report(
        self, client_id: Optional[str] = None, round_num: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive performance report.

        Args:
            client_id: Optional client identifier
            round_num: Optional round number

        Returns:
            Performance report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "client_id": client_id,
            "round_num": round_num,
            "metrics_summary": {},
            "performance_counters": dict(self.performance_counters),
            "timing_summary": {},
        }

        # Summarize all metric series
        for name, series in self.metric_series.items():
            report["metrics_summary"][name] = self.get_metric_summary(name)

        # Summarize timing data
        for operation, times in self.timing_data.items():
            if times:
                report["timing_summary"][operation] = {
                    "count": len(times),
                    "total_time": sum(times),
                    "avg_time": statistics.mean(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "std_time": statistics.stdev(times) if len(times) > 1 else 0.0,
                }

        return report

    def start_timer(self, operation: str) -> float:
        """Start timing an operation."""
        return time.time()

    def end_timer(self, operation: str, start_time: float):
        """End timing an operation and record the duration."""
        duration = time.time() - start_time
        self.timing_data[operation].append(duration)
        return duration

    def increment_counter(self, counter_name: str, increment: int = 1):
        """Increment a performance counter."""
        self.performance_counters[counter_name] += increment

    def reset_metrics(self, metric_names: Optional[List[str]] = None):
        """
        Reset metrics data.

        Args:
            metric_names: Optional list of specific metrics to reset
        """
        if metric_names is None:
            # Reset all metrics
            self.metric_series.clear()
            self.performance_counters.clear()
            self.timing_data.clear()
        else:
            # Reset specific metrics
            for name in metric_names:
                if name in self.metric_series:
                    self.metric_series[name].points.clear()
                if name in self.performance_counters:
                    self.performance_counters[name] = 0
                if name in self.timing_data:
                    self.timing_data[name].clear()

    def export_metrics(self, format_type: str = "json") -> Union[str, Dict[str, Any]]:
        """
        Export metrics in specified format.

        Args:
            format_type: Export format ("json", "csv", "dict")

        Returns:
            Exported metrics data
        """
        if format_type == "dict":
            return self.create_performance_report()

        elif format_type == "json":
            import json

            return json.dumps(self.create_performance_report(), indent=2, default=str)

        elif format_type == "csv":
            # Simple CSV export for metric series
            csv_lines = ["metric_name,timestamp,value"]

            for name, series in self.metric_series.items():
                for point in series.points:
                    csv_lines.append(
                        f"{name},{point.timestamp.isoformat()},{point.value}"
                    )

            return "\n".join(csv_lines)

        else:
            raise MetricsError(f"Unsupported export format: {format_type}")

    def add_custom_aggregator(
        self, name: str, aggregator_func: Callable[[List[float]], float]
    ):
        """
        Add custom aggregation function.

        Args:
            name: Aggregator name
            aggregator_func: Function that takes list of values and returns aggregated result
        """
        self.custom_aggregators[name] = aggregator_func

    def apply_custom_aggregation(
        self, metric_name: str, aggregator_name: str
    ) -> Optional[float]:
        """
        Apply custom aggregation to a metric.

        Args:
            metric_name: Name of metric to aggregate
            aggregator_name: Name of aggregator to use

        Returns:
            Aggregated value or None if not possible
        """
        if aggregator_name not in self.custom_aggregators:
            raise MetricsError(f"Unknown aggregator: {aggregator_name}")

        if metric_name not in self.metric_series:
            return None

        values = [p.value for p in self.metric_series[metric_name].points]
        if not values:
            return None

        return self.custom_aggregators[aggregator_name](values)
