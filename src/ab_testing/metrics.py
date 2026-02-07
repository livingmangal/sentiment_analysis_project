"""
Metrics Collection System for A/B Testing
Collects and aggregates performance metrics for model variants
"""
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class VariantMetrics:
    """Metrics for a single variant"""
    variant_id: str
    total_predictions: int = 0
    avg_confidence: float = 0.0
    avg_inference_time_ms: float = 0.0
    sentiment_distribution: Optional[Dict[str, int]] = None
    confidence_histogram: Optional[List[int]] = None
    error_count: int = 0

    def __post_init__(self) -> None:
        if self.sentiment_distribution is None:
            self.sentiment_distribution = {"Positive": 0, "Negative": 0, "Neutral": 0}
        if self.confidence_histogram is None:
            # 10 bins from 0.0 to 1.0
            self.confidence_histogram = [0] * 10


class MetricsCollector:
    """
    Collects and aggregates metrics from A/B test predictions
    """

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "experiments",
            experiment_name
        )

    def collect_metrics(
        self,
        variant_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> VariantMetrics:
        """
        Collect metrics for a specific variant
        
        Args:
            variant_id: Variant identifier
            start_time: Filter predictions after this time
            end_time: Filter predictions before this time
            
        Returns:
            VariantMetrics object with aggregated metrics
        """
        log_file = os.path.join(
            self.experiment_dir,
            f"{variant_id}_predictions.jsonl"
        )

        if not os.path.exists(log_file):
            return VariantMetrics(variant_id=variant_id)

        metrics = VariantMetrics(variant_id=variant_id)

        confidences: List[float] = []
        inference_times: List[float] = []

        with open(log_file, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())

                # Filter by time if specified
                timestamp = datetime.fromisoformat(entry["timestamp"])
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue

                # Aggregate metrics
                metrics.total_predictions += 1

                sentiment = entry.get("sentiment")
                if metrics.sentiment_distribution is not None and sentiment in metrics.sentiment_distribution:
                    metrics.sentiment_distribution[sentiment] += 1

                confidence = entry.get("confidence")
                if confidence is not None:
                    confidences.append(float(confidence))
                    # Add to histogram (bin index)
                    bin_idx = min(int(float(confidence) * 10), 9)
                    if metrics.confidence_histogram is not None:
                        metrics.confidence_histogram[bin_idx] += 1

                inference_time = entry.get("inference_time_ms")
                if inference_time is not None:
                    inference_times.append(float(inference_time))

        # Calculate averages
        if confidences:
            metrics.avg_confidence = float(np.mean(confidences))
        if inference_times:
            metrics.avg_inference_time_ms = float(np.mean(inference_times))

        return metrics

    def collect_all_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, VariantMetrics]:
        """
        Collect metrics for all variants in the experiment
        
        Returns:
            Dictionary mapping variant_id to VariantMetrics
        """
        # Find all prediction log files
        metrics_dict: Dict[str, VariantMetrics] = {}

        if not os.path.exists(self.experiment_dir):
            return metrics_dict

        for filename in os.listdir(self.experiment_dir):
            if filename.endswith("_predictions.jsonl"):
                variant_id = filename.replace("_predictions.jsonl", "")
                metrics = self.collect_metrics(variant_id, start_time, end_time)
                metrics_dict[variant_id] = metrics

        return metrics_dict

    def get_time_series_metrics(
        self,
        variant_id: str,
        interval_hours: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get time-series metrics for a variant
        
        Args:
            variant_id: Variant identifier
            interval_hours: Aggregation interval in hours
            
        Returns:
            List of metric dictionaries with timestamps
        """
        log_file = os.path.join(
            self.experiment_dir,
            f"{variant_id}_predictions.jsonl"
        )

        if not os.path.exists(log_file):
            return []

        # Group predictions by time interval
        interval_data: Dict[datetime, Dict[str, Any]] = defaultdict(lambda: {
            "predictions": 0,
            "confidences": [],
            "inference_times": [],
            "sentiments": {"Positive": 0, "Negative": 0, "Neutral": 0}
        })

        with open(log_file, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                timestamp = datetime.fromisoformat(entry["timestamp"])

                # Round to interval
                interval_key = timestamp.replace(
                    minute=0,
                    second=0,
                    microsecond=0
                )
                interval_key = interval_key.replace(
                    hour=(interval_key.hour // interval_hours) * interval_hours
                )

                data = interval_data[interval_key]
                data["predictions"] = data["predictions"] + 1

                if entry.get("confidence"):
                    data["confidences"].append(float(entry["confidence"]))
                if entry.get("inference_time_ms"):
                    data["inference_times"].append(float(entry["inference_time_ms"]))

                sentiment = entry.get("sentiment")
                if sentiment in data["sentiments"]:
                    data["sentiments"][sentiment] += 1

        # Convert to list of metrics
        time_series = []
        for timestamp, data in sorted(interval_data.items()):
            metric = {
                "timestamp": timestamp.isoformat(),
                "predictions": data["predictions"],
                "avg_confidence": float(np.mean(data["confidences"])) if data["confidences"] else 0.0,
                "avg_inference_time_ms": float(np.mean(data["inference_times"])) if data["inference_times"] else 0.0,
                "sentiment_distribution": data["sentiments"]
            }
            time_series.append(metric)

        return time_series

    def export_metrics(
        self,
        output_file: Optional[str] = None
    ) -> str:
        """
        Export all metrics to JSON file
        
        Args:
            output_file: Output file path (optional)
            
        Returns:
            Path to exported file
        """
        if output_file is None:
            output_file = os.path.join(
                self.experiment_dir,
                f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        all_metrics = self.collect_all_metrics()

        export_data = {
            "experiment_name": self.experiment_name,
            "export_timestamp": datetime.now().isoformat(),
            "variants": {
                vid: asdict(metrics)
                for vid, metrics in all_metrics.items()
            }
        }

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        return output_file

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics across all variants
        
        Returns:
            Dictionary with summary statistics
        """
        all_metrics = self.collect_all_metrics()

        if not all_metrics:
            return {
                "total_variants": 0,
                "total_predictions": 0
            }

        total_predictions = sum(m.total_predictions for m in all_metrics.values())

        summary: Dict[str, Any] = {
            "experiment_name": self.experiment_name,
            "total_variants": len(all_metrics),
            "total_predictions": total_predictions,
            "variants": {}
        }

        for variant_id, metrics in all_metrics.items():
            summary["variants"][variant_id] = {
                "predictions": metrics.total_predictions,
                "percentage": (metrics.total_predictions / total_predictions * 100) if total_predictions > 0 else 0.0,
                "avg_confidence": round(metrics.avg_confidence, 4),
                "avg_inference_time_ms": round(metrics.avg_inference_time_ms, 2),
                "sentiment_distribution": metrics.sentiment_distribution
            }

        return summary
