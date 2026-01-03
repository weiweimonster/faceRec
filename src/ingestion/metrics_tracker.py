import numpy as np
import json
from collections import defaultdict
from typing import Dict, Any, List
from src.common.types import ImageAnalysisResult
from src.util.logger import logger

class IngestionMetricsTracker:
    def __init__(self):
        # Stores lists of raw values (e.g., "iso": [100, 200, 800...])
        self.numeric_data: Dict[str, List[float]] = defaultdict(list)

        # Stores counts of missing fields
        self.missing_counts: Dict[str, int] = defaultdict(int)

        # Counters
        self.total_photos = 0
        self.total_faces = 0

    def update(self, result: ImageAnalysisResult):
        """
        Ingests stats from a single photo analysis result.
        Automatically discovers metrics from the data objects via their .metrics property.
        """
        if not result:
            logger.error(f"Result is empty. Skip tracking metrics")
            return

        self.total_photos += 1

        # --- 1. Global Metrics ---
        # Iterate over all exposed metrics (Strings, Ints, Floats, None)
        for name, value in result.metrics.items():
            self._track_metric(name, value)

        # --- 2. Face Metrics ---
        if result.faces:
            for face in result.faces:
                self.total_faces += 1

                # Iterate over face metrics
                for name, value in face.metrics.items():
                    # Prefix face metrics to distinguish them from global ones
                    # e.g., "blur_score" -> "face_blur_score"
                    key = f"face_{name}" if not name.startswith("face_") else name
                    self._track_metric(key, value)

    def _track_metric(self, name: str, value: Any):
        """Helper to safely add numeric values or count missing ones."""
        if value is None:
            self.missing_counts[name] += 1
            return
        # Check for Numeric Data (Statistics)
        # We explicitly verify it's a number to avoid crashing on strings like "Medium-Shot"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                self.numeric_data[name].append(float(value))
            except (ValueError, TypeError):
                # Should not happen given the isinstance check, but good safety
                logger.error(f"Error tracking field: {name}")
                pass

    def finalize_report(self, output_path: str = "ingestion_metrics.json") -> Dict[str, Any]:
        """
        Computes statistics (Min, Max, Mean, Percentiles) for all numeric data
        and saves the full report to a JSON file.
        """
        logger.info("Computing final ingestion statistics...")

        report = {
            "summary": {
                "total_photos": self.total_photos,
                "total_faces": self.total_faces,
                "avg_faces_per_photo": round(self.total_faces / max(1, self.total_photos), 2)
            },
            "numeric_stats": {},
            "missing_field_counts": dict(self.missing_counts)
        }

        # Calculate stats for every numeric field
        for field, values in self.numeric_data.items():
            if not values:
                continue

            arr = np.array(values)

            # Standard statistics
            report["numeric_stats"][field] = {
                "count": int(len(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": round(float(np.mean(arr)), 3),
                "median": round(float(np.median(arr)), 3),
                "std_dev": round(float(np.std(arr)), 3),

                # Percentiles (Critical for tuning Ranker bounds)
                "p05": round(float(np.percentile(arr, 5)), 3),
                "p25": round(float(np.percentile(arr, 25)), 3),
                "p75": round(float(np.percentile(arr, 75)), 3),
                "p95": round(float(np.percentile(arr, 95)), 3)
            }

        # Save to disk
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=4)
            logger.info(f"✅ Metrics report saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics report: {e}")

        return report