import numpy as np
import json
from collections import defaultdict
from typing import Dict, Any, List, Optional
from src.common.types import ImageAnalysisResult
from src.features.registry import (
    get_trackable_features,
    get_feature,
    FeatureType,
)
from src.util.logger import logger


class IngestionMetricsTracker:
    """
    Tracks statistics for features during ingestion.
    Uses the feature registry to determine which features to track.
    """

    def __init__(self):
        # Get trackable features from registry
        self.trackable_features = get_trackable_features()

        # Stores lists of raw values (e.g., "aesthetic_score": [4.5, 5.2, ...])
        self.numeric_data: Dict[str, List[float]] = defaultdict(list)

        # Stores counts of missing fields
        self.missing_counts: Dict[str, int] = defaultdict(int)

        # Counters
        self.total_photos = 0
        self.total_faces = 0

    def update(self, result: ImageAnalysisResult):
        """
        Ingests stats from a single photo analysis result.
        Uses feature registry to extract values.
        """
        if not result:
            logger.error("Result is empty. Skip tracking metrics")
            return

        self.total_photos += 1

        # Count faces
        if result.faces:
            self.total_faces += len(result.faces)

        # Extract and track each trackable feature
        for feat_name in self.trackable_features:
            feat_def = get_feature(feat_name)
            if not feat_def:
                continue

            value = self._extract_value(result, feat_def)
            self._track_metric(feat_name, value)

    def _extract_value(self, result: ImageAnalysisResult, feat_def) -> Optional[float]:
        """
        Extract feature value from result using the feature definition.

        Args:
            result: The image analysis result
            feat_def: Feature definition from registry

        Returns:
            Extracted numeric value or None
        """
        try:
            # Direct attribute access via sql_column
            if feat_def.sql_column:
                value = getattr(result, feat_def.sql_column, None)
                return float(value) if value is not None else None

            # Use extractor callable
            if feat_def.extractor:
                # Pass empty context for ingestion (no query-time data)
                value = feat_def.extractor(result, {})
                return float(value) if value is not None else None

            return None
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Could not extract {feat_def.name}: {e}")
            return None

    def _track_metric(self, name: str, value: Optional[float]):
        """Helper to safely add numeric values or count missing ones."""
        if value is None:
            self.missing_counts[name] += 1
            return

        # Validate it's a number (not bool)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            self.numeric_data[name].append(float(value))

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
