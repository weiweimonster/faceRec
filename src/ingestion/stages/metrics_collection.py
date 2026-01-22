"""
MetricsCollectionStage: Collects statistics from processed items for analysis.

This is an optional stage that can be added to the pipeline to track
feature distributions and generate metrics reports.
"""
from __future__ import annotations

from typing import List, Optional

from src.ingestion.stages.base import ProcessingStage, PipelineItem
from src.ingestion.metrics_tracker import IngestionMetricsTracker
from src.util.logger import logger


class MetricsCollectionStage(ProcessingStage):
    """
    Collects metrics from processed items for data analysis.

    This is a pass-through stage - it doesn't modify items, just collects
    statistics from them. Add this stage after all processing is complete
    (typically before or after PersistStage).

    Usage:
        # Add to pipeline when you want metrics
        stages = [
            PreprocessStage(...),
            FaceDetectionStage(...),
            ...
            MetricsCollectionStage(output_path="metrics.json"),  # Optional
            PersistStage(...),
        ]
    """

    def __init__(self, output_path: str = "ingestion_metrics.json"):
        """
        Args:
            output_path: Path to save the metrics report JSON
        """
        self.output_path = output_path
        self.tracker = IngestionMetricsTracker()

    @property
    def name(self) -> str:
        return "metrics_collection"

    @property
    def supports_gpu_batching(self) -> bool:
        return False  # CPU-only, just collecting stats

    def process(self, items: List[PipelineItem]) -> List[PipelineItem]:
        """
        Collect metrics from items. Pass-through - doesn't modify items.
        """
        for item in items:
            if not item.skipped and not item.error and item.result:
                self.tracker.update(item.result)

        return items

    def cleanup(self):
        """Generate and save the metrics report when pipeline finishes."""
        if self.tracker.total_photos > 0:
            report = self.tracker.finalize_report(self.output_path)
            logger.info(f"Metrics collected for {self.tracker.total_photos} photos")
        else:
            logger.warning("No photos processed, skipping metrics report")