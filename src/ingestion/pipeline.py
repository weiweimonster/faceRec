"""
Pipeline orchestrator for the ingestion pipeline.

Manages stage execution order based on dependencies, supports both
sequential (batch_size=1) and parallel (batch_size>1) processing modes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from tqdm import tqdm

from src.ingestion.stages.base import ProcessingStage, PipelineItem
from src.ingestion.stages import (
    PreprocessStage,
    FaceDetectionStage,
    PoseExtractionStage,
    CLIPEncodingStage,
    AestheticScoreStage,
    CaptionGenerationStage,
    CaptionEmbeddingStage,
    MetricsCollectionStage,
    PersistStage,
)
from src.db.storage import DatabaseManager
from src.util.logger import logger


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""
    total_images: int = 0
    processed: int = 0
    skipped: int = 0
    errors: int = 0
    total_time_s: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)

    @property
    def images_per_second(self) -> float:
        if self.total_time_s <= 0:
            return 0.0
        return self.processed / self.total_time_s

    def to_dict(self) -> dict:
        return {
            "total_images": self.total_images,
            "processed": self.processed,
            "skipped": self.skipped,
            "errors": self.errors,
            "total_time_s": round(self.total_time_s, 2),
            "images_per_second": round(self.images_per_second, 2),
            "stage_times": {k: round(v, 2) for k, v in self.stage_times.items()},
        }


class Pipeline:
    """
    Orchestrates the execution of processing stages.

    Handles dependency ordering, batch processing, and error recovery.
    Works identically for sequential (batch_size=1) and parallel (batch_size>1) modes.
    """

    def __init__(
        self,
        stages: List[ProcessingStage],
        batch_size: int = 1,
    ):
        """
        Args:
            stages: List of processing stages to run
            batch_size: Number of images to process in each batch
                       1 = sequential mode, >1 = parallel mode
        """
        self.stages = self._order_by_dependencies(stages)
        self.batch_size = batch_size
        self._initialized = False

    def _order_by_dependencies(self, stages: List[ProcessingStage]) -> List[ProcessingStage]:
        """
        Topological sort of stages based on their dependencies.

        Args:
            stages: Unordered list of stages

        Returns:
            Stages ordered so dependencies come before dependents
        """
        # Build name -> stage mapping
        stage_map = {s.name: s for s in stages}
        ordered = []
        visited = set()

        def visit(stage: ProcessingStage):
            if stage.name in visited:
                return
            visited.add(stage.name)

            # Visit dependencies first
            for dep_name in stage.dependencies:
                if dep_name in stage_map:
                    visit(stage_map[dep_name])
                else:
                    logger.warning(f"Stage '{stage.name}' depends on unknown stage '{dep_name}'")

            ordered.append(stage)

        for stage in stages:
            visit(stage)

        return ordered

    def initialize(self) -> None:
        """Initialize all stages (load models, etc.)."""
        if self._initialized:
            return

        logger.info("Initializing pipeline stages...")
        for stage in self.stages:
            logger.info(f"  Initializing {stage.name}...")
            stage.initialize()

        self._initialized = True
        logger.info("Pipeline initialized")

    def cleanup(self) -> None:
        """Cleanup all stages (unload models, etc.)."""
        logger.info("Cleaning up pipeline stages...")
        for stage in reversed(self.stages):
            try:
                stage.cleanup()
            except Exception as e:
                logger.warning(f"Cleanup failed for {stage.name}: {e}")

        self._initialized = False

    def run(
        self,
        file_paths: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> PipelineStats:
        """
        Run the pipeline on a list of file paths.

        Args:
            file_paths: List of image file paths to process
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            PipelineStats with timing and count information
        """
        self.initialize()

        stats = PipelineStats(total_images=len(file_paths))
        start_time = time.time()

        # Process in batches
        total_batches = (len(file_paths) + self.batch_size - 1) // self.batch_size

        for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(file_paths))
            batch_paths = file_paths[batch_start:batch_end]

            # Create pipeline items for this batch
            items = [PipelineItem.from_path(path) for path in batch_paths]

            # Run all stages on this batch
            for stage in self.stages:
                stage_start = time.time()

                try:
                    items = stage.process(items)
                except Exception as e:
                    logger.error(f"Stage {stage.name} failed on batch: {e}")
                    # Mark all items in batch as errors
                    for item in items:
                        if item.is_processable():
                            item.mark_error(f"Stage {stage.name} failed: {e}")

                stage_time = time.time() - stage_start
                stats.stage_times[stage.name] = stats.stage_times.get(stage.name, 0) + stage_time

            # Count results
            for item in items:
                if item.error:
                    stats.errors += 1
                elif item.skipped:
                    stats.skipped += 1
                else:
                    stats.processed += 1

            # Progress callback
            if progress_callback:
                progress_callback(batch_end, len(file_paths))

        stats.total_time_s = time.time() - start_time
        return stats


def create_default_pipeline(
    db: DatabaseManager,
    batch_size: int = 16,
    use_gpu: bool = True,
    cache_dir: str = "./photos/cache",
    collect_metrics: bool = False,
    metrics_output_path: str = "ingestion_metrics.json",
) -> Pipeline:
    """
    Create a pipeline with the default stage configuration.

    Args:
        db: DatabaseManager for persistence
        batch_size: Batch size (1=sequential, >1=parallel)
        use_gpu: Whether to use GPU for model inference
        cache_dir: Directory for HEIC conversion cache
        collect_metrics: Whether to collect and save ingestion metrics
        metrics_output_path: Path to save metrics JSON (if collect_metrics=True)

    Returns:
        Configured Pipeline instance
    """
    stages = [
        PreprocessStage(
            cache_dir=cache_dir,
            duplicate_checker=db.photo_exists,
            max_workers=4,
        ),
        FaceDetectionStage(use_gpu=use_gpu),
        PoseExtractionStage(use_gpu=use_gpu),
        CLIPEncodingStage(use_gpu=use_gpu, batch_size=batch_size),
        AestheticScoreStage(use_gpu=use_gpu),
        CaptionGenerationStage(batch_size=min(batch_size, 8)),  # Qwen2-VL uses more VRAM
        CaptionEmbeddingStage(batch_size=min(batch_size * 2, 32)),  # E5 is lighter
    ]

    # Optional metrics collection (before persist to capture all data)
    if collect_metrics:
        stages.append(MetricsCollectionStage(output_path=metrics_output_path))

    stages.append(PersistStage(db=db))

    return Pipeline(stages=stages, batch_size=batch_size)
