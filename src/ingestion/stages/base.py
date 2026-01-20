"""
Base class for processing stages in the ingestion pipeline.

This module defines the abstract interface that all processing stages must implement,
enabling a unified pipeline that works for both sequential (batch_size=1) and
parallel (batch_size>1) processing modes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any
import numpy as np
from src.common.types import ImageAnalysisResult, FaceData


@dataclass
class PipelineItem:
    """
    Thin wrapper around ImageAnalysisResult for pipeline processing.

    Holds the final output object (ImageAnalysisResult) plus transient
    processing state that is not persisted to the database.
    """
    # The final output object - stages write directly to this
    result: ImageAnalysisResult

    # Transient processing state (not persisted, garbage collected after pipeline)
    file_hash: str = ""
    cv_image: Optional[np.ndarray] = None
    pil_image: Optional[Any] = None  # PIL.Image
    raw_faces: List[Any] = field(default_factory=list)  # InsightFace raw face objects
    clip_features: Optional[Any] = None  # torch.Tensor for aesthetic scoring

    # Pipeline control flags
    skipped: bool = False
    skip_reason: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def from_path(cls, raw_path: str) -> "PipelineItem":
        """Create a new PipelineItem from a file path."""
        result = ImageAnalysisResult(
            original_path=raw_path,
            photo_id=None,
            display_path=None,
            faces=[]
        )
        return cls(result=result)

    def mark_skipped(self, reason: str) -> None:
        """Mark this item as skipped with a reason."""
        self.skipped = True
        self.skip_reason = reason

    def mark_error(self, error: str) -> None:
        """Mark this item as having an error."""
        self.error = error
        self.skipped = True

    def is_processable(self) -> bool:
        """Check if this item should be processed (not skipped/errored)."""
        return not self.skipped and self.error is None


class ProcessingStage(ABC):
    """
    Abstract base class for pipeline stages.

    Each stage processes a batch of items and returns them with additional
    data populated. Stages can declare dependencies on other stages.

    The same implementation works for both:
    - Sequential mode (batch_size=1): Items processed one at a time
    - Parallel mode (batch_size>1): Items processed in batches for GPU efficiency
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this stage."""
        pass

    @property
    def dependencies(self) -> List[str]:
        """
        List of stage names that must run before this stage.

        Returns:
            List of stage names this stage depends on.
        """
        return []

    @property
    def supports_gpu_batching(self) -> bool:
        """
        Whether this stage benefits from larger batches on GPU.

        Stages that use GPU models (CLIP, Qwen2-VL, E5) should return True.
        CPU-bound stages (hash calculation, EXIF extraction) return False.
        """
        return False

    @abstractmethod
    def process(self, items: List[PipelineItem]) -> List[PipelineItem]:
        """
        Process a batch of items.

        This method must:
        1. Handle items that are already marked as skipped (pass through unchanged)
        2. Preserve the original list order (critical for index tracking)
        3. Handle errors gracefully (mark individual items, don't crash the batch)

        Args:
            items: List of PipelineItem objects to process

        Returns:
            The same list with additional fields populated
        """
        pass

    def initialize(self) -> None:
        """
        Initialize any resources needed by this stage (e.g., load models).

        Called once before processing begins. Override in subclasses that
        need lazy initialization.
        """
        pass

    def cleanup(self) -> None:
        """
        Release any resources held by this stage.

        Called after processing completes. Override in subclasses that
        need to free GPU memory or close connections.
        """
        pass

    def get_processable(self, items: List[PipelineItem]) -> tuple[List[int], List[PipelineItem]]:
        """
        Helper to filter out skipped items for processing.

        Returns:
            Tuple of (indices, items) for items that should be processed
        """
        indices = []
        processable = []
        for i, item in enumerate(items):
            if item.is_processable():
                indices.append(i)
                processable.append(item)
        return indices, processable
