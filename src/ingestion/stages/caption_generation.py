"""
CaptionGenerationStage: Generate image captions using Qwen2-VL.
"""
from __future__ import annotations

from typing import List, Optional

from src.ingestion.stages.base import ProcessingStage, PipelineItem
from src.model.florence import VisionScanner
from src.util.logger import logger


class CaptionGenerationStage(ProcessingStage):
    """
    Generate descriptive captions for images using Qwen2-VL-2B.

    Uses the existing VisionScanner class which supports batch processing.

    Outputs:
    - item.result.caption: Text description of the image
    """

    def __init__(self, batch_size: int = 8):
        """
        Args:
            batch_size: Batch size for GPU processing (8 recommended for ~6GB VRAM)
        """
        self.batch_size = batch_size
        self.vision_scanner: Optional[VisionScanner] = None

    @property
    def name(self) -> str:
        return "caption_generation"

    @property
    def supports_gpu_batching(self) -> bool:
        return True

    def initialize(self) -> None:
        """Load Qwen2-VL model."""
        if self.vision_scanner is not None:
            return

        logger.info("Loading VisionScanner (Qwen2-VL)...")
        self.vision_scanner = VisionScanner()
        # Model is lazy-loaded on first use
        logger.info("VisionScanner initialized")

    def cleanup(self) -> None:
        """Release model resources."""
        if self.vision_scanner is not None:
            self.vision_scanner.unload()
            self.vision_scanner = None

    def process(self, items: List[PipelineItem]) -> List[PipelineItem]:
        """Generate captions for all processable items in batches."""
        if self.vision_scanner is None:
            self.initialize()

        # Get processable items with valid display paths
        indices = []
        paths = []

        for i, item in enumerate(items):
            if item.is_processable() and item.result.display_path:
                indices.append(i)
                paths.append(item.result.display_path)

        if not paths:
            return items

        try:
            # Use batch processing from VisionScanner
            captions = self.vision_scanner.extract_caption_batch(paths, batch_size=self.batch_size)

            # Map results back to items (preserving order)
            for i, caption in zip(indices, captions):
                items[i].result.caption = caption

        except Exception as e:
            logger.error(f"Batch caption generation failed: {e}")
            # Fall back to individual processing
            for idx in indices:
                item = items[idx]
                try:
                    caption = self.vision_scanner.extract_caption(item.result.display_path)
                    item.result.caption = caption
                except Exception as e2:
                    logger.warning(f"Caption generation failed for {item.result.display_path}: {e2}")
                    item.result.caption = ""

        return items
