"""
CaptionEmbeddingStage: Generate embeddings for captions using E5.
"""
from __future__ import annotations

from typing import List, Optional

from src.ingestion.stages.base import ProcessingStage, PipelineItem
from src.model.text_embedder import TextEmbedder
from src.util.logger import logger


class CaptionEmbeddingStage(ProcessingStage):
    """
    Generate embeddings for image captions using multilingual-e5-large.

    Dependencies:
    - caption_generation: Requires caption text

    Outputs:
    - item.result.caption_vector: List of floats (E5 embedding)
    """

    def __init__(self, batch_size: int = 32):
        """
        Args:
            batch_size: Batch size for GPU processing (32 is safe for E5-Large)
        """
        self.batch_size = batch_size
        self.embedder: Optional[TextEmbedder] = None

    @property
    def name(self) -> str:
        return "caption_embedding"

    @property
    def dependencies(self) -> List[str]:
        return ["caption_generation"]

    @property
    def supports_gpu_batching(self) -> bool:
        return True

    def initialize(self) -> None:
        """Load E5 model (singleton, may already be loaded)."""
        if self.embedder is not None:
            return

        logger.info("Initializing TextEmbedder (E5)...")
        self.embedder = TextEmbedder()
        logger.info("TextEmbedder initialized")

    def cleanup(self) -> None:
        """TextEmbedder is a singleton, don't cleanup."""
        pass

    def process(self, items: List[PipelineItem]) -> List[PipelineItem]:
        """Generate caption embeddings for all items with captions."""
        if self.embedder is None:
            self.initialize()

        # Collect items with valid captions
        indices = []
        captions = []

        for i, item in enumerate(items):
            if item.is_processable() and item.result.caption:
                indices.append(i)
                captions.append(item.result.caption)

        if not captions:
            return items

        try:
            # Use batch embedding
            embeddings = self.embedder.embed_batch(captions, input_type="passage")

            # Map results back
            for i, embedding in zip(indices, embeddings):
                items[i].result.caption_vector = embedding

        except Exception as e:
            logger.error(f"Batch caption embedding failed: {e}")
            # Fall back to individual processing
            for idx in indices:
                item = items[idx]
                try:
                    embedding = self.embedder.embed(item.result.caption, input_type="passage")
                    item.result.caption_vector = embedding
                except Exception as e2:
                    logger.warning(f"Caption embedding failed for {item.result.display_path}: {e2}")

        return items
