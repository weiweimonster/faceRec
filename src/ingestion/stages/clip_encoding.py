"""
CLIPEncodingStage: Extract CLIP semantic vectors from images.
"""
from __future__ import annotations

from typing import List, Optional
import torch
import clip
import numpy as np
from PIL import Image

from src.ingestion.stages.base import ProcessingStage, PipelineItem
from src.util.logger import logger


class CLIPEncodingStage(ProcessingStage):
    """
    Extract CLIP ViT-L/14 embeddings for semantic search.

    Outputs:
    - item.clip_features: Raw CLIP features (for aesthetic scoring)
    - item.result.semantic_vector: Normalized 768D embedding
    """

    def __init__(self, use_gpu: bool = True, batch_size: int = 16):
        """
        Args:
            use_gpu: Whether to use GPU
            batch_size: Batch size for GPU processing
        """
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.model = None
        self.preprocess = None
        self.device = None

    @property
    def name(self) -> str:
        return "clip_encoding"

    @property
    def supports_gpu_batching(self) -> bool:
        return True

    def initialize(self) -> None:
        """Load CLIP model."""
        if self.model is not None:
            return

        self.device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading CLIP ViT-L/14 on {self.device}...")
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        logger.info("CLIP model loaded")

    def cleanup(self) -> None:
        """Release model resources."""
        if self.model is not None:
            del self.model
            del self.preprocess
            self.model = None
            self.preprocess = None
            torch.cuda.empty_cache()

    def process(self, items: List[PipelineItem]) -> List[PipelineItem]:
        """Encode images with CLIP in batches."""
        if self.model is None:
            self.initialize()

        # Get processable items with PIL images
        indices, processable = self.get_processable(items)
        valid_indices = []
        valid_items = []

        for idx, item in zip(indices, processable):
            if item.pil_image is not None:
                valid_indices.append(idx)
                valid_items.append(item)
            elif item.cv_image is not None:
                # Try to create PIL image from CV image
                try:
                    import cv2
                    item.pil_image = Image.fromarray(cv2.cvtColor(item.cv_image, cv2.COLOR_BGR2RGB))
                    valid_indices.append(idx)
                    valid_items.append(item)
                except Exception as e:
                    logger.warning(f"Could not create PIL image: {e}")

        if not valid_items:
            return items

        # Process in batches
        for batch_start in range(0, len(valid_items), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(valid_items))
            batch_items = valid_items[batch_start:batch_end]

            try:
                self._process_batch(batch_items)
            except Exception as e:
                logger.error(f"CLIP batch encoding failed: {e}")
                # Fall back to individual processing
                for item in batch_items:
                    try:
                        self._process_single(item)
                    except Exception as e2:
                        item.mark_error(f"CLIP encoding failed: {e2}")

        return items

    def _process_batch(self, items: List[PipelineItem]) -> None:
        """Process a batch of items on GPU."""
        # Preprocess images
        batch_tensors = []
        for item in items:
            tensor = self.preprocess(item.pil_image)
            batch_tensors.append(tensor)

        # Stack into batch
        batch_input = torch.stack(batch_tensors).to(self.device)

        # Encode batch
        with torch.no_grad():
            image_features = self.model.encode_image(batch_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Store results
            for i, item in enumerate(items):
                # Store raw features for aesthetic scoring (keep on GPU)
                item.clip_features = image_features[i:i+1]
                # Store normalized numpy array for DB
                item.result.semantic_vector = image_features[i].cpu().numpy()

    def _process_single(self, item: PipelineItem) -> None:
        """Process a single item (fallback)."""
        tensor = self.preprocess(item.pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            item.clip_features = image_features
            item.result.semantic_vector = image_features[0].cpu().numpy()
