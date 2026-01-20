"""
AestheticScoreStage: Calculate aesthetic quality score from CLIP features.
"""
from __future__ import annotations

from typing import List, Optional
from pathlib import Path

from src.ingestion.stages.base import ProcessingStage, PipelineItem
from src.model.aesthetic_predictor import AestheticPredictor
from src.util.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class AestheticScoreStage(ProcessingStage):
    """
    Calculate aesthetic quality scores using CLIP features.

    Dependencies:
    - clip_encoding: Requires clip_features

    Outputs:
    - item.result.aesthetic_score: Float score (typically 4.0-5.5 range)
    """

    def __init__(self, use_gpu: bool = True):
        """
        Args:
            use_gpu: Whether to use GPU
        """
        self.use_gpu = use_gpu
        self.predictor: Optional[AestheticPredictor] = None
        self.device = None

    @property
    def name(self) -> str:
        return "aesthetic_score"

    @property
    def dependencies(self) -> List[str]:
        return ["clip_encoding"]

    @property
    def supports_gpu_batching(self) -> bool:
        # The predictor is a small MLP, batching has minimal benefit
        return False

    def initialize(self) -> None:
        """Load aesthetic predictor model."""
        if self.predictor is not None:
            return

        import torch
        self.device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'

        model_path = PROJECT_ROOT / "sac_logos_ava1-l14-linearMSE.pth"
        logger.info(f"Loading AestheticPredictor from {model_path}...")

        self.predictor = AestheticPredictor(
            input_size=768,
            model_path=str(model_path),
            device=self.device,
        )
        logger.info("AestheticPredictor loaded")

    def cleanup(self) -> None:
        """Release model resources."""
        self.predictor = None

    def process(self, items: List[PipelineItem]) -> List[PipelineItem]:
        """Calculate aesthetic scores for items with CLIP features."""
        if self.predictor is None:
            self.initialize()

        for item in items:
            if not item.is_processable():
                continue

            if item.clip_features is None:
                logger.warning(f"No CLIP features for {item.result.original_path}")
                continue

            try:
                # AestheticPredictor expects float32, CLIP outputs float16
                features = item.clip_features.to(self.device).float()
                score = self.predictor(features).item()
                item.result.aesthetic_score = score
            except Exception as e:
                logger.warning(f"Aesthetic scoring failed for {item.result.original_path}: {e}")
                # Not fatal - continue without aesthetic score

        return items
