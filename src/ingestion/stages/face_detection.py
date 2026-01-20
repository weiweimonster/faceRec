"""
FaceDetectionStage: Detect faces using InsightFace and extract embeddings.
"""
from __future__ import annotations

from typing import List, Optional
import torch
from insightface.app import FaceAnalysis

from src.ingestion.stages.base import ProcessingStage, PipelineItem
from src.common.types import FaceData
from src.util.image_util import is_face_too_small, calculate_face_quality, calculate_shot_type
from src.util.logger import logger


class FaceDetectionStage(ProcessingStage):
    """
    Detect faces in images using InsightFace (buffalo_l model).

    Outputs:
    - item.raw_faces: Raw InsightFace face objects (for pose extraction)
    - item.result.faces: List of FaceData with bbox, embedding, confidence, quality metrics
    """

    def __init__(self, use_gpu: bool = True, det_size: tuple = (640, 640)):
        """
        Args:
            use_gpu: Whether to use GPU for face detection
            det_size: Detection input size (width, height)
        """
        self.use_gpu = use_gpu
        self.det_size = det_size
        self.face_app: Optional[FaceAnalysis] = None

    @property
    def name(self) -> str:
        return "face_detection"

    @property
    def supports_gpu_batching(self) -> bool:
        # InsightFace processes images one at a time internally
        return False

    def initialize(self) -> None:
        """Load InsightFace model."""
        if self.face_app is not None:
            return

        device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
        ctx_id = 0 if device == 'cuda' else -1

        logger.info(f"Loading InsightFace model (ctx_id={ctx_id})...")
        self.face_app = FaceAnalysis(name='buffalo_l')
        self.face_app.prepare(ctx_id=ctx_id, det_size=self.det_size)
        logger.info("InsightFace model loaded")

    def cleanup(self) -> None:
        """Release model resources."""
        self.face_app = None

    def process(self, items: List[PipelineItem]) -> List[PipelineItem]:
        """Detect faces in all processable items."""
        if self.face_app is None:
            self.initialize()

        for item in items:
            if not item.is_processable():
                continue

            if item.cv_image is None:
                item.mark_error("No CV image available for face detection")
                continue

            try:
                self._detect_faces(item)
            except Exception as e:
                item.mark_error(f"Face detection failed: {e}")
                logger.error(f"Face detection failed for {item.result.original_path}: {e}")

        return items

    def _detect_faces(self, item: PipelineItem) -> None:
        """Detect faces in a single image."""
        cv_img = item.cv_image
        h, w = cv_img.shape[:2]

        # Run InsightFace detection
        raw_faces = self.face_app.get(cv_img)

        faces: List[FaceData] = []
        kept_raw_faces = []

        for face in raw_faces:
            bbox = face.bbox.astype(int).tolist()
            x1, y1, x2, y2 = bbox

            # Filter out faces that are too small
            if is_face_too_small((x1, y1, x2, y2), image_width=w, image_height=h):
                continue

            kept_raw_faces.append(face)

            # Calculate quality metrics
            blur, brightness = calculate_face_quality(cv_img, bbox)
            shot_type = calculate_shot_type(cv_img, bbox)

            face_data = FaceData(
                bbox=bbox,
                embedding=face.embedding,
                confidence=float(face.det_score),
                blur_score=blur,
                brightness=brightness,
                shot_type=shot_type,
            )
            faces.append(face_data)

        item.raw_faces = kept_raw_faces
        item.result.faces = faces
