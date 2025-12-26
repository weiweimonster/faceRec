from __future__ import annotations

import cv2, torch, clip
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

@dataclass
class FaceData:
    """
    Structured data for storing a single face
    """
    bbox: List[int]
    embedding: np.ndarray
    confidence: float

@dataclass
class ImageAnalysisResult:
    """
    The complete AI analysis of a photo
    """
    # The output of CLIP model for semantic embedding
    semantic_vector: np.ndarray
    # The detected faces in the image
    faces: List[FaceData]
    original_width: int
    original_height: int

class FeatureExtractor:
    """
    The brain of the ingestion pipeline
    """

    def __init__(self, use_gpu: bool = True):
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f'Using device: {self.device}')
        self.face_app = FaceAnalysis(name='buffalo_l')
        self.face_app.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(640, 640))

        self.clip_mode, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)

    def process_image(self, image_path: str) -> ImageAnalysisResult:
        def is_face_too_small(
                bbox: Tuple[int, int, int, int],
                image_width: int,
                image_height: int,
                min_side_px: int = 32,
                min_area_ratio: float = 0.001,
        ):
            x1, y1, x2, y2 = bbox
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)

            if w < min_side_px or h < min_side_px:
                return True

            face_area = w * h
            image_area = image_width * image_height

            if image_area <= 0:
                return True

            if (face_area / image_area) < min_area_ratio:
                return True

            return False
        try:
            cv_img = cv2.imread(image_path)
            if cv_img is None:
               return None
            h, w, _ = cv_img.shape
            raw_faces = self.face_app.get(cv_img)
            preprocess_faces: List[FaceData] = []

            for face in raw_faces:
                # Filter out the faces that are too small
                if is_face_too_small(face.bbox, w, h):
                    continue
                preprocess_faces.append(FaceData(
                    face.bbox.astype(int).tolist(),
                    face.embedding,
                    float(face.det_score)
                ))
            pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            clip_input = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.clip_mode.encode_image(clip_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                semantic_vector = image_features.cpu().numpy()[0]

            return ImageAnalysisResult(semantic_vector, preprocess_faces, w, h)
        except Exception as e:
            raise e




