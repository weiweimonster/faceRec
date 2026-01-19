from __future__ import annotations

import cv2, torch, clip
import numpy as np
from PIL import Image
from pathlib import Path
from insightface.app import FaceAnalysis
from typing import List, Tuple, Optional, Any
from src.pose.pose_extractor import PoseExtractor
from src.pose.pose import Pose
from src.util.image_util import is_face_too_small, calculate_face_quality, calculate_shot_type, get_exif_timestamp, \
    get_exif_iso, get_disk_timestamp, compute_global_visual_stats, extract_time_features
from src.common.types import FaceData, ImageAnalysisResult
from src.util.logger import logger
from src.model.aesthetic_predictor import AestheticPredictor
from src.model.text_embedder import TextEmbedder
from src.model.florence import VisionScanner

# Project root: go up from src/ingestion/processor.py -> src/ingestion -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class FeatureExtractor:
    """
    The brain of the ingestion pipeline
    """

    def __init__(self, use_gpu: bool = True):
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        logger.info(f'Using device: {self.device}')
        self.face_app = FaceAnalysis(name='buffalo_l')
        self.face_app.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(640, 640))

        self.clip_mode, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)
        self.pose_extractor = PoseExtractor(gpu=use_gpu)
        self.aesthetic_predictor = AestheticPredictor(
            input_size=768,
            model_path=str(PROJECT_ROOT / "sac_logos_ava1-l14-linearMSE.pth"),
            device=self.device,
        )

        self.vision_scanner = VisionScanner()
        self.text_embedder = TextEmbedder()

    def process_image(self, image_path: str, raw_path: str) -> ImageAnalysisResult:
        try:
            cv_img = cv2.imread(image_path)
            if cv_img is None:
               return None
            h, w, _ = cv_img.shape

            global_stats = compute_global_visual_stats(cv_img)

            raw_faces = self.face_app.get(cv_img)
            preprocess_faces: List[FaceData] = []

            kept_raw_faces: List[Any] = []

            # Iterate through each faces
            for face in raw_faces:
                bbox = face.bbox.astype(int).tolist()
                x1, y1, x2, y2 = bbox

                # Filter out faces that are too small
                if is_face_too_small((x1, y1, x2, y2), image_width=w, image_height=h):
                    continue
                kept_raw_faces.append(face)
                # Calculate quality metrics
                blur, bright = calculate_face_quality(cv_img, bbox)
                # Calculate shot type
                shot_type = calculate_shot_type(cv_img, bbox)
                preprocess_faces.append(FaceData(
                    bbox=face.bbox.astype(int).tolist(),
                    embedding=face.embedding,
                    confidence=float(face.det_score),
                    blur_score=blur,
                    brightness=bright,
                    shot_type=shot_type
                ))

            poses = self.pose_extractor.extract_pose_from_faces(cv_img, kept_raw_faces)

            # assert the index we are using is correct
            assert len(poses) == len(preprocess_faces), (len(poses), len(preprocess_faces))

            # Insert pose information into FaceData
            for pose, fd in zip(poses, preprocess_faces):
                fd.yaw, fd.pitch, fd.roll = pose
                fd.pose = Pose.from_angles(fd.yaw, fd.pitch)

            pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            clip_input = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.clip_mode.encode_image(clip_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                semantic_vector = image_features.cpu().numpy()[0]

                # Calculate aesthetic score
                # Note: CLIP uses Half Prevision, and AestheticPredictor use Full floating point precision
                # TODO: Write test cases to ensure that the data types are matching
                aesthetic_score = self.aesthetic_predictor(image_features.to(self.device).float()).item()

            caption = self.vision_scanner.extract_caption(image_path)
            caption_vector = self.text_embedder.embed(caption)

            timestamp = get_exif_timestamp(image_path)
            if not timestamp:
                logger.debug(f"No timestamp found for {image_path}. Using disk timestamp")
                timestamp = get_disk_timestamp(raw_path)

            month = None
            time_period = None
            if timestamp:
                month, time_period = extract_time_features(timestamp)

            iso = get_exif_iso(image_path)

            if not iso:
                logger.debug(f"No iso value found for {image_path}")

            # TODO: Write a test cases to ensure that we are using preprocess_faces, and not kept_raw_faces
            # TODO: Write a test cases to ensure that we are saving timestamp
            return ImageAnalysisResult(
                original_path=raw_path,
                photo_id=None, # Used in ranker later, for the purpose of ingestion, we don't need it
                display_path=image_path,
                semantic_vector=semantic_vector,
                original_width=w,
                original_height=h,
                aesthetic_score=aesthetic_score,
                faces=preprocess_faces,
                timestamp=timestamp,
                month=month,
                time_period=time_period,
                iso=iso,
                global_blur=global_stats["global_blur"],
                global_brightness=global_stats["global_brightness"],
                global_contrast=global_stats["global_contrast"],
                caption=caption,
                caption_vector=caption_vector
            )
        except Exception as e:
            raise e





