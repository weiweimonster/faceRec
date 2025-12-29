from dataclasses import dataclass
from typing import List
import numpy as np
from src.pose.pose import Pose

@dataclass
class FaceData:
    """
    Structured data for storing a single face
    """
    bbox: List[int]
    embedding: np.ndarray
    confidence: float
    shot_type: str = ""
    blur_score: float = -1.0
    brightness: float = -1.0
    yaw: float = -1.0
    pitch: float = -1.0
    roll: float = -1.0
    pose: Pose = None

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
    timestamp: str