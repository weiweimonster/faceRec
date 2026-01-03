from dataclasses import dataclass
from typing import List, Optional
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
    name: Optional[str] = None

@dataclass
class ImageAnalysisResult:
    """
    The complete AI analysis of a photo
    """
    # Essential keys (always loaded)
    original_path: Optional[str]
    photo_id: Optional[str]
    display_path: Optional[str]

    # Lazy fields (None until hydrated)
    timestamp: Optional[str] = None
    semantic_vector: Optional[np.ndarray] = None
    original_width: Optional[int] = None
    original_height: Optional[int] = None
    aesthetic_score: Optional[float] = None
    iso: Optional[int] = None
    global_blur: float = 0.0
    global_brightness: float = 0.0
    global_contrast: float = 0.0

    # Complex nested data
    faces: Optional[List[FaceData]] = None