from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, List, TypedDict
from enum import Enum

class FeatureType(Enum):
    """Feature category for organization"""
    GLOBAL = "global"  # Image-level quality metrics
    FACE = "face"  # Face-specific metrics
    SEMANTIC = "semantic"  # Search relevance scores
    META = "meta"  # Temporal/camera metadata
    COMPUTED = "computed"  # Derived during ranking (e.g., MMR rank)

class FeatureDataType(Enum):
    """Expected data type for validation"""
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"

@dataclass
class FeatureDefinition:
    """
    Declarative feature specification

    Attributes:
        name: Unique feature identifier
        category: Feature category from FeatureType
        dtype: Expected data type
        sql_column: Direct attribute name on ImageAnalysisResult (e.g., "aesthetic_score")
        extractor: Callable for computed features, signature: (result, context) -> value
        normalization: Min/max bounds for heuristic normalization {"min": x, "max": y}
        is_trainable: Whether to include in XGBoost training (False for MMR rank, etc.)
        default_value: Fallback when feature extraction fails
        description: Human-readable documentation

    Example:
        FeatureDefinition(
            name="aesthetic_score",
            category=FeatureType.GLOBAL,
            dtype=FeatureDataType.FLOAT,
            sql_column="aesthetic_score",
            is_trainable=True,
            normalization={"min": 4.0, "max": 5.5},
            description="Neural aesthetic quality prediction"
        )
    """
    name: str # Feature name for ML (e.g.. "g_blur")
    category: FeatureType # Feature category
    dtype: FeatureDataType # Expected data type

    # Extraction strategy (pick One)
    sql_column: Optional[str] = None
    extractor: Optional[Callable] = None

    # Heuristic ranker optimization
    normalization: Optional[Dict[str, float]] = None

    # Training configuration
    is_trainable: bool = True
    default_value: float = 0.0

    # Documentation
    description: str = ""


# ============================================
# GLOBAL FEATURE REGISTRY
# ============================================
FEATURE_REGISTRY: Dict[str, FeatureDefinition] = {

    # ==================== GLOBAL QUALITY FEATURES ====================

    "aesthetic_score": FeatureDefinition(
        name="aesthetic_score",
        category=FeatureType.GLOBAL,
        dtype=FeatureDataType.FLOAT,
        sql_column="aesthetic_score",
        normalization={"min": 4.0, "max": 5.5},
        is_trainable=True,
        description="Neural aesthetic quality prediction from LAION model"
    ),

    "g_blur": FeatureDefinition(
        name="g_blur",
        category=FeatureType.GLOBAL,
        dtype=FeatureDataType.FLOAT,
        sql_column="global_blur",
        normalization={"min": 21.0, "max": 1400.0},
        is_trainable=True,
        description="Laplacian variance of entire image (higher = sharper)"
    ),

    "g_brightness": FeatureDefinition(
        name="g_brightness",
        category=FeatureType.GLOBAL,
        dtype=FeatureDataType.FLOAT,
        sql_column="global_brightness",
        normalization={"min": 0.0, "max": 255.0},
        is_trainable=True,
        description="Average pixel intensity across image"
    ),

    "g_contrast": FeatureDefinition(
        name="g_contrast",
        category=FeatureType.GLOBAL,
        dtype=FeatureDataType.FLOAT,
        sql_column="global_contrast",
        normalization={"min": 35.0, "max": 80.0},
        is_trainable=True,
        description="Standard deviation of pixel values (dynamic range)"
    ),

    "g_iso": FeatureDefinition(
        name="g_iso",
        category=FeatureType.GLOBAL,
        dtype=FeatureDataType.INT,
        sql_column="iso",
        normalization={"min": 40, "max": 1600},
        is_trainable=True,
        default_value=-1.0,
        description="Camera ISO sensitivity (lower = less noise, -1 = unknown)"
    ),

    # ==================== SEMANTIC FEATURES ====================

    "semantic_score": FeatureDefinition(
        name="semantic_score",
        category=FeatureType.SEMANTIC,
        dtype=FeatureDataType.FLOAT,
        extractor=lambda result, ctx: ctx.get("semantic_score", 0.0),
        is_trainable=True,
        description="CLIP cosine similarity between query and image"
    ),

    "caption_score": FeatureDefinition(
        name="caption_score",
        category=FeatureType.SEMANTIC,
        dtype=FeatureDataType.FLOAT,
        extractor=lambda result, ctx: ctx.get("caption_score", 0.0),
        is_trainable=True,
        description="E5 cosine similarity between query and caption embedding"
    ),

    "mmr_rank": FeatureDefinition(
        name="mmr_rank",
        category=FeatureType.COMPUTED,
        dtype=FeatureDataType.INT,
        extractor=lambda result, ctx: ctx.get("mmr_rank", -1),
        is_trainable=False,  # Computed AFTER ranking, not an input feature
        default_value=-1.0,
        description="Position in MMR-diversified results (1 = top result)"
    ),

    "final_relevance": FeatureDefinition(
        name="final_relevance",
        category=FeatureType.COMPUTED,
        dtype=FeatureDataType.FLOAT,
        extractor=lambda result, ctx: ctx.get("final_relevance", 0.0),
        is_trainable=False,
        description="Combined weighted score from ranking strategy"
    ),

    # ==================== FACE FEATURES ====================

    "f_blur": FeatureDefinition(
        name="f_blur",
        category=FeatureType.FACE,
        dtype=FeatureDataType.FLOAT,
        extractor=lambda result, ctx: _extract_face_attr(
            result,
            ctx.get("target_face_name"),
            lambda face: face.blur_score if face.blur_score is not None else -1.0
        ),
        normalization={"min": 30, "max": 1200},
        is_trainable=True,
        default_value=-1.0,
        description="Laplacian variance of face crop (-1 = no face)"
    ),

    "f_conf": FeatureDefinition(
        name="f_conf",
        category=FeatureType.FACE,
        dtype=FeatureDataType.FLOAT,
        extractor=lambda result, ctx: _extract_face_attr(
            result,
            ctx.get("target_face_name"),
            lambda face: face.confidence if face.confidence is not None else 0.0
        ),
        is_trainable=True,
        default_value=0.0,
        description="Face detection confidence [0-1]"
    ),

    "f_orient_score": FeatureDefinition(
        name="f_orient_score",
        category=FeatureType.FACE,
        dtype=FeatureDataType.FLOAT,
        extractor=lambda result, ctx: 0.0,  # TODO: Implement pose scoring
        is_trainable=True,
        default_value=0.0,
        description="Pose alignment score [0-1] (1 = perfect frontal)"
    ),

    "f_width": FeatureDefinition(
        name="f_width",
        category=FeatureType.FACE,
        dtype=FeatureDataType.INT,
        extractor=lambda result, ctx: _extract_face_attr(
            result,
            ctx.get("target_face_name"),
            lambda face: (face.bbox[2] - face.bbox[0]) if face.bbox and len(face.bbox) == 4 else 0.0
        ),
        normalization={"min": 50, "max": 550},
        is_trainable=True,
        default_value=0.0,
        description="Face bounding box width in pixels"
    ),

    "f_height": FeatureDefinition(
        name="f_height",
        category=FeatureType.FACE,
        dtype=FeatureDataType.INT,
        extractor=lambda result, ctx: _extract_face_attr(
            result,
            ctx.get("target_face_name"),
            lambda face: (face.bbox[3] - face.bbox[1]) if face.bbox and len(face.bbox) == 4 else 0.0
        ),
        normalization={"min": 50, "max": 550},
        is_trainable=True,
        default_value=0.0,
        description="Face bounding box height in pixels"
    ),

    "f_yaw": FeatureDefinition(
        name="f_yaw",
        category=FeatureType.FACE,
        dtype=FeatureDataType.FLOAT,
        extractor=lambda result, ctx: _extract_face_attr(
            result,
            ctx.get("target_face_name"),
            lambda face: face.yaw if face.yaw is not None else 0.0
        ),
        is_trainable=True,
        default_value=0.0,
        description="Head rotation left/right in degrees [-90, 90]"
    ),

    "f_pitch": FeatureDefinition(
        name="f_pitch",
        category=FeatureType.FACE,
        dtype=FeatureDataType.FLOAT,
        extractor=lambda result, ctx: _extract_face_attr(
            result,
            ctx.get("target_face_name"),
            lambda face: face.pitch if face.pitch is not None else 0.0
        ),
        is_trainable=True,
        default_value=0.0,
        description="Head rotation up/down in degrees [-90, 90]"
    ),

    "f_roll": FeatureDefinition(
        name="f_roll",
        category=FeatureType.FACE,
        dtype=FeatureDataType.FLOAT,
        extractor=lambda result, ctx: _extract_face_attr(
            result,
            ctx.get("target_face_name"),
            lambda face: face.roll if face.roll is not None else 0.0
        ),
        is_trainable=True,
        default_value=0.0,
        description="Head tilt in degrees [-180, 180]"
    ),

    # ==================== METADATA FEATURES ====================

    "year": FeatureDefinition(
        name="year",
        category=FeatureType.META,
        dtype=FeatureDataType.INT,
        extractor=lambda result, ctx: _extract_year(result.timestamp),
        is_trainable=True,
        default_value=-1.0,
        description="Year from EXIF timestamp (-1 = unknown)"
    ),

    "month": FeatureDefinition(
        name="month",
        category=FeatureType.META,
        dtype=FeatureDataType.INT,
        sql_column="month",
        is_trainable=True,
        default_value=-1.0,
        description="Month from EXIF [1-12] (-1 = unknown)"
    ),

    "date": FeatureDefinition(
        name="date",
        category=FeatureType.META,
        dtype=FeatureDataType.INT,
        extractor=lambda result, ctx: _extract_date(result.timestamp),
        is_trainable=True,
        default_value=-1.0,
        description="Day of month [1-31] (-1 = unknown)"
    ),

    "time_period": FeatureDefinition(
        name="time_period",
        category=FeatureType.META,
        dtype=FeatureDataType.FLOAT,
        extractor=lambda result, ctx: _encode_time_period(result.time_period),
        is_trainable=True,
        default_value=0.0,
        description="Time of day encoded: morning=1, afternoon=2, evening=3, night=4"
    ),

    "has_face": FeatureDefinition(
        name="has_face",
        category=FeatureType.META,
        dtype=FeatureDataType.BOOL,
        extractor=lambda result, ctx: 1.0 if (result.faces and len(result.faces) > 0) else 0.0,
        is_trainable=True,
        description="Boolean: 1.0 if photo contains any faces, 0.0 otherwise"
    ),
}

# ============================================
# HELPER FUNCTIONS
# ============================================

def _extract_face_attr(result, target_name: Optional[str], attr_extractor: Callable):
    """
    Generic face attribute extractor.

    Handles face selection logic:
    1. If target_name provided: Find face by name (case-insensitive)
    2. Fallback: Use first face
    3. If no faces: Return None (caller handles default)

    Args:
        result: ImageAnalysisResult object
        target_name: Optional person name to find
        attr_extractor: Lambda to extract attribute from selected FaceData

    Returns:
        Extracted attribute value or None
    """
    if not result.faces or len(result.faces) == 0:
        return None

    # Select target face
    selected_face = None

    if target_name:
        # Find by name (case-insensitive)
        for face in result.faces:
            if face.name and face.name.lower() == target_name.lower():
                selected_face = face
                break

    # Fallback to first face
    if not selected_face:
        selected_face = result.faces[0]

    # Extract attribute
    return attr_extractor(selected_face)

def _encode_time_period(period: Optional[str]) -> float:
    """Convert time_period string to numeric encoding for ML models"""
    if not period:
        return 0.0
    mapping = {
        "morning": 1.0,
        "afternoon": 2.0,
        "evening": 3.0,
        "night": 4.0
    }
    return mapping.get(period.lower(), 0.0)


def _extract_year(timestamp: Optional[str]) -> int:
    """
    Extract year from timestamp.

    Expected format: "YYYY-MM-DD HH:MM:SS" (e.g., "2023-06-15 12:00:00")
    """
    if not timestamp:
        return -1
    ts = str(timestamp)
    if len(ts) >= 4:
        try:
            return int(ts[:4])
        except ValueError:
            return -1
    return -1


def _extract_date(timestamp: Optional[str]) -> int:
    """
    Extract day of month from timestamp.

    Expected format: "YYYY-MM-DD HH:MM:SS" (e.g., "2023-06-15 12:00:00")
    Day is at position [8:10]
    """
    if not timestamp:
        return -1
    ts = str(timestamp)

    if len(ts) >= 10:
        try:
            return int(ts[8:10])
        except ValueError:
            return -1

    return -1

# ============================================
# REGISTRY ACCESS FUNCTIONS
# ============================================

def get_feature(name: str) -> Optional[FeatureDefinition]:
    """Get feature definition by name"""
    return FEATURE_REGISTRY.get(name)

def get_all_features() -> Dict[str, FeatureDefinition]:
    """Get all registered features"""
    return FEATURE_REGISTRY

def get_trainable_features() -> List[str]:
    """
    Get list of feature names suitable for XGBoost training.
    Excludes computed features like mmr_rank and final_relevance.
    """
    return [name for name, feat in FEATURE_REGISTRY.items() if feat.is_trainable]

def get_features_by_category(category: FeatureType) -> Dict[str, FeatureDefinition]:
    """Get all features in a specific category"""
    return {name: feat for name, feat in FEATURE_REGISTRY.items() if feat.category == category}

def get_normalization_bounds(feature_name: str) -> Optional[Dict[str, float]]:
    """
    Get normalization bounds for a feature (used by heuristic ranker).

    Returns:
        {"min": x, "max": y} or None if no bounds defined
    """
    feat = FEATURE_REGISTRY.get(feature_name)
    return feat.normalization if feat else None

def get_feature_names() -> List[str]:
    """Get all registered feature names"""
    return list(FEATURE_REGISTRY.keys())

