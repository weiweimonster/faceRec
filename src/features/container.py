from typing import Dict, Any, Optional, List
from src.common.types import ImageAnalysisResult, FaceData
from src.util.logger import logger
from .registry import FEATURE_REGISTRY, FeatureDefinition, get_trainable_features

class FeatureExtractor:
    """
    Centralized feature extraction using the registry.

    This class provides a single extraction interface for all features,
    handling both direct attributes and computed values.
    """

    @staticmethod
    def extract_from_result(
            result: ImageAnalysisResult,
            context: Optional[Dict[str, Any]] = None,
            target_face_name: Optional[str] = None,
            feature_subset: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Extract features from an ImageAnalysisResult object.

        Args:
            result: The image analysis result object
            context: Dynamic scores dict (e.g., {"semantic_score": 0.85, "mmr_rank": 3})
            target_face_name: If provided, extract face features for this specific person
            feature_subset: If provided, only extract these features (else extract all trainable)

        Returns:
            Dict mapping feature names to float values

        Example:
            features = FeatureExtractor.extract_from_result(
                result=img,
                context={"semantic_score": 0.88},
                target_face_name="John"
            )
            # Returns: {"aesthetic_score": 4.5, "g_blur": 120.3, "semantic_score": 0.88, ...}
        """
        if context is None:
            context = {}

        # Add target_face_name to context so face extractors can access it
        if target_face_name:
            context["target_face_name"] = target_face_name

        features_to_extract = feature_subset if feature_subset else get_trainable_features()
        extracted = {}

        for feature_name in features_to_extract:
            feat_def = FEATURE_REGISTRY.get(feature_name)
            if not feat_def:
                logger.warning(f"Feature '{feature_name}' not in registry, skipping")
                continue

            try:
                value = FeatureExtractor._extract_single_feature(
                    feat_def, result, context
                )
                if value is None:
                    logger.warning(f"Failed to extract '{feature_name}. Applying default value {feat_def.default_value}")
                    value = feat_def.default_value
                extracted[feature_name] = value
            except Exception as e:
                logger.error(f"Failed to extract '{feature_name}': {e}. Applying default value {feat_def.default_value}")
                extracted[feature_name] = feat_def.default_value
        return extracted

    @staticmethod
    def _extract_single_feature(
            feat_def: FeatureDefinition,
            result: ImageAnalysisResult,
            context: Dict[str, Any],
    ) -> Any:
        """
        Extract a single feature value using the appropriate strategy.

        Strategies:
        1. sql_column: Direct attribute access (e.g., result.aesthetic_score)
        2. Callable extractor: Lambda function (e.g., lambda result, ctx: ...)
        3. "face_extractor": Special face feature handler
        """

        if feat_def.sql_column:
            value = getattr(result, feat_def.sql_column, None)
            if value is None:
                logger.warning(f"Failed to extract '{feat_def.sql_column}' from result. Applying default value {feat_def.default_value}")
                return feat_def.default_value
            return value

        if callable(feat_def.extractor):
            return feat_def.extractor(result, context)

        logger.error(f"Falling back to default value: {feat_def.default_value} for ")
        return feat_def.default_value

