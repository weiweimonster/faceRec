"""
Feature management system for photo search ranking.

This module provides a centralized registry for all features used in:
- Heuristic ranking
- XGBoost training
- Database logging
- UI display

Key components:
- FEATURE_REGISTRY: Global registry of all features
- FeatureExtractor: Unified extraction interface
- Helper functions: get_trainable_features(), get_normalization_bounds(), etc.
"""

from .registry import (
    FEATURE_REGISTRY,
    FeatureDefinition,
    FeatureType,
    FeatureDataType,
    get_feature,
    get_all_features,
    get_trainable_features,
    get_features_by_category,
    get_normalization_bounds,
    get_feature_names
)

from .container import FeatureExtractor

__all__ = [
    # Core classes
    "FEATURE_REGISTRY",
    "FeatureDefinition",
    "FeatureType",
    "FeatureDataType",
    "FeatureExtractor",

    # Helper functions
    "get_feature",
    "get_all_features",
    "get_trainable_features",
    "get_features_by_category",
    "get_normalization_bounds",
    "get_feature_names"
]
