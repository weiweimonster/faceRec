"""
Tests for src/features/registry.py

Tests cover:
- FeatureType enum
- FeatureDataType enum
- FeatureDefinition dataclass
- Helper functions (_extract_face_attr, _encode_time_period)
- Registry access functions
"""

import pytest
from unittest.mock import MagicMock

from src.features.registry import (
    FeatureType,
    FeatureDataType,
    FeatureDefinition,
    FEATURE_REGISTRY,
    get_feature,
    get_all_features,
    get_trainable_features,
    get_features_by_category,
    get_normalization_bounds,
    get_feature_names,
    _extract_face_attr,
    _encode_time_period,
    _extract_year,
    _extract_date,
)
from src.common.types import FaceData, ImageAnalysisResult


# ============================================
# FeatureType Enum Tests
# ============================================

class TestFeatureType:
    """Tests for the FeatureType enum"""

    def test_feature_type_values(self):
        """Test that all expected feature types exist with correct values"""
        assert FeatureType.GLOBAL.value == "global"
        assert FeatureType.FACE.value == "face"
        assert FeatureType.SEMANTIC.value == "semantic"
        assert FeatureType.META.value == "meta"
        assert FeatureType.COMPUTED.value == "computed"

    def test_feature_type_count(self):
        """Test that we have exactly 5 feature types"""
        assert len(FeatureType) == 5

    def test_feature_type_membership(self):
        """Test that values can be checked for membership"""
        assert FeatureType.GLOBAL in FeatureType
        assert FeatureType.FACE in FeatureType

    def test_feature_type_from_value(self):
        """Test creating FeatureType from string value"""
        assert FeatureType("global") == FeatureType.GLOBAL
        assert FeatureType("face") == FeatureType.FACE


# ============================================
# FeatureDataType Enum Tests
# ============================================

class TestFeatureDataType:
    """Tests for the FeatureDataType enum"""

    def test_feature_data_type_values(self):
        """Test that all expected data types exist with correct values"""
        assert FeatureDataType.FLOAT.value == "float"
        assert FeatureDataType.INT.value == "int"
        assert FeatureDataType.BOOL.value == "bool"

    def test_feature_data_type_count(self):
        """Test that we have exactly 3 data types"""
        assert len(FeatureDataType) == 3


# ============================================
# FeatureDefinition Dataclass Tests
# ============================================

class TestFeatureDefinition:
    """Tests for the FeatureDefinition dataclass"""

    def test_basic_feature_definition(self):
        """Test creating a basic feature definition"""
        feat = FeatureDefinition(
            name="test_feature",
            category=FeatureType.GLOBAL,
            dtype=FeatureDataType.FLOAT,
        )
        assert feat.name == "test_feature"
        assert feat.category == FeatureType.GLOBAL
        assert feat.dtype == FeatureDataType.FLOAT
        assert feat.sql_column is None
        assert feat.extractor is None
        assert feat.normalization is None
        assert feat.is_trainable is True
        assert feat.default_value == 0.0
        assert feat.description == ""

    def test_feature_definition_with_sql_column(self):
        """Test feature definition with SQL column extraction"""
        feat = FeatureDefinition(
            name="aesthetic_score",
            category=FeatureType.GLOBAL,
            dtype=FeatureDataType.FLOAT,
            sql_column="aesthetic_score",
            normalization={"min": 4.0, "max": 5.5},
            description="Test description",
        )
        assert feat.sql_column == "aesthetic_score"
        assert feat.normalization == {"min": 4.0, "max": 5.5}
        assert feat.description == "Test description"

    def test_feature_definition_with_extractor(self):
        """Test feature definition with callable extractor"""
        extractor = lambda result, ctx: ctx.get("test_value", 0.0)
        feat = FeatureDefinition(
            name="computed_feature",
            category=FeatureType.COMPUTED,
            dtype=FeatureDataType.FLOAT,
            extractor=extractor,
            is_trainable=False,
        )
        assert feat.extractor is not None
        assert callable(feat.extractor)
        assert feat.is_trainable is False

        # Test the extractor works
        result = feat.extractor(None, {"test_value": 42.0})
        assert result == 42.0

    def test_feature_definition_non_trainable(self):
        """Test feature definition marked as non-trainable"""
        feat = FeatureDefinition(
            name="mmr_rank",
            category=FeatureType.COMPUTED,
            dtype=FeatureDataType.INT,
            is_trainable=False,
            default_value=-1.0,
        )
        assert feat.is_trainable is False
        assert feat.default_value == -1.0


# ============================================
# Helper Function Tests
# ============================================

class TestExtractFaceAttr:
    """Tests for the _extract_face_attr helper function"""

    def test_extract_face_attr_no_faces(self):
        """Test extraction when result has no faces"""
        result = MagicMock()
        result.faces = None

        value = _extract_face_attr(result, None, lambda f: f.blur_score)
        assert value is None

    def test_extract_face_attr_empty_faces_list(self):
        """Test extraction when faces list is empty"""
        result = MagicMock()
        result.faces = []

        value = _extract_face_attr(result, None, lambda f: f.blur_score)
        assert value is None

    def test_extract_face_attr_first_face_fallback(self):
        """Test that first face is used when no target name specified"""
        face1 = FaceData(name="John", blur_score=100.0)
        face2 = FaceData(name="Jane", blur_score=200.0)

        result = MagicMock()
        result.faces = [face1, face2]

        value = _extract_face_attr(result, None, lambda f: f.blur_score)
        assert value == 100.0  # Should get first face

    def test_extract_face_attr_by_name(self):
        """Test extraction by target face name"""
        face1 = FaceData(name="John", blur_score=100.0)
        face2 = FaceData(name="Jane", blur_score=200.0)

        result = MagicMock()
        result.faces = [face1, face2]

        value = _extract_face_attr(result, "Jane", lambda f: f.blur_score)
        assert value == 200.0

    def test_extract_face_attr_case_insensitive(self):
        """Test that name matching is case-insensitive"""
        face = FaceData(name="John", blur_score=150.0)

        result = MagicMock()
        result.faces = [face]

        value = _extract_face_attr(result, "JOHN", lambda f: f.blur_score)
        assert value == 150.0

        value = _extract_face_attr(result, "john", lambda f: f.blur_score)
        assert value == 150.0

    def test_extract_face_attr_name_not_found_fallback(self):
        """Test fallback to first face when target name not found"""
        face1 = FaceData(name="John", blur_score=100.0)
        face2 = FaceData(name="Jane", blur_score=200.0)

        result = MagicMock()
        result.faces = [face1, face2]

        # Request "Alice" which doesn't exist - should fallback to first face
        value = _extract_face_attr(result, "Alice", lambda f: f.blur_score)
        assert value == 100.0

    def test_extract_face_attr_face_without_name(self):
        """Test extraction when face has no name"""
        face = FaceData(blur_score=100.0)  # No name

        result = MagicMock()
        result.faces = [face]

        value = _extract_face_attr(result, "John", lambda f: f.blur_score)
        assert value == 100.0  # Falls back to first face


class TestEncodeTimePeriod:
    """Tests for the _encode_time_period helper function"""

    def test_encode_time_period_morning(self):
        """Test encoding morning"""
        assert _encode_time_period("morning") == 1.0

    def test_encode_time_period_afternoon(self):
        """Test encoding afternoon"""
        assert _encode_time_period("afternoon") == 2.0

    def test_encode_time_period_evening(self):
        """Test encoding evening"""
        assert _encode_time_period("evening") == 3.0

    def test_encode_time_period_night(self):
        """Test encoding night"""
        assert _encode_time_period("night") == 4.0

    def test_encode_time_period_case_insensitive(self):
        """Test that encoding is case-insensitive"""
        assert _encode_time_period("MORNING") == 1.0
        assert _encode_time_period("Afternoon") == 2.0
        assert _encode_time_period("EVENING") == 3.0
        assert _encode_time_period("Night") == 4.0

    def test_encode_time_period_none(self):
        """Test encoding None returns 0.0"""
        assert _encode_time_period(None) == 0.0

    def test_encode_time_period_unknown(self):
        """Test encoding unknown value returns 0.0"""
        assert _encode_time_period("unknown") == 0.0
        assert _encode_time_period("midday") == 0.0


class TestExtractYear:
    """Tests for the _extract_year helper function"""

    def test_extract_year_standard_format(self):
        """Test year extraction from standard timestamp format YYYY-MM-DD HH:MM:SS"""
        assert _extract_year("2023-06-15 12:00:00") == 2023
        assert _extract_year("2021-12-31 23:59:59") == 2021

    def test_extract_year_date_only(self):
        """Test year extraction from date-only format"""
        assert _extract_year("2023-06-15") == 2023
        assert _extract_year("2021-12-31") == 2021

    def test_extract_year_none(self):
        """Test year extraction with None returns -1"""
        assert _extract_year(None) == -1

    def test_extract_year_empty_string(self):
        """Test year extraction with empty string returns -1"""
        assert _extract_year("") == -1

    def test_extract_year_short_string(self):
        """Test year extraction with string too short returns -1"""
        assert _extract_year("202") == -1

    def test_extract_year_invalid_format(self):
        """Test year extraction with invalid format returns -1"""
        assert _extract_year("abcd-06-15") == -1


class TestExtractDate:
    """Tests for the _extract_date helper function"""

    def test_extract_date_standard_format(self):
        """Test date extraction from standard timestamp format YYYY-MM-DD HH:MM:SS"""
        assert _extract_date("2023-06-15 12:00:00") == 15
        assert _extract_date("2021-12-31 23:59:59") == 31
        assert _extract_date("2023-01-01 00:00:00") == 1

    def test_extract_date_date_only(self):
        """Test date extraction from date-only format"""
        assert _extract_date("2023-06-15") == 15
        assert _extract_date("2021-12-31") == 31
        assert _extract_date("2023-01-01") == 1

    def test_extract_date_none(self):
        """Test date extraction with None returns -1"""
        assert _extract_date(None) == -1

    def test_extract_date_empty_string(self):
        """Test date extraction with empty string returns -1"""
        assert _extract_date("") == -1

    def test_extract_date_short_string(self):
        """Test date extraction with string too short returns -1"""
        assert _extract_date("2023-06") == -1

    def test_extract_date_single_digit_day(self):
        """Test date extraction with single digit day (zero-padded)"""
        assert _extract_date("2023-06-05 10:30:00") == 5
        assert _extract_date("2023-06-05") == 5


# ============================================
# Registry Access Function Tests
# ============================================

class TestRegistryAccessFunctions:
    """Tests for registry access helper functions"""

    def test_get_feature_existing(self):
        """Test getting an existing feature"""
        feat = get_feature("aesthetic_score")
        assert feat is not None
        assert feat.name == "aesthetic_score"
        assert feat.category == FeatureType.GLOBAL

    def test_get_feature_nonexistent(self):
        """Test getting a non-existent feature returns None"""
        feat = get_feature("nonexistent_feature")
        assert feat is None

    def test_get_all_features(self):
        """Test getting all features returns the registry"""
        all_features = get_all_features()
        assert isinstance(all_features, dict)
        assert len(all_features) > 0
        assert "aesthetic_score" in all_features
        assert "f_blur" in all_features

    def test_get_trainable_features(self):
        """Test getting trainable features excludes non-trainable ones"""
        trainable = get_trainable_features()
        assert isinstance(trainable, list)
        assert "aesthetic_score" in trainable
        assert "f_blur" in trainable
        # Non-trainable features should be excluded
        assert "mmr_rank" not in trainable
        assert "final_relevance" not in trainable

    def test_get_features_by_category_global(self):
        """Test getting features by GLOBAL category"""
        global_features = get_features_by_category(FeatureType.GLOBAL)
        assert isinstance(global_features, dict)
        assert "aesthetic_score" in global_features
        assert "g_blur" in global_features
        assert "g_brightness" in global_features
        # Face features should not be included
        assert "f_blur" not in global_features

    def test_get_features_by_category_face(self):
        """Test getting features by FACE category"""
        face_features = get_features_by_category(FeatureType.FACE)
        assert isinstance(face_features, dict)
        assert "f_blur" in face_features
        assert "f_conf" in face_features
        assert "f_yaw" in face_features
        # Global features should not be included
        assert "aesthetic_score" not in face_features

    def test_get_features_by_category_semantic(self):
        """Test getting features by SEMANTIC category"""
        semantic_features = get_features_by_category(FeatureType.SEMANTIC)
        assert isinstance(semantic_features, dict)
        assert "semantic_score" in semantic_features

    def test_get_features_by_category_computed(self):
        """Test getting features by COMPUTED category"""
        computed_features = get_features_by_category(FeatureType.COMPUTED)
        assert isinstance(computed_features, dict)
        assert "mmr_rank" in computed_features
        assert "final_relevance" in computed_features

    def test_get_features_by_category_meta(self):
        """Test getting features by META category"""
        meta_features = get_features_by_category(FeatureType.META)
        assert isinstance(meta_features, dict)
        assert "year" in meta_features
        assert "month" in meta_features
        assert "time_period" in meta_features

    def test_get_normalization_bounds_existing(self):
        """Test getting normalization bounds for a feature with bounds"""
        bounds = get_normalization_bounds("aesthetic_score")
        assert bounds is not None
        assert bounds == {"min": 4.0, "max": 5.5}

    def test_get_normalization_bounds_no_bounds(self):
        """Test getting normalization bounds for feature without bounds"""
        bounds = get_normalization_bounds("semantic_score")
        assert bounds is None

    def test_get_normalization_bounds_nonexistent(self):
        """Test getting normalization bounds for non-existent feature"""
        bounds = get_normalization_bounds("nonexistent_feature")
        assert bounds is None

    def test_get_feature_names(self):
        """Test getting all feature names"""
        names = get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert "aesthetic_score" in names
        assert "f_blur" in names
        assert "semantic_score" in names


# ============================================
# FEATURE_REGISTRY Tests
# ============================================

class TestFeatureRegistry:
    """Tests for the FEATURE_REGISTRY global dictionary"""

    def test_registry_is_dict(self):
        """Test that registry is a dictionary"""
        assert isinstance(FEATURE_REGISTRY, dict)

    def test_registry_not_empty(self):
        """Test that registry contains features"""
        assert len(FEATURE_REGISTRY) > 0

    def test_registry_values_are_feature_definitions(self):
        """Test that all values are FeatureDefinition instances"""
        for name, feat in FEATURE_REGISTRY.items():
            assert isinstance(feat, FeatureDefinition), f"{name} is not a FeatureDefinition"

    def test_registry_keys_match_feature_names(self):
        """Test that registry keys match feature definition names"""
        for key, feat in FEATURE_REGISTRY.items():
            assert key == feat.name, f"Key '{key}' doesn't match feature name '{feat.name}'"

    def test_all_features_have_category(self):
        """Test that all features have a valid category"""
        for name, feat in FEATURE_REGISTRY.items():
            assert isinstance(feat.category, FeatureType), f"{name} has invalid category"

    def test_all_features_have_dtype(self):
        """Test that all features have a valid data type"""
        for name, feat in FEATURE_REGISTRY.items():
            assert isinstance(feat.dtype, FeatureDataType), f"{name} has invalid dtype"

    def test_features_have_extraction_strategy(self):
        """Test that all features have either sql_column or extractor"""
        for name, feat in FEATURE_REGISTRY.items():
            has_sql = feat.sql_column is not None
            has_extractor = feat.extractor is not None
            assert has_sql or has_extractor, f"{name} has no extraction strategy"

    def test_expected_global_features_exist(self):
        """Test that expected global features are registered"""
        expected = ["aesthetic_score", "g_blur", "g_brightness", "g_contrast", "g_iso"]
        for name in expected:
            assert name in FEATURE_REGISTRY, f"Missing global feature: {name}"
            assert FEATURE_REGISTRY[name].category == FeatureType.GLOBAL

    def test_expected_face_features_exist(self):
        """Test that expected face features are registered"""
        expected = ["f_blur", "f_conf", "f_orient_score", "f_width", "f_height",
                    "f_yaw", "f_pitch", "f_roll"]
        for name in expected:
            assert name in FEATURE_REGISTRY, f"Missing face feature: {name}"
            assert FEATURE_REGISTRY[name].category == FeatureType.FACE

    def test_expected_semantic_features_exist(self):
        """Test that expected semantic features are registered"""
        expected = ["semantic_score"]
        for name in expected:
            assert name in FEATURE_REGISTRY, f"Missing semantic feature: {name}"
            assert FEATURE_REGISTRY[name].category == FeatureType.SEMANTIC

    def test_expected_computed_features_exist(self):
        """Test that expected computed features are registered"""
        expected = ["mmr_rank", "final_relevance"]
        for name in expected:
            assert name in FEATURE_REGISTRY, f"Missing computed feature: {name}"
            assert FEATURE_REGISTRY[name].category == FeatureType.COMPUTED
            assert FEATURE_REGISTRY[name].is_trainable is False

    def test_expected_meta_features_exist(self):
        """Test that expected metadata features are registered"""
        expected = ["year", "month", "date", "time_period", "has_face"]
        for name in expected:
            assert name in FEATURE_REGISTRY, f"Missing meta feature: {name}"
            assert FEATURE_REGISTRY[name].category == FeatureType.META