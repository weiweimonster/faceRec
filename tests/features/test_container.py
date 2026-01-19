"""
Tests for src/features/container.py

Tests cover:
- FeatureExtractor.extract_from_result()
- FeatureExtractor._extract_single_feature()
"""

import pytest
from unittest.mock import MagicMock, patch

from src.features.container import FeatureExtractor
from src.features.registry import (
    FeatureDefinition,
    FeatureType,
    FeatureDataType,
    FEATURE_REGISTRY,
)
from src.features import FEATURE_REGISTRY
from src.common.types import FaceData, ImageAnalysisResult


# ============================================
# Test Fixtures
# ============================================

@pytest.fixture
def mock_result_basic():
    """Create a basic ImageAnalysisResult mock with global features only"""
    result = ImageAnalysisResult(
        original_path="/test/image.jpg",
        photo_id="test123",
        display_path="/test/display.jpg",
    )
    result.aesthetic_score = 4.8
    result.global_blur = 500.0
    result.global_brightness = 128.0
    result.global_contrast = 55.0
    result.iso = 400
    result.timestamp = "2023-06-15 12:00:00"
    result.month = 6
    result.time_period = "afternoon"
    result.faces = None
    return result


@pytest.fixture
def mock_result_with_faces():
    """Create an ImageAnalysisResult mock with face data"""
    face1 = FaceData(
        name="John",
        bbox=[100, 100, 200, 250],
        blur_score=300.0,
        confidence=0.95,
        yaw=10.0,
        pitch=5.0,
        roll=-2.0,
    )
    face2 = FaceData(
        name="Jane",
        bbox=[300, 100, 400, 230],
        blur_score=450.0,
        confidence=0.88,
        yaw=-15.0,
        pitch=8.0,
        roll=3.0,
    )

    result = ImageAnalysisResult(
        original_path="/test/image.jpg",
        photo_id="test123",
        display_path="/test/display.jpg",
    )
    result.aesthetic_score = 4.5
    result.global_blur = 600.0
    result.global_brightness = 140.0
    result.global_contrast = 60.0
    result.iso = 200
    result.timestamp = "2023-12-25 18:00:00"
    result.month = 12
    result.time_period = "evening"
    result.faces = [face1, face2]
    return result


@pytest.fixture
def mock_result_minimal():
    """Create a minimal ImageAnalysisResult with missing attributes"""
    result = ImageAnalysisResult(
        original_path="/test/image.jpg",
        photo_id="test123",
        display_path="/test/display.jpg",
    )
    result.faces = None
    return result


# ============================================
# FeatureExtractor.extract_from_result Tests
# ============================================

class TestFeatureExtractorExtractFromResult:
    """Tests for FeatureExtractor.extract_from_result method"""

    def test_extract_global_features(self, mock_result_basic):
        """Test extracting global quality features"""
        features = FeatureExtractor.extract_from_result(
            result=mock_result_basic,
            feature_subset=["aesthetic_score", "g_blur", "g_brightness", "g_contrast"]
        )

        assert features["aesthetic_score"] == 4.8
        assert features["g_blur"] == 500.0
        assert features["g_brightness"] == 128.0
        assert features["g_contrast"] == 55.0

    def test_extract_with_context(self, mock_result_basic):
        """Test extracting features with context values"""
        context = {
            "semantic_score": 0.85,
            "mmr_rank": 3,
            "final_relevance": 0.72,
        }

        features = FeatureExtractor.extract_from_result(
            result=mock_result_basic,
            context=context,
            feature_subset=["semantic_score", "mmr_rank", "final_relevance"]
        )

        assert features["semantic_score"] == 0.85
        assert features["mmr_rank"] == 3
        assert features["final_relevance"] == 0.72

    def test_extract_face_features_with_target(self, mock_result_with_faces):
        """Test extracting face features for a specific person"""
        features = FeatureExtractor.extract_from_result(
            result=mock_result_with_faces,
            target_face_name="Jane",
            feature_subset=["f_blur", "f_conf", "f_yaw"]
        )

        assert features["f_blur"] == 450.0
        assert features["f_conf"] == 0.88
        assert features["f_yaw"] == -15.0

    def test_extract_face_features_first_face_fallback(self, mock_result_with_faces):
        """Test that first face is used when no target specified"""
        features = FeatureExtractor.extract_from_result(
            result=mock_result_with_faces,
            feature_subset=["f_blur", "f_conf"]
        )

        # Should get first face (John)
        assert features["f_blur"] == 300.0
        assert features["f_conf"] == 0.95

    def test_extract_face_dimensions(self, mock_result_with_faces):
        """Test extracting face width and height"""
        features = FeatureExtractor.extract_from_result(
            result=mock_result_with_faces,
            target_face_name="John",
            feature_subset=["f_width", "f_height"]
        )

        # John's bbox is [100, 100, 200, 250]
        # width = 200 - 100 = 100, height = 250 - 100 = 150
        assert features["f_width"] == 100
        assert features["f_height"] == 150

    def test_extract_meta_features(self, mock_result_basic):
        """Test extracting metadata features"""
        features = FeatureExtractor.extract_from_result(
            result=mock_result_basic,
            feature_subset=["year", "month", "date", "time_period"]
        )

        # timestamp is "2023-06-15 12:00:00"
        assert features["year"] == 2023
        assert features["month"] == 6
        assert features["date"] == 15
        assert features["time_period"] == 2.0  # afternoon

    def test_extract_has_face_true(self, mock_result_with_faces):
        """Test has_face feature when faces exist"""
        features = FeatureExtractor.extract_from_result(
            result=mock_result_with_faces,
            feature_subset=["has_face"]
        )
        assert features["has_face"] == 1.0

    def test_extract_has_face_false(self, mock_result_basic):
        """Test has_face feature when no faces"""
        features = FeatureExtractor.extract_from_result(
            result=mock_result_basic,
            feature_subset=["has_face"]
        )
        assert features["has_face"] == 0.0

    def test_extract_all_trainable_features(self, mock_result_with_faces):
        """Test extracting all trainable features (default behavior)"""
        features = FeatureExtractor.extract_from_result(
            result=mock_result_with_faces,
            context={"semantic_score": 0.9},
        )

        # Should have extracted all trainable features
        assert "aesthetic_score" in features
        assert "g_blur" in features
        assert "f_blur" in features
        assert "semantic_score" in features
        assert "year" in features

        # Non-trainable features should NOT be present (only trainable by default)
        assert "mmr_rank" not in features
        assert "final_relevance" not in features

    def test_extract_with_none_context(self, mock_result_basic):
        """Test extraction with None context defaults to empty dict"""
        features = FeatureExtractor.extract_from_result(
            result=mock_result_basic,
            context=None,
            feature_subset=["semantic_score"]
        )

        # Should use default value since context is empty
        assert features["semantic_score"] == 0.0

    def test_extract_unknown_feature_skipped(self, mock_result_basic):
        """Test that unknown features are skipped with warning"""
        with patch("src.features.container.logger") as mock_logger:
            features = FeatureExtractor.extract_from_result(
                result=mock_result_basic,
                feature_subset=["aesthetic_score", "nonexistent_feature"]
            )

            # Should have extracted the valid feature
            assert "aesthetic_score" in features
            # Unknown feature should not be present
            assert "nonexistent_feature" not in features
            # Should have logged a warning
            mock_logger.warning.assert_called()

    def test_extract_missing_attribute_uses_default(self, mock_result_minimal):
        """Test that missing attributes use default values"""
        features = FeatureExtractor.extract_from_result(
            result=mock_result_minimal,
            feature_subset=["aesthetic_score", "g_iso"]
        )

        # aesthetic_score has default 0.0
        assert features["aesthetic_score"] == 0.0
        # g_iso has default -1.0
        assert features["g_iso"] == -1.0

    def test_extract_face_features_no_faces_uses_default(self, mock_result_basic):
        """Test face feature extraction with no faces uses defaults"""
        features = FeatureExtractor.extract_from_result(
            result=mock_result_basic,
            feature_subset=["f_blur", "f_conf"]
        )

        # f_blur has default -1.0
        assert features["f_blur"] == -1.0
        # f_conf has default 0.0
        assert features["f_conf"] == 0.0


# ============================================
# FeatureExtractor._extract_single_feature Tests
# ============================================

class TestFeatureExtractorExtractSingleFeature:
    """Tests for FeatureExtractor._extract_single_feature method"""

    def test_extract_via_sql_column(self, mock_result_basic):
        """Test extraction using sql_column strategy"""
        feat_def = FeatureDefinition(
            name="test",
            category=FeatureType.GLOBAL,
            dtype=FeatureDataType.FLOAT,
            sql_column="aesthetic_score",
        )

        value = FeatureExtractor._extract_single_feature(
            feat_def, mock_result_basic, {}
        )
        assert value == 4.8

    def test_extract_via_sql_column_missing_attribute(self, mock_result_minimal):
        """Test sql_column extraction when attribute is missing"""
        feat_def = FeatureDefinition(
            name="test",
            category=FeatureType.GLOBAL,
            dtype=FeatureDataType.FLOAT,
            sql_column="aesthetic_score",
            default_value=-999.0,
        )

        value = FeatureExtractor._extract_single_feature(
            feat_def, mock_result_minimal, {}
        )
        assert value == -999.0

    def test_extract_via_callable_extractor(self, mock_result_basic):
        """Test extraction using callable extractor"""
        feat_def = FeatureDefinition(
            name="test",
            category=FeatureType.COMPUTED,
            dtype=FeatureDataType.FLOAT,
            extractor=lambda result, ctx: ctx.get("test_value", 0.0),
        )

        value = FeatureExtractor._extract_single_feature(
            feat_def, mock_result_basic, {"test_value": 42.5}
        )
        assert value == 42.5

    def test_extract_via_extractor_accessing_result(self, mock_result_basic):
        """Test extractor that accesses result object"""
        feat_def = FeatureDefinition(
            name="test",
            category=FeatureType.COMPUTED,
            dtype=FeatureDataType.FLOAT,
            extractor=lambda result, ctx: result.aesthetic_score * 2,
        )

        value = FeatureExtractor._extract_single_feature(
            feat_def, mock_result_basic, {}
        )
        assert value == 9.6  # 4.8 * 2

    def test_extract_fallback_to_default(self):
        """Test fallback to default when no extraction strategy works"""
        feat_def = FeatureDefinition(
            name="test",
            category=FeatureType.GLOBAL,
            dtype=FeatureDataType.FLOAT,
            default_value=123.0,
            # No sql_column or extractor
        )

        result = MagicMock()
        value = FeatureExtractor._extract_single_feature(
            feat_def, result, {}
        )
        assert value == 123.0


# ============================================
# Integration Tests
# ============================================

class TestFeatureExtractorIntegration:
    """Integration tests for FeatureExtractor with real registry features"""

    def test_extract_real_aesthetic_score(self, mock_result_basic):
        """Test extracting aesthetic_score using actual registry definition"""
        feat_def = FEATURE_REGISTRY["aesthetic_score"]
        value = FeatureExtractor._extract_single_feature(
            feat_def, mock_result_basic, {}
        )
        assert value == 4.8

    def test_extract_real_semantic_score(self, mock_result_basic):
        """Test extracting semantic_score using actual registry definition"""
        feat_def = FEATURE_REGISTRY["semantic_score"]
        value = FeatureExtractor._extract_single_feature(
            feat_def, mock_result_basic, {"semantic_score": 0.92}
        )
        assert value == 0.92

    def test_extract_real_year_feature(self, mock_result_basic):
        """Test extracting year using actual registry definition"""
        feat_def = FEATURE_REGISTRY["year"]
        value = FeatureExtractor._extract_single_feature(
            feat_def, mock_result_basic, {}
        )
        assert value == 2023

    def test_extract_real_face_blur(self, mock_result_with_faces):
        """Test extracting f_blur using actual registry definition"""
        feat_def = FEATURE_REGISTRY["f_blur"]
        # Add target face name to context
        value = FeatureExtractor._extract_single_feature(
            feat_def, mock_result_with_faces, {"target_face_name": "Jane"}
        )
        assert value == 450.0

    def test_extract_real_time_period(self, mock_result_basic):
        """Test extracting time_period using actual registry definition"""
        feat_def = FEATURE_REGISTRY["time_period"]
        value = FeatureExtractor._extract_single_feature(
            feat_def, mock_result_basic, {}
        )
        assert value == 2.0  # afternoon

    def test_full_extraction_workflow(self, mock_result_with_faces):
        """Test a complete feature extraction workflow"""
        context = {
            "semantic_score": 0.88,
        }

        features = FeatureExtractor.extract_from_result(
            result=mock_result_with_faces,
            context=context,
            target_face_name="John",
        )

        # Verify global features
        assert features["aesthetic_score"] == 4.5
        assert features["g_blur"] == 600.0

        # Verify semantic features from context
        assert features["semantic_score"] == 0.88

        # Verify face features for John
        assert features["f_blur"] == 300.0
        assert features["f_conf"] == 0.95
        assert features["f_yaw"] == 10.0

        # Verify meta features
        assert features["year"] == 2023
        assert features["month"] == 12
        assert features["has_face"] == 1.0
        assert features["time_period"] == 3.0  # evening

        # Non-trainable features should NOT be present
        assert "mmr_rank" not in features
        assert "final_relevance" not in features


# ============================================
# Edge Case Tests
# ============================================

class TestFeatureExtractorEdgeCases:
    """Edge case tests for FeatureExtractor"""

    def test_empty_feature_subset_extracts_trainable(self, mock_result_basic):
        """Test that empty feature subset falls back to extracting trainable features.

        Note: Empty list is falsy in Python, so the code treats it as
        'no subset specified' and extracts only trainable features.
        """
        from src.features.registry import get_trainable_features

        features = FeatureExtractor.extract_from_result(
            result=mock_result_basic,
            feature_subset=[]
        )
        # Empty list is falsy, so falls back to extracting trainable features only
        trainable = get_trainable_features()
        assert len(features) == len(trainable)
        assert "aesthetic_score" in features
        # Non-trainable features should NOT be present
        assert "mmr_rank" not in features
        assert "final_relevance" not in features

    def test_face_with_missing_bbox(self, mock_result_basic):
        """Test face dimension extraction when bbox is missing"""
        face = FaceData(name="Test", blur_score=100.0)  # No bbox
        mock_result_basic.faces = [face]

        features = FeatureExtractor.extract_from_result(
            result=mock_result_basic,
            feature_subset=["f_width", "f_height"]
        )

        assert features["f_width"] == 0.0
        assert features["f_height"] == 0.0

    def test_face_with_incomplete_bbox(self, mock_result_basic):
        """Test face dimension extraction when bbox is incomplete"""
        face = FaceData(name="Test", bbox=[100, 100])  # Incomplete bbox
        mock_result_basic.faces = [face]

        features = FeatureExtractor.extract_from_result(
            result=mock_result_basic,
            feature_subset=["f_width", "f_height"]
        )

        assert features["f_width"] == 0.0
        assert features["f_height"] == 0.0

    def test_timestamp_too_short(self, mock_result_minimal):
        """Test year/date extraction when timestamp is too short"""
        mock_result_minimal.timestamp = "2023"  # Only year

        features = FeatureExtractor.extract_from_result(
            result=mock_result_minimal,
            feature_subset=["year", "date"]
        )

        assert features["year"] == 2023
        assert features["date"] == -1  # Not enough chars for date

    def test_null_timestamp(self, mock_result_minimal):
        """Test year extraction when timestamp is None"""
        mock_result_minimal.timestamp = None

        features = FeatureExtractor.extract_from_result(
            result=mock_result_minimal,
            feature_subset=["year", "date"]
        )

        assert features["year"] == -1
        assert features["date"] == -1

    def test_exception_in_extractor_uses_default(self, mock_result_basic):
        """Test that exceptions in extractors fall back to default"""
        # Create a feature with an extractor that raises an exception
        with patch.dict(FEATURE_REGISTRY, {
            "test_broken": FeatureDefinition(
                name="test_broken",
                category=FeatureType.COMPUTED,
                dtype=FeatureDataType.FLOAT,
                extractor=lambda result, ctx: 1 / 0,  # Will raise ZeroDivisionError
                default_value=-999.0,
            )
        }):
            with patch("src.features.container.logger"):
                features = FeatureExtractor.extract_from_result(
                    result=mock_result_basic,
                    feature_subset=["test_broken"]
                )
                assert features["test_broken"] == -999.0

    def test_face_pose_values(self, mock_result_with_faces):
        """Test extraction of face pose values (yaw, pitch, roll)"""
        features = FeatureExtractor.extract_from_result(
            result=mock_result_with_faces,
            target_face_name="John",
            feature_subset=["f_yaw", "f_pitch", "f_roll"]
        )

        assert features["f_yaw"] == 10.0
        assert features["f_pitch"] == 5.0
        assert features["f_roll"] == -2.0

    def test_iso_feature(self, mock_result_basic):
        """Test ISO feature extraction"""
        features = FeatureExtractor.extract_from_result(
            result=mock_result_basic,
            feature_subset=["g_iso"]
        )
        assert features["g_iso"] == 400