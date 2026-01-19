"""
Tests for src/rank/xgboost_ranker.py

Tests cover:
- XGBoostRanker initialization
- XGBoostRanker.score_candidates()
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.rank.xgboost_ranker import XGBoostRanker
from src.rank.base import RankingResult
from src.common.types import ImageAnalysisResult, FaceData
from src.pose.pose import Pose


# ============================================
# Test Fixtures
# ============================================

@pytest.fixture
def mock_image():
    """Create a mock ImageAnalysisResult"""
    result = ImageAnalysisResult(
        original_path="/path/img.jpg",
        photo_id="test1",
        display_path="/display/img.jpg"
    )
    result.aesthetic_score = 4.8
    result.global_blur = 500.0
    result.global_brightness = 128.0
    result.global_contrast = 55.0
    result.iso = 200
    result.timestamp = "20230615120000"
    result.month = 6
    result.time_period = "afternoon"
    result.semantic_vector = np.random.rand(512).astype(np.float32)
    result.faces = None
    return result


@pytest.fixture
def mock_images():
    """Create multiple mock ImageAnalysisResult objects"""
    images = []
    for i in range(3):
        result = ImageAnalysisResult(
            original_path=f"/path/img{i}.jpg",
            photo_id=f"test{i}",
            display_path=f"/display/img{i}.jpg"
        )
        result.aesthetic_score = 4.0 + (i * 0.3)
        result.global_blur = 400.0 + (i * 100)
        result.global_brightness = 120.0 + (i * 10)
        result.global_contrast = 50.0 + (i * 5)
        result.iso = 100 * (i + 1)
        result.timestamp = f"2023061{i}120000"
        result.month = 6
        result.time_period = "afternoon"
        result.semantic_vector = np.random.rand(512).astype(np.float32)
        result.faces = None
        images.append(result)
    return images


# ============================================
# XGBoostRanker Initialization Tests
# ============================================

class TestXGBoostRankerInit:
    """Tests for XGBoostRanker initialization"""

    def test_init_model_not_found(self):
        """Test initialization when model file doesn't exist"""
        with patch("src.rank.xgboost_ranker.os.path.exists", return_value=False):
            with patch("src.rank.xgboost_ranker.logger"):
                ranker = XGBoostRanker(model_path="nonexistent.json")
                assert ranker.model is None
                assert ranker.feature_cols == []

    def test_init_with_model(self):
        """Test initialization with existing model"""
        mock_booster = MagicMock()
        mock_booster.feature_names = ["semantic_score", "aesthetic_score", "f_blur"]

        with patch("src.rank.xgboost_ranker.os.path.exists", return_value=True):
            with patch("src.rank.xgboost_ranker.xgb.Booster", return_value=mock_booster):
                with patch("src.rank.xgboost_ranker.logger"):
                    ranker = XGBoostRanker(model_path="model.json")
                    assert ranker.model is not None
                    mock_booster.load_model.assert_called_once_with("model.json")

    def test_init_feature_cols_from_model(self):
        """Test that feature columns are pulled from the model file"""
        mock_booster = MagicMock()
        # Model uses a subset of features (source of truth)
        mock_booster.feature_names = ["semantic_score", "aesthetic_score", "f_blur"]

        with patch("src.rank.xgboost_ranker.os.path.exists", return_value=True):
            with patch("src.rank.xgboost_ranker.xgb.Booster", return_value=mock_booster):
                with patch("src.rank.xgboost_ranker.logger"):
                    ranker = XGBoostRanker(model_path="model.json")
                    # Feature cols should match what's in the model
                    assert ranker.feature_cols == ["semantic_score", "aesthetic_score", "f_blur"]
                    assert len(ranker.feature_cols) == 3


# ============================================
# XGBoostRanker.score_candidates Tests
# ============================================

class TestXGBoostRankerScoreCandidates:
    """Tests for XGBoostRanker.score_candidates method"""

    def test_score_candidates_no_model(self, mock_image):
        """Test scoring when model is not loaded"""
        with patch("src.rank.xgboost_ranker.os.path.exists", return_value=False):
            with patch("src.rank.xgboost_ranker.logger"):
                ranker = XGBoostRanker()

        semantic_scores = {mock_image.display_path: 0.85}
        caption_scores = {mock_image.display_path: 0.80}
        result = ranker.score_candidates([mock_image], semantic_scores, caption_scores)

        assert isinstance(result, RankingResult)
        assert len(result.ranked_results) == 1
        assert result.ranked_results[0] == mock_image

    def test_score_candidates_empty_results(self):
        """Test scoring with empty results list"""
        with patch("src.rank.xgboost_ranker.os.path.exists", return_value=False):
            with patch("src.rank.xgboost_ranker.logger"):
                ranker = XGBoostRanker()

        result = ranker.score_candidates([], {}, {})

        assert isinstance(result, RankingResult)
        assert result.ranked_results == []

    def test_score_candidates_with_model(self, mock_images):
        """Test scoring with a loaded model"""
        mock_booster = MagicMock()
        mock_booster.feature_names = ["semantic_score", "aesthetic_score"]
        # Simulate XGBoost predictions (array of scores)
        mock_booster.predict.return_value = np.array([0.8, 0.6, 0.9])

        with patch("src.rank.xgboost_ranker.os.path.exists", return_value=True):
            with patch("src.rank.xgboost_ranker.xgb.Booster", return_value=mock_booster):
                with patch("src.rank.xgboost_ranker.logger"):
                    ranker = XGBoostRanker(model_path="model.json")

        semantic_scores = {
            "/display/img0.jpg": 0.85,
            "/display/img1.jpg": 0.80,
            "/display/img2.jpg": 0.75
        }
        caption_scores = {
            "/display/img0.jpg": 0.80,
            "/display/img1.jpg": 0.75,
            "/display/img2.jpg": 0.70
        }

        result = ranker.score_candidates(mock_images, semantic_scores, caption_scores)

        assert isinstance(result, RankingResult)
        assert len(result.ranked_results) == 3

        # Should be sorted by XGBoost score (highest first)
        # img2 has score 0.9, img0 has 0.8, img1 has 0.6
        assert result.ranked_results[0].display_path == "/display/img2.jpg"
        assert result.ranked_results[1].display_path == "/display/img0.jpg"
        assert result.ranked_results[2].display_path == "/display/img1.jpg"

    def test_score_candidates_display_metrics(self, mock_images):
        """Test that display_metrics contain xgboost_score"""
        mock_booster = MagicMock()
        mock_booster.feature_names = ["semantic_score", "aesthetic_score"]
        mock_booster.predict.return_value = np.array([0.8, 0.6, 0.9])

        with patch("src.rank.xgboost_ranker.os.path.exists", return_value=True):
            with patch("src.rank.xgboost_ranker.xgb.Booster", return_value=mock_booster):
                with patch("src.rank.xgboost_ranker.logger"):
                    ranker = XGBoostRanker(model_path="model.json")

        semantic_scores = {img.display_path: 0.8 for img in mock_images}
        caption_scores = {img.display_path: 0.75 for img in mock_images}
        result = ranker.score_candidates(mock_images, semantic_scores, caption_scores)

        for path in semantic_scores:
            assert path in result.display_metrics
            assert "xgboost_score" in result.display_metrics[path]

    def test_score_candidates_training_features(self, mock_images):
        """Test that training_features are populated with model's features"""
        mock_booster = MagicMock()
        mock_booster.feature_names = ["semantic_score", "aesthetic_score"]
        mock_booster.predict.return_value = np.array([0.8, 0.6, 0.9])

        with patch("src.rank.xgboost_ranker.os.path.exists", return_value=True):
            with patch("src.rank.xgboost_ranker.xgb.Booster", return_value=mock_booster):
                with patch("src.rank.xgboost_ranker.logger"):
                    ranker = XGBoostRanker(model_path="model.json")

        semantic_scores = {img.display_path: 0.8 for img in mock_images}
        caption_scores = {img.display_path: 0.75 for img in mock_images}
        result = ranker.score_candidates(mock_images, semantic_scores, caption_scores)

        for path in semantic_scores:
            assert path in result.training_features
            # Should only have features from the model
            assert "semantic_score" in result.training_features[path]
            assert "aesthetic_score" in result.training_features[path]

    def test_score_candidates_with_target_name(self, mock_images):
        """Test scoring with target_name parameter"""
        # Add face to first image
        face = FaceData(
            name="John",
            bbox=[100, 100, 200, 200],
            blur_score=300.0,
            confidence=0.9,
            yaw=0.0,
            pitch=0.0,
            roll=0.0
        )
        mock_images[0].faces = [face]

        mock_booster = MagicMock()
        mock_booster.feature_names = ["semantic_score", "f_blur"]
        mock_booster.predict.return_value = np.array([0.8, 0.6, 0.9])

        with patch("src.rank.xgboost_ranker.os.path.exists", return_value=True):
            with patch("src.rank.xgboost_ranker.xgb.Booster", return_value=mock_booster):
                with patch("src.rank.xgboost_ranker.logger"):
                    ranker = XGBoostRanker(model_path="model.json")

        semantic_scores = {img.display_path: 0.8 for img in mock_images}
        caption_scores = {img.display_path: 0.75 for img in mock_images}
        result = ranker.score_candidates(
            mock_images,
            semantic_scores,
            caption_scores,
            target_name="John"
        )

        assert isinstance(result, RankingResult)
        # First image should have face features extracted
        features = result.training_features["/display/img0.jpg"]
        assert "f_blur" in features

    def test_score_candidates_uses_model_features_only(self, mock_image):
        """Test that only features from the model are extracted"""
        mock_booster = MagicMock()
        # Model only uses these 2 features
        mock_booster.feature_names = ["semantic_score", "aesthetic_score"]
        mock_booster.predict.return_value = np.array([0.8])

        with patch("src.rank.xgboost_ranker.os.path.exists", return_value=True):
            with patch("src.rank.xgboost_ranker.xgb.Booster", return_value=mock_booster):
                with patch("src.rank.xgboost_ranker.logger"):
                    ranker = XGBoostRanker(model_path="model.json")

        semantic_scores = {mock_image.display_path: 0.85}
        caption_scores = {mock_image.display_path: 0.80}
        result = ranker.score_candidates([mock_image], semantic_scores, caption_scores)

        # Verify only model's features are extracted
        features = result.training_features[mock_image.display_path]

        # Only features from the model should be present
        assert "aesthetic_score" in features
        assert "semantic_score" in features
        assert len(features) == 2

        # Other features should NOT be present
        assert "g_blur" not in features
        assert "f_blur" not in features
        assert "mmr_rank" not in features
