"""
Tests for src/rank/heuristic_ranker.py

Tests cover:
- HeuristicStrategy.score_candidates()
- HeuristicStrategy._normalize()
- HeuristicStrategy._calculate_iso_score()
- HeuristicStrategy._calculate_global_quality()
- HeuristicStrategy._calculate_face_quality()
- HeuristicStrategy._calculate_orientation_score()
"""

import pytest
import math
from unittest.mock import patch, MagicMock
import numpy as np

from src.rank.heuristic_ranker import HeuristicStrategy
from src.rank.base import RankingResult
from src.common.types import ImageAnalysisResult, FaceData
from src.pose.pose import Pose


# ============================================
# Test Fixtures
# ============================================

@pytest.fixture
def strategy():
    """Create a HeuristicStrategy instance"""
    return HeuristicStrategy()


@pytest.fixture
def mock_image_basic():
    """Create a basic ImageAnalysisResult without faces"""
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
def mock_image_with_face():
    """Create an ImageAnalysisResult with face data"""
    face = FaceData(
        name="John",
        bbox=[100, 100, 300, 350],  # width=200, height=250
        blur_score=400.0,
        confidence=0.95,
        yaw=5.0,
        pitch=3.0,
        roll=0.0,
    )

    result = ImageAnalysisResult(
        original_path="/path/img_face.jpg",
        photo_id="test2",
        display_path="/display/img_face.jpg"
    )
    result.aesthetic_score = 4.5
    result.global_blur = 600.0
    result.global_brightness = 140.0
    result.global_contrast = 60.0
    result.iso = 400
    result.timestamp = "20231225180000"
    result.month = 12
    result.time_period = "evening"
    result.semantic_vector = np.random.rand(512).astype(np.float32)
    result.faces = [face]
    return result


# ============================================
# HeuristicStrategy._normalize Tests
# ============================================

class TestNormalize:
    """Tests for the _normalize method"""

    def test_normalize_within_bounds(self, strategy):
        """Test normalization for value within bounds"""
        # aesthetic_score has bounds {"min": 4.0, "max": 5.5}
        result = strategy._normalize("aesthetic_score", 4.75)
        assert 0.0 <= result <= 1.0
        assert result == pytest.approx(0.5, abs=0.01)

    def test_normalize_at_min(self, strategy):
        """Test normalization at minimum bound"""
        result = strategy._normalize("aesthetic_score", 4.0)
        assert result == 0.0

    def test_normalize_at_max(self, strategy):
        """Test normalization at maximum bound"""
        result = strategy._normalize("aesthetic_score", 5.5)
        assert result == 1.0

    def test_normalize_below_min(self, strategy):
        """Test normalization below minimum returns 0"""
        result = strategy._normalize("aesthetic_score", 3.0)
        assert result == 0.0

    def test_normalize_above_max(self, strategy):
        """Test normalization above maximum returns 1"""
        result = strategy._normalize("aesthetic_score", 6.0)
        assert result == 1.0

    def test_normalize_none_value(self, strategy):
        """Test normalization with None value returns 0"""
        result = strategy._normalize("aesthetic_score", None)
        assert result == 0.0

    def test_normalize_unknown_feature(self, strategy):
        """Test normalization for unknown feature returns 0"""
        with patch("src.rank.heuristic_ranker.logger"):
            result = strategy._normalize("unknown_feature", 5.0)
            assert result == 0.0


# ============================================
# HeuristicStrategy._calculate_iso_score Tests
# ============================================

class TestCalculateIsoScore:
    """Tests for the _calculate_iso_score method"""

    def test_iso_score_low_iso(self, strategy):
        """Test ISO 100 or below returns 1.0"""
        assert strategy._calculate_iso_score(100) == 1.0
        assert strategy._calculate_iso_score(50) == 1.0

    def test_iso_score_none(self, strategy):
        """Test None ISO returns neutral 0.5"""
        assert strategy._calculate_iso_score(None) == 0.5

    def test_iso_score_high_iso(self, strategy):
        """Test high ISO returns lower score"""
        score_200 = strategy._calculate_iso_score(200)
        score_800 = strategy._calculate_iso_score(800)
        score_1600 = strategy._calculate_iso_score(1600)

        # Higher ISO should give lower scores
        assert score_200 > score_800 > score_1600

    def test_iso_score_range(self, strategy):
        """Test ISO scores are in valid range"""
        for iso in [100, 200, 400, 800, 1600, 3200]:
            score = strategy._calculate_iso_score(iso)
            assert 0.0 <= score <= 1.0


# ============================================
# HeuristicStrategy._calculate_global_quality Tests
# ============================================

class TestCalculateGlobalQuality:
    """Tests for the _calculate_global_quality method"""

    def test_global_quality_returns_tuple(self, strategy, mock_image_basic):
        """Test that method returns score and metrics dict"""
        score, metrics = strategy._calculate_global_quality(mock_image_basic)

        assert isinstance(score, float)
        assert isinstance(metrics, dict)

    def test_global_quality_score_range(self, strategy, mock_image_basic):
        """Test that global quality score is in valid range"""
        score, _ = strategy._calculate_global_quality(mock_image_basic)
        assert 0.0 <= score <= 1.0

    def test_global_quality_metrics_keys(self, strategy, mock_image_basic):
        """Test that metrics dict has expected keys"""
        _, metrics = strategy._calculate_global_quality(mock_image_basic)

        expected_keys = ["g_aesthetic", "g_sharpness", "g_iso", "g_contrast", "global_score"]
        for key in expected_keys:
            assert key in metrics

    def test_global_quality_metrics_values_range(self, strategy, mock_image_basic):
        """Test that individual metric values are in valid range"""
        _, metrics = strategy._calculate_global_quality(mock_image_basic)

        for key in ["g_aesthetic", "g_sharpness", "g_iso", "g_contrast"]:
            assert 0.0 <= metrics[key] <= 1.0

    def test_global_quality_high_quality_image(self, strategy):
        """Test global quality for high quality image"""
        img = ImageAnalysisResult(
            original_path="/path/hq.jpg",
            photo_id="hq",
            display_path="/display/hq.jpg"
        )
        img.aesthetic_score = 5.5  # Max
        img.global_blur = 1400.0  # Max sharpness
        img.iso = 100  # Best ISO
        img.global_contrast = 80.0  # Max contrast

        score, metrics = strategy._calculate_global_quality(img)

        assert score > 0.8  # Should be high
        assert metrics["g_aesthetic"] == 1.0
        assert metrics["g_sharpness"] == 1.0
        assert metrics["g_iso"] == 1.0


# ============================================
# HeuristicStrategy._calculate_face_quality Tests
# ============================================

class TestCalculateFaceQuality:
    """Tests for the _calculate_face_quality method"""

    def test_face_quality_no_faces(self, strategy, mock_image_basic):
        """Test face quality with no faces returns 0"""
        score, metrics = strategy._calculate_face_quality(mock_image_basic, "John")

        assert score == 0.0
        assert metrics == {}

    def test_face_quality_target_not_found(self, strategy, mock_image_with_face):
        """Test face quality when target name not found"""
        score, metrics = strategy._calculate_face_quality(mock_image_with_face, "Jane")

        assert score == 0.0
        assert metrics == {}

    def test_face_quality_target_found(self, strategy, mock_image_with_face):
        """Test face quality when target is found"""
        score, metrics = strategy._calculate_face_quality(mock_image_with_face, "John")

        assert score > 0.0
        assert isinstance(metrics, dict)
        assert "f_blur" in metrics
        assert "f_size" in metrics
        assert "f_orient" in metrics
        assert "f_score" in metrics

    def test_face_quality_case_insensitive(self, strategy, mock_image_with_face):
        """Test face quality name matching is case-insensitive"""
        score1, _ = strategy._calculate_face_quality(mock_image_with_face, "John")
        score2, _ = strategy._calculate_face_quality(mock_image_with_face, "JOHN")
        score3, _ = strategy._calculate_face_quality(mock_image_with_face, "john")

        assert score1 == score2 == score3

    def test_face_quality_includes_confidence(self, strategy, mock_image_with_face):
        """Test that face quality is scaled by confidence"""
        # The face has confidence 0.95, so final score should be multiplied by it
        score, metrics = strategy._calculate_face_quality(mock_image_with_face, "John")

        # Raw f_score before confidence scaling
        raw_score = metrics["f_score"]
        # Final score should be raw_score * confidence (0.95)
        expected = raw_score * 0.95
        assert score == pytest.approx(expected, abs=0.01)

    def test_face_quality_with_pose(self, strategy, mock_image_with_face):
        """Test face quality with specific pose request"""
        # John has yaw=5, pitch=3 - close to frontal
        score_front, _ = strategy._calculate_face_quality(
            mock_image_with_face, "John", Pose.FRONT
        )

        # Should have high orientation score for frontal pose
        assert score_front > 0.0


# ============================================
# HeuristicStrategy._calculate_orientation_score Tests
# ============================================

class TestCalculateOrientationScore:
    """Tests for the _calculate_orientation_score method"""

    def test_orientation_score_frontal(self, strategy):
        """Test perfect frontal face gets high score"""
        face = FaceData(yaw=0.0, pitch=0.0)
        score = strategy._calculate_orientation_score(face, Pose.FRONT)
        assert score == 1.0

    def test_orientation_score_slight_deviation(self, strategy):
        """Test slight deviation from frontal"""
        face = FaceData(yaw=10.0, pitch=5.0)
        score = strategy._calculate_orientation_score(face, Pose.FRONT)

        # Should still be high but less than 1.0
        assert 0.7 < score < 1.0

    def test_orientation_score_large_deviation(self, strategy):
        """Test large deviation gives low score"""
        face = FaceData(yaw=50.0, pitch=40.0)
        score = strategy._calculate_orientation_score(face, Pose.FRONT)

        # Should be very low
        assert score < 0.3

    def test_orientation_score_side_pose(self, strategy):
        """Test side pose request"""
        # Face looking left (yaw ~45)
        face = FaceData(yaw=45.0, pitch=0.0)
        score = strategy._calculate_orientation_score(face, Pose.SIDE_LEFT)

        assert score == 1.0  # Perfect match for side-left

    def test_orientation_score_default_to_frontal(self, strategy):
        """Test that None pose defaults to frontal"""
        face = FaceData(yaw=0.0, pitch=0.0)
        score = strategy._calculate_orientation_score(face, None)

        assert score == 1.0  # Should match frontal

    def test_orientation_score_range(self, strategy):
        """Test orientation score is always in valid range"""
        for yaw in range(-90, 91, 30):
            for pitch in range(-90, 91, 30):
                face = FaceData(yaw=float(yaw), pitch=float(pitch))
                score = strategy._calculate_orientation_score(face, Pose.FRONT)
                assert 0.0 <= score <= 1.0


# ============================================
# HeuristicStrategy.score_candidates Tests
# ============================================

class TestScoreCandidates:
    """Tests for the score_candidates method"""

    def test_score_candidates_returns_ranking_result(self, strategy, mock_image_basic):
        """Test that score_candidates returns RankingResult"""
        semantic_scores = {mock_image_basic.display_path: 0.85}

        result = strategy.score_candidates([mock_image_basic], semantic_scores)

        assert isinstance(result, RankingResult)

    def test_score_candidates_display_metrics(self, strategy, mock_image_basic):
        """Test that display_metrics are populated"""
        semantic_scores = {mock_image_basic.display_path: 0.85}

        result = strategy.score_candidates([mock_image_basic], semantic_scores)

        path = mock_image_basic.display_path
        assert path in result.display_metrics
        assert "final_relevance" in result.display_metrics[path]
        assert "semantic" in result.display_metrics[path]

    def test_score_candidates_training_features(self, strategy, mock_image_basic):
        """Test that training_features are populated"""
        semantic_scores = {mock_image_basic.display_path: 0.85}

        result = strategy.score_candidates([mock_image_basic], semantic_scores)

        path = mock_image_basic.display_path
        assert path in result.training_features
        assert "semantic_score" in result.training_features[path]
        assert "aesthetic_score" in result.training_features[path]

    def test_score_candidates_with_target_name(self, strategy, mock_image_with_face):
        """Test scoring with target_name for face quality"""
        semantic_scores = {mock_image_with_face.display_path: 0.80}

        result = strategy.score_candidates(
            [mock_image_with_face],
            semantic_scores,
            target_name="John"
        )

        path = mock_image_with_face.display_path
        # Should have face-related metrics
        assert "f_blur" in result.display_metrics[path]
        assert "f_score" in result.display_metrics[path]

    def test_score_candidates_multiple_images(self, strategy):
        """Test scoring multiple images"""
        img1 = ImageAnalysisResult(
            original_path="/path/img1.jpg",
            photo_id="1",
            display_path="/display/img1.jpg"
        )
        img1.aesthetic_score = 5.0
        img1.global_blur = 800.0
        img1.iso = 100
        img1.global_contrast = 60.0
        img1.semantic_vector = np.random.rand(512).astype(np.float32)
        img1.faces = None

        img2 = ImageAnalysisResult(
            original_path="/path/img2.jpg",
            photo_id="2",
            display_path="/display/img2.jpg"
        )
        img2.aesthetic_score = 4.2
        img2.global_blur = 300.0
        img2.iso = 800
        img2.global_contrast = 40.0
        img2.semantic_vector = np.random.rand(512).astype(np.float32)
        img2.faces = None

        semantic_scores = {
            "/display/img1.jpg": 0.90,
            "/display/img2.jpg": 0.85
        }

        result = strategy.score_candidates([img1, img2], semantic_scores)

        # Both images should have metrics
        assert "/display/img1.jpg" in result.display_metrics
        assert "/display/img2.jpg" in result.display_metrics

    def test_score_candidates_semantic_weight_dominates(self, strategy):
        """Test that semantic score has the most weight"""
        # Image with high semantic, low quality
        img_high_semantic = ImageAnalysisResult(
            original_path="/path/hs.jpg",
            photo_id="hs",
            display_path="/display/hs.jpg"
        )
        img_high_semantic.aesthetic_score = 4.0
        img_high_semantic.global_blur = 100.0
        img_high_semantic.iso = 1600
        img_high_semantic.global_contrast = 35.0
        img_high_semantic.semantic_vector = np.random.rand(512).astype(np.float32)
        img_high_semantic.faces = None

        # Image with low semantic, high quality
        img_high_quality = ImageAnalysisResult(
            original_path="/path/hq.jpg",
            photo_id="hq",
            display_path="/display/hq.jpg"
        )
        img_high_quality.aesthetic_score = 5.5
        img_high_quality.global_blur = 1400.0
        img_high_quality.iso = 100
        img_high_quality.global_contrast = 80.0
        img_high_quality.semantic_vector = np.random.rand(512).astype(np.float32)
        img_high_quality.faces = None

        semantic_scores = {
            "/display/hs.jpg": 0.95,  # High semantic
            "/display/hq.jpg": 0.50   # Low semantic
        }

        result = strategy.score_candidates(
            [img_high_semantic, img_high_quality],
            semantic_scores
        )

        # High semantic image should have higher final_relevance
        hs_score = result.display_metrics["/display/hs.jpg"]["final_relevance"]
        hq_score = result.display_metrics["/display/hq.jpg"]["final_relevance"]

        assert hs_score > hq_score


# ============================================
# Integration Tests
# ============================================

class TestHeuristicStrategyIntegration:
    """Integration tests for HeuristicStrategy"""

    def test_full_ranking_workflow(self, strategy):
        """Test complete ranking workflow with multiple candidates"""
        # Create diverse set of images
        images = []
        semantic_scores = {}

        for i in range(5):
            img = ImageAnalysisResult(
                original_path=f"/path/img{i}.jpg",
                photo_id=str(i),
                display_path=f"/display/img{i}.jpg"
            )
            img.aesthetic_score = 4.0 + (i * 0.3)
            img.global_blur = 200.0 + (i * 100)
            img.iso = 100 * (i + 1)
            img.global_contrast = 40.0 + (i * 5)
            img.semantic_vector = np.random.rand(512).astype(np.float32)
            img.faces = None

            images.append(img)
            semantic_scores[img.display_path] = 0.9 - (i * 0.1)

        result = strategy.score_candidates(images, semantic_scores)

        # Verify all images are in results
        assert len(result.display_metrics) == 5
        assert len(result.training_features) == 5

        # Verify metrics structure
        for path in semantic_scores:
            assert "final_relevance" in result.display_metrics[path]
            assert "semantic_score" in result.training_features[path]
