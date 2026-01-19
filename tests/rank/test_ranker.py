"""
Tests for src/rank/ranker.py

Tests cover:
- SearchResultRanker initialization
- SearchResultRanker.set_strategy()
- SearchResultRanker.process()
- SearchResultRanker._apply_mmr()
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from src.rank.ranker import SearchResultRanker
from src.rank.base import BaseRankingStrategy, RankingResult
from src.rank.heuristic_ranker import HeuristicStrategy
from src.common.types import ImageAnalysisResult
from src.pose.pose import Pose


# ============================================
# Test Fixtures
# ============================================

@pytest.fixture
def ranker():
    """Create a SearchResultRanker with default strategy"""
    with patch("src.rank.ranker.logger"):
        return SearchResultRanker()


@pytest.fixture
def mock_images():
    """Create multiple mock ImageAnalysisResult objects with semantic vectors"""
    images = []
    for i in range(5):
        result = ImageAnalysisResult(
            original_path=f"/path/img{i}.jpg",
            photo_id=f"test{i}",
            display_path=f"/display/img{i}.jpg"
        )
        result.aesthetic_score = 4.0 + (i * 0.2)
        result.global_blur = 400.0 + (i * 50)
        result.global_brightness = 120.0
        result.global_contrast = 50.0
        result.iso = 200
        result.timestamp = f"2023061{i}120000"
        result.month = 6
        result.time_period = "afternoon"
        # Create semi-random but reproducible vectors
        np.random.seed(i)
        result.semantic_vector = np.random.rand(512).astype(np.float32)
        result.faces = None
        images.append(result)
    return images


@pytest.fixture
def mock_semantic_scores(mock_images):
    """Create semantic scores for mock images"""
    return {img.display_path: 0.9 - (i * 0.1) for i, img in enumerate(mock_images)}


class MockStrategy(BaseRankingStrategy):
    """Mock strategy for testing"""

    def score_candidates(self, results, semantic_scores, caption_scores=None, target_name=None, pose=None):
        scored = [(r, semantic_scores.get(r.display_path, 0.0)) for r in results]
        scored.sort(key=lambda x: x[1], reverse=True)

        return RankingResult(
            ranked_results=[r for r, _ in scored],
            display_metrics={
                r.display_path: {"final_relevance": s}
                for r, s in scored
            },
            training_features={
                r.display_path: {"semantic_score": s}
                for r, s in scored
            }
        )


# ============================================
# SearchResultRanker Initialization Tests
# ============================================

class TestSearchResultRankerInit:
    """Tests for SearchResultRanker initialization"""

    def test_init_default_strategy(self):
        """Test initialization with default HeuristicStrategy"""
        with patch("src.rank.ranker.logger"):
            ranker = SearchResultRanker()
            assert isinstance(ranker.strategy, HeuristicStrategy)

    def test_init_custom_strategy(self):
        """Test initialization with custom strategy"""
        mock_strategy = MockStrategy()
        with patch("src.rank.ranker.logger"):
            ranker = SearchResultRanker(strategy=mock_strategy)
            assert ranker.strategy == mock_strategy

    def test_init_device_detection(self):
        """Test that device is set correctly"""
        with patch("src.rank.ranker.logger"):
            ranker = SearchResultRanker()
            assert ranker.device in ['cuda', 'cpu']


# ============================================
# SearchResultRanker.set_strategy Tests
# ============================================

class TestSetStrategy:
    """Tests for set_strategy method"""

    def test_set_strategy_changes_strategy(self, ranker):
        """Test that set_strategy changes the active strategy"""
        new_strategy = MockStrategy()
        ranker.set_strategy(new_strategy)
        assert ranker.strategy == new_strategy

    def test_set_strategy_logs_change(self):
        """Test that strategy change is logged"""
        with patch("src.rank.ranker.logger") as mock_logger:
            ranker = SearchResultRanker()
            new_strategy = MockStrategy()
            ranker.set_strategy(new_strategy)
            mock_logger.info.assert_called()


# ============================================
# SearchResultRanker.process Tests
# ============================================

class TestProcess:
    """Tests for process method"""

    def test_process_empty_results(self, ranker):
        """Test processing empty results list"""
        result = ranker.process([], {})

        assert isinstance(result, RankingResult)
        assert result.ranked_results == []
        assert result.display_metrics == {}
        assert result.training_features == {}

    def test_process_returns_ranking_result(self, ranker, mock_images, mock_semantic_scores):
        """Test that process returns RankingResult"""
        result = ranker.process(mock_images, mock_semantic_scores)

        assert isinstance(result, RankingResult)
        assert len(result.ranked_results) > 0

    def test_process_respects_top_k(self, ranker, mock_images, mock_semantic_scores):
        """Test that process respects top_k parameter"""
        result = ranker.process(
            mock_images,
            mock_semantic_scores,
            top_k=3
        )

        assert len(result.ranked_results) == 3

    def test_process_top_k_larger_than_results(self, ranker, mock_images, mock_semantic_scores):
        """Test top_k larger than number of results"""
        result = ranker.process(
            mock_images,
            mock_semantic_scores,
            top_k=100
        )

        # Should return all available results
        assert len(result.ranked_results) == len(mock_images)

    def test_process_filters_metrics_to_top_k(self, ranker, mock_images, mock_semantic_scores):
        """Test that metrics are filtered to only top_k results"""
        result = ranker.process(
            mock_images,
            mock_semantic_scores,
            top_k=2
        )

        # Only top 2 results should have metrics
        assert len(result.display_metrics) == 2
        assert len(result.training_features) == 2

    def test_process_with_target_name(self, ranker, mock_images, mock_semantic_scores):
        """Test processing with target_name parameter"""
        result = ranker.process(
            mock_images,
            mock_semantic_scores,
            target_name="John"
        )

        assert isinstance(result, RankingResult)

    def test_process_with_pose(self, ranker, mock_images, mock_semantic_scores):
        """Test processing with pose parameter"""
        result = ranker.process(
            mock_images,
            mock_semantic_scores,
            pose=Pose.FRONT
        )

        assert isinstance(result, RankingResult)

    def test_process_lambda_param(self, ranker, mock_images, mock_semantic_scores):
        """Test that lambda_param affects MMR diversity"""
        # High lambda = more relevance, less diversity
        result_high_lambda = ranker.process(
            mock_images,
            mock_semantic_scores,
            lambda_param=1.0,
            top_k=3
        )

        # Low lambda = more diversity, less relevance
        result_low_lambda = ranker.process(
            mock_images,
            mock_semantic_scores,
            lambda_param=0.3,
            top_k=3
        )

        # Both should return valid results
        assert len(result_high_lambda.ranked_results) == 3
        assert len(result_low_lambda.ranked_results) == 3


# ============================================
# SearchResultRanker._apply_mmr Tests
# ============================================

class TestApplyMMR:
    """Tests for _apply_mmr method"""

    def test_mmr_empty_items(self, ranker):
        """Test MMR with empty items list"""
        result = ranker._apply_mmr([], 0.6, 10, {})
        assert result == []

    def test_mmr_returns_correct_count(self, ranker, mock_images):
        """Test MMR returns correct number of items"""
        display_metrics = {
            img.display_path: {"final_relevance": 0.9 - (i * 0.1)}
            for i, img in enumerate(mock_images)
        }

        result = ranker._apply_mmr(mock_images, 0.6, 3, display_metrics)
        assert len(result) == 3

    def test_mmr_adds_rank_to_metrics(self, ranker, mock_images):
        """Test that MMR adds mmr_rank to display_metrics"""
        display_metrics = {
            img.display_path: {"final_relevance": 0.9 - (i * 0.1)}
            for i, img in enumerate(mock_images)
        }

        result = ranker._apply_mmr(mock_images, 0.6, 3, display_metrics)

        # Check that selected items have mmr_rank
        for i, item in enumerate(result):
            assert "mmr_rank" in display_metrics[item.display_path]
            assert display_metrics[item.display_path]["mmr_rank"] == i + 1

    def test_mmr_high_lambda_prefers_relevance(self, ranker, mock_images):
        """Test that high lambda prefers high relevance scores"""
        display_metrics = {
            img.display_path: {"final_relevance": 0.9 - (i * 0.1)}
            for i, img in enumerate(mock_images)
        }

        # Lambda = 1.0 means pure relevance, no diversity
        result = ranker._apply_mmr(mock_images, 1.0, 3, display_metrics)

        # First result should be the highest relevance
        assert result[0] == mock_images[0]

    def test_mmr_handles_identical_scores(self, ranker, mock_images):
        """Test MMR handles identical relevance scores"""
        display_metrics = {
            img.display_path: {"final_relevance": 0.5}  # All same score
            for img in mock_images
        }

        # Should not crash, should use rank position as fallback
        result = ranker._apply_mmr(mock_images, 0.6, 3, display_metrics)
        assert len(result) == 3

    def test_mmr_top_k_larger_than_candidates(self, ranker, mock_images):
        """Test MMR when top_k exceeds number of candidates"""
        display_metrics = {
            img.display_path: {"final_relevance": 0.5}
            for img in mock_images
        }

        result = ranker._apply_mmr(mock_images, 0.6, 100, display_metrics)

        # Should return all candidates
        assert len(result) == len(mock_images)


# ============================================
# Integration Tests
# ============================================

class TestSearchResultRankerIntegration:
    """Integration tests for SearchResultRanker"""

    def test_full_ranking_pipeline(self, mock_images, mock_semantic_scores):
        """Test complete ranking pipeline"""
        with patch("src.rank.ranker.logger"):
            ranker = SearchResultRanker()

        result = ranker.process(
            mock_images,
            mock_semantic_scores,
            lambda_param=0.7,
            top_k=3
        )

        # Verify structure
        assert isinstance(result, RankingResult)
        assert len(result.ranked_results) == 3
        assert len(result.display_metrics) == 3
        assert len(result.training_features) == 3

        # Verify all selected items have mmr_rank
        for item in result.ranked_results:
            path = item.display_path
            assert "mmr_rank" in result.display_metrics[path]

    def test_switch_strategy_and_process(self, mock_images, mock_semantic_scores):
        """Test switching strategy and processing"""
        with patch("src.rank.ranker.logger"):
            ranker = SearchResultRanker()

        # Process with default strategy
        result1 = ranker.process(mock_images, mock_semantic_scores, top_k=3)

        # Switch to mock strategy
        ranker.set_strategy(MockStrategy())

        # Process with new strategy
        result2 = ranker.process(mock_images, mock_semantic_scores, top_k=3)

        # Both should return valid results
        assert len(result1.ranked_results) == 3
        assert len(result2.ranked_results) == 3

    def test_diversity_with_similar_vectors(self):
        """Test MMR diversity with very similar semantic vectors"""
        # Create images with very similar vectors
        base_vector = np.random.rand(512).astype(np.float32)

        images = []
        for i in range(5):
            result = ImageAnalysisResult(
                original_path=f"/path/img{i}.jpg",
                photo_id=f"test{i}",
                display_path=f"/display/img{i}.jpg"
            )
            result.aesthetic_score = 4.5
            result.global_blur = 500.0
            result.global_brightness = 128.0
            result.global_contrast = 55.0
            result.iso = 200
            result.timestamp = "20230615120000"
            result.month = 6
            result.time_period = "afternoon"
            # Very similar vectors (small perturbations)
            result.semantic_vector = base_vector + (np.random.rand(512) * 0.01).astype(np.float32)
            result.faces = None
            images.append(result)

        semantic_scores = {img.display_path: 0.9 for img in images}

        with patch("src.rank.ranker.logger"):
            ranker = SearchResultRanker()

        # With low lambda, should try to diversify
        result = ranker.process(
            images,
            semantic_scores,
            lambda_param=0.3,
            top_k=3
        )

        assert len(result.ranked_results) == 3
