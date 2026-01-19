"""
Tests for src/rank/base.py

Tests cover:
- RankingResult dataclass
- BaseRankingStrategy abstract class
"""

import pytest
from typing import List, Dict, Optional

from src.rank.base import RankingResult, BaseRankingStrategy
from src.common.types import ImageAnalysisResult
from src.pose.pose import Pose


# ============================================
# RankingResult Tests
# ============================================

class TestRankingResult:
    """Tests for the RankingResult dataclass"""

    def test_ranking_result_creation(self):
        """Test creating a basic RankingResult"""
        result = RankingResult(
            ranked_results=[],
            display_metrics={},
            training_features={}
        )
        assert result.ranked_results == []
        assert result.display_metrics == {}
        assert result.training_features == {}

    def test_ranking_result_with_data(self):
        """Test RankingResult with actual data"""
        img1 = ImageAnalysisResult(
            original_path="/path/img1.jpg",
            photo_id="1",
            display_path="/display/img1.jpg"
        )
        img2 = ImageAnalysisResult(
            original_path="/path/img2.jpg",
            photo_id="2",
            display_path="/display/img2.jpg"
        )

        display_metrics = {
            "/display/img1.jpg": {"final_relevance": 0.92, "semantic": 0.88},
            "/display/img2.jpg": {"final_relevance": 0.85, "semantic": 0.80}
        }

        training_features = {
            "/display/img1.jpg": {"semantic_score": 0.88, "aesthetic_score": 4.5},
            "/display/img2.jpg": {"semantic_score": 0.80, "aesthetic_score": 4.2}
        }

        result = RankingResult(
            ranked_results=[img1, img2],
            display_metrics=display_metrics,
            training_features=training_features
        )

        assert len(result.ranked_results) == 2
        assert result.ranked_results[0] == img1
        assert result.ranked_results[1] == img2
        assert result.display_metrics["/display/img1.jpg"]["final_relevance"] == 0.92
        assert result.training_features["/display/img2.jpg"]["aesthetic_score"] == 4.2

    def test_ranking_result_metrics_access(self):
        """Test accessing metrics by path"""
        result = RankingResult(
            ranked_results=[],
            display_metrics={
                "path1": {"score": 0.9},
                "path2": {"score": 0.8}
            },
            training_features={}
        )

        assert "path1" in result.display_metrics
        assert result.display_metrics["path1"]["score"] == 0.9
        assert "path3" not in result.display_metrics


# ============================================
# BaseRankingStrategy Tests
# ============================================

class ConcreteStrategy(BaseRankingStrategy):
    """Concrete implementation for testing abstract class"""

    def score_candidates(
        self,
        results: List[ImageAnalysisResult],
        semantic_scores: Dict[str, float],
        target_name: Optional[str] = None,
        pose: Optional[Pose] = None
    ) -> RankingResult:
        # Simple implementation: sort by semantic score descending
        scored = [(r, semantic_scores.get(r.display_path, 0.0)) for r in results]
        scored.sort(key=lambda x: x[1], reverse=True)

        return RankingResult(
            ranked_results=[r for r, _ in scored],
            display_metrics={r.display_path: {"score": s} for r, s in scored},
            training_features={r.display_path: {"semantic_score": s} for r, s in scored}
        )


class TestBaseRankingStrategy:
    """Tests for the BaseRankingStrategy abstract class"""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseRankingStrategy cannot be directly instantiated"""
        with pytest.raises(TypeError):
            BaseRankingStrategy()

    def test_concrete_implementation_works(self):
        """Test that a concrete implementation can be instantiated and used"""
        strategy = ConcreteStrategy()

        img1 = ImageAnalysisResult(
            original_path="/path/img1.jpg",
            photo_id="1",
            display_path="/display/img1.jpg"
        )
        img2 = ImageAnalysisResult(
            original_path="/path/img2.jpg",
            photo_id="2",
            display_path="/display/img2.jpg"
        )

        semantic_scores = {
            "/display/img1.jpg": 0.7,
            "/display/img2.jpg": 0.9
        }

        result = strategy.score_candidates([img1, img2], semantic_scores)

        # Should be sorted by semantic score (img2 first)
        assert result.ranked_results[0] == img2
        assert result.ranked_results[1] == img1

    def test_concrete_implementation_with_target_name(self):
        """Test concrete implementation with target_name parameter"""
        strategy = ConcreteStrategy()

        img = ImageAnalysisResult(
            original_path="/path/img.jpg",
            photo_id="1",
            display_path="/display/img.jpg"
        )

        result = strategy.score_candidates(
            results=[img],
            semantic_scores={"/display/img.jpg": 0.8},
            target_name="John"
        )

        assert len(result.ranked_results) == 1

    def test_concrete_implementation_with_pose(self):
        """Test concrete implementation with pose parameter"""
        strategy = ConcreteStrategy()

        img = ImageAnalysisResult(
            original_path="/path/img.jpg",
            photo_id="1",
            display_path="/display/img.jpg"
        )

        result = strategy.score_candidates(
            results=[img],
            semantic_scores={"/display/img.jpg": 0.8},
            pose=Pose.FRONT
        )

        assert len(result.ranked_results) == 1

    def test_concrete_implementation_empty_results(self):
        """Test concrete implementation with empty results"""
        strategy = ConcreteStrategy()

        result = strategy.score_candidates(
            results=[],
            semantic_scores={}
        )

        assert result.ranked_results == []
        assert result.display_metrics == {}
        assert result.training_features == {}
