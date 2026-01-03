import pytest
import numpy as np
import torch
from src.common.types import ImageAnalysisResult, FaceData
from src.retrieval.search_results import SearchResultRanker

@pytest.fixture
def sample_results():
    """Creates a list of ImageAnalysisResults for testing Ranker logic."""
    def _create_result(path, vec_val, blur=400, yaw=0, aesthetic=7.0):
        # We create a dummy embedding for the face as well
        dummy_face_emb = np.full((512,), vec_val, dtype=np.float32)

        return ImageAnalysisResult(
            display_path=path,
            photo_id=f"id_{path}",
            original_path=f"raw/{path}",
            timestamp="2025-01-01 10:00:00",
            # Semantic vector used for MMR diversity
            semantic_vector=np.full((512,), vec_val, dtype=np.float32),
            aesthetic_score=aesthetic,
            original_width=1920,
            original_height=1080,
            faces=[
                FaceData(
                    name="Alice",
                    embedding=dummy_face_emb, # Essential for clustering/ID logic
                    bbox=[100, 100, 200, 200], # [x, y, w, h]
                    blur_score=blur,
                    brightness=120,
                    yaw=yaw,
                    pitch=0,
                    roll=0,
                    confidence=1.0,
                    shot_type="Medium-Shot"
                )
            ]
        )

    # Result A: Strong semantic match, good quality
    res_a = _create_result("photo_a.jpg", 0.1, blur=600, yaw=0)

    # Result B: Identical vector to A (0.1), but poor quality (blurry)
    # This is the "redundant" test case for MMR
    res_b = _create_result("photo_b.jpg", 0.1, blur=100, yaw=0)

    # Result C: Different vector (0.5), very high quality
    # This tests if high quality can overcome lower semantic similarity
    res_c = _create_result("photo_c.jpg", 0.5, blur=700, yaw=0)

    return [res_a, res_b, res_c]

@pytest.fixture
def semantic_scores():
    return {
        "photo_a.jpg": 0.9,
        "photo_b.jpg": 0.89,
        "photo_c.jpg": 0.5
    }

## --- Tests ---

def test_normalization_clamping(sample_results, semantic_scores):
    """Ensure values outside min/max bounds are clamped to 0.0 or 1.0."""
    ranker = SearchResultRanker(sample_results, semantic_scores)

    # Value below min
    assert ranker._normalize(50, 100, 700) == 0.0
    # Value above max
    assert ranker._normalize(800, 100, 700) == 1.0
    # Value at mid
    assert ranker._normalize(400, 100, 700) == 0.5

def test_calculate_face_quality_metrics(sample_results, semantic_scores):
    """Verify quality components (blur, size, etc.) are correctly computed."""
    ranker = SearchResultRanker(sample_results, semantic_scores)

    # Test Alice in photo_a
    score, metrics = ranker.calculate_face_quality(sample_results[0], "Alice")

    assert score > 0
    assert "norm_blur" in metrics
    assert "norm_size" in metrics
    # Since blur is 600 and max is 700, norm_blur should be high
    assert metrics["norm_blur"] > 0.8

def test_rank_integration_and_metrics_storage(sample_results, semantic_scores):
    """Verify that the rank function populates self.metrics sidecar."""
    ranker = SearchResultRanker(sample_results, semantic_scores)
    final_results, metrics = ranker.rank(target_name="Alice")

    assert len(metrics) == 3
    for path in ["photo_a.jpg", "photo_b.jpg", "photo_c.jpg"]:
        assert "final_relevance" in metrics[path]
        assert "semantic_sim" in metrics[path]
        assert "mmr_rank" in metrics[path]

def test_mmr_diversity_impact(sample_results, semantic_scores):
    """Verify that MMR penalizes redundant (visually similar) images."""
    # TODO: Fix this test cases
    ranker = SearchResultRanker(sample_results, semantic_scores)

    # Lambda = 0.0 (Pure Diversity, ignores relevance)
    # Lambda = 1.0 (Pure Relevance, ignores diversity)

    # With high lambda, Photo B (blurry but similar) should be higher
    results_rel, _ = ranker.rank(target_name="Alice", lambda_param=1.0)
    idx_b_rel = next(i for i, r in enumerate(results_rel) if r.display_path == "photo_b.jpg")

    # With low lambda, Photo B should be pushed down because it's too similar to A
    results_div, _ = ranker.rank(target_name="Alice", lambda_param=0.1)
    idx_b_div = next(i for i, r in enumerate(results_div) if r.display_path == "photo_b.jpg")

    assert idx_b_div > idx_b_rel

def test_empty_results_handling():
    """Ensure ranker handles empty input gracefully."""
    ranker = SearchResultRanker([], {})
    results, metrics = ranker.rank()
    assert results == []
    assert metrics == {}

def test_target_name_case_insensitivity(sample_results, semantic_scores):
    """Ensure 'alice' matches 'Alice'."""
    ranker = SearchResultRanker(sample_results, semantic_scores)
    score, _ = ranker.calculate_face_quality(sample_results[0], "alice")
    assert score > 0