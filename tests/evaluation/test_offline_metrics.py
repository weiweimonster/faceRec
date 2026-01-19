"""
Tests for src/evaluation/offline_metrics.py

Tests cover:
- DCG computation
- NDCG@K computation
- Precision@K computation
- Recall@K computation
- MRR computation
- Average Precision computation
- Edge cases (empty inputs, no relevant items, etc.)
"""

import pytest
import numpy as np

from src.evaluation.offline_metrics import (
    compute_dcg_at_k,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_mrr,
    compute_average_precision
)


# ============================================
# DCG Tests
# ============================================

class TestComputeDCG:
    """Tests for compute_dcg_at_k"""

    def test_dcg_perfect_ranking(self):
        """Test DCG with all relevant items at top"""
        relevances = [1, 1, 1, 0, 0]
        dcg = compute_dcg_at_k(relevances, k=5)
        # DCG = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.63 + 0.5 = ~2.13
        assert dcg > 2.0

    def test_dcg_no_relevant(self):
        """Test DCG with no relevant items"""
        relevances = [0, 0, 0, 0, 0]
        dcg = compute_dcg_at_k(relevances, k=5)
        assert dcg == 0.0

    def test_dcg_empty(self):
        """Test DCG with empty relevances"""
        dcg = compute_dcg_at_k([], k=5)
        assert dcg == 0.0

    def test_dcg_k_smaller_than_list(self):
        """Test DCG truncates at k"""
        relevances = [1, 1, 1, 1, 1]
        dcg_k3 = compute_dcg_at_k(relevances, k=3)
        dcg_k5 = compute_dcg_at_k(relevances, k=5)
        assert dcg_k3 < dcg_k5


# ============================================
# NDCG@K Tests
# ============================================

class TestComputeNDCG:
    """Tests for compute_ndcg_at_k"""

    def test_ndcg_perfect_ranking(self):
        """Test NDCG with perfect ranking (all relevant items at top)"""
        predictions = ["a", "b", "c", "d", "e"]
        ground_truth = ["a", "b"]
        ndcg = compute_ndcg_at_k(predictions, ground_truth, k=5)
        assert ndcg == 1.0

    def test_ndcg_worst_ranking(self):
        """Test NDCG with relevant items at bottom"""
        predictions = ["c", "d", "e", "a", "b"]
        ground_truth = ["a", "b"]
        ndcg = compute_ndcg_at_k(predictions, ground_truth, k=5)
        # Relevant items at positions 4, 5 - should be less than perfect
        assert ndcg < 1.0
        assert ndcg > 0.0

    def test_ndcg_no_relevant_items(self):
        """Test NDCG when no predictions match ground truth"""
        predictions = ["x", "y", "z"]
        ground_truth = ["a", "b"]
        ndcg = compute_ndcg_at_k(predictions, ground_truth, k=3)
        assert ndcg == 0.0

    def test_ndcg_empty_predictions(self):
        """Test NDCG with empty predictions"""
        ndcg = compute_ndcg_at_k([], ["a", "b"], k=5)
        assert ndcg == 0.0

    def test_ndcg_empty_ground_truth(self):
        """Test NDCG with empty ground truth"""
        ndcg = compute_ndcg_at_k(["a", "b", "c"], [], k=5)
        assert ndcg == 0.0

    def test_ndcg_k_limits_evaluation(self):
        """Test that k parameter limits evaluation depth"""
        predictions = ["x", "x", "x", "a", "b"]
        ground_truth = ["a", "b"]
        ndcg_k2 = compute_ndcg_at_k(predictions, ground_truth, k=2)
        ndcg_k5 = compute_ndcg_at_k(predictions, ground_truth, k=5)
        # With k=2, relevant items are not seen
        assert ndcg_k2 == 0.0
        # With k=5, relevant items are found
        assert ndcg_k5 > 0.0


# ============================================
# Precision@K Tests
# ============================================

class TestComputePrecision:
    """Tests for compute_precision_at_k"""

    def test_precision_all_relevant(self):
        """Test precision when all top-K are relevant"""
        predictions = ["a", "b", "c", "d", "e"]
        ground_truth = ["a", "b", "c", "d", "e"]
        precision = compute_precision_at_k(predictions, ground_truth, k=5)
        assert precision == 1.0

    def test_precision_none_relevant(self):
        """Test precision when no top-K are relevant"""
        predictions = ["x", "y", "z"]
        ground_truth = ["a", "b", "c"]
        precision = compute_precision_at_k(predictions, ground_truth, k=3)
        assert precision == 0.0

    def test_precision_half_relevant(self):
        """Test precision with 50% relevant"""
        predictions = ["a", "x", "b", "y"]
        ground_truth = ["a", "b"]
        precision = compute_precision_at_k(predictions, ground_truth, k=4)
        assert precision == 0.5

    def test_precision_k_truncates(self):
        """Test that k truncates predictions"""
        predictions = ["a", "b", "x", "y", "z"]
        ground_truth = ["a", "b"]
        precision_k2 = compute_precision_at_k(predictions, ground_truth, k=2)
        precision_k5 = compute_precision_at_k(predictions, ground_truth, k=5)
        assert precision_k2 == 1.0  # 2/2
        assert precision_k5 == 0.4  # 2/5

    def test_precision_empty_predictions(self):
        """Test precision with empty predictions"""
        precision = compute_precision_at_k([], ["a"], k=5)
        assert precision == 0.0

    def test_precision_empty_ground_truth(self):
        """Test precision with empty ground truth"""
        precision = compute_precision_at_k(["a", "b"], [], k=5)
        assert precision == 0.0


# ============================================
# Recall@K Tests
# ============================================

class TestComputeRecall:
    """Tests for compute_recall_at_k"""

    def test_recall_all_found(self):
        """Test recall when all relevant items found in top-K"""
        predictions = ["a", "b", "x", "y", "z"]
        ground_truth = ["a", "b"]
        recall = compute_recall_at_k(predictions, ground_truth, k=5)
        assert recall == 1.0

    def test_recall_none_found(self):
        """Test recall when no relevant items found"""
        predictions = ["x", "y", "z"]
        ground_truth = ["a", "b"]
        recall = compute_recall_at_k(predictions, ground_truth, k=3)
        assert recall == 0.0

    def test_recall_partial(self):
        """Test recall with partial match"""
        predictions = ["a", "x", "y"]
        ground_truth = ["a", "b"]
        recall = compute_recall_at_k(predictions, ground_truth, k=3)
        assert recall == 0.5  # 1/2 relevant items found

    def test_recall_empty_predictions(self):
        """Test recall with empty predictions"""
        recall = compute_recall_at_k([], ["a", "b"], k=5)
        assert recall == 0.0

    def test_recall_empty_ground_truth(self):
        """Test recall with empty ground truth"""
        recall = compute_recall_at_k(["a", "b"], [], k=5)
        assert recall == 0.0


# ============================================
# MRR Tests
# ============================================

class TestComputeMRR:
    """Tests for compute_mrr"""

    def test_mrr_first_is_relevant(self):
        """Test MRR when first item is relevant"""
        predictions = ["a", "b", "c"]
        ground_truth = ["a"]
        mrr = compute_mrr(predictions, ground_truth)
        assert mrr == 1.0

    def test_mrr_second_is_relevant(self):
        """Test MRR when second item is relevant"""
        predictions = ["x", "a", "c"]
        ground_truth = ["a"]
        mrr = compute_mrr(predictions, ground_truth)
        assert mrr == 0.5

    def test_mrr_third_is_relevant(self):
        """Test MRR when third item is relevant"""
        predictions = ["x", "y", "a"]
        ground_truth = ["a"]
        mrr = compute_mrr(predictions, ground_truth)
        assert mrr == pytest.approx(1/3)

    def test_mrr_none_relevant(self):
        """Test MRR when no items are relevant"""
        predictions = ["x", "y", "z"]
        ground_truth = ["a", "b"]
        mrr = compute_mrr(predictions, ground_truth)
        assert mrr == 0.0

    def test_mrr_multiple_relevant_returns_first(self):
        """Test MRR uses position of FIRST relevant item"""
        predictions = ["x", "a", "b"]
        ground_truth = ["a", "b"]
        mrr = compute_mrr(predictions, ground_truth)
        assert mrr == 0.5  # First relevant at position 2

    def test_mrr_empty_predictions(self):
        """Test MRR with empty predictions"""
        mrr = compute_mrr([], ["a"])
        assert mrr == 0.0

    def test_mrr_empty_ground_truth(self):
        """Test MRR with empty ground truth"""
        mrr = compute_mrr(["a", "b"], [])
        assert mrr == 0.0


# ============================================
# Average Precision Tests
# ============================================

class TestComputeAveragePrecision:
    """Tests for compute_average_precision"""

    def test_ap_perfect_ranking(self):
        """Test AP with all relevant items at top"""
        predictions = ["a", "b", "c", "x", "y"]
        ground_truth = ["a", "b", "c"]
        ap = compute_average_precision(predictions, ground_truth)
        # P@1=1, P@2=1, P@3=1 -> AP = (1+1+1)/3 = 1.0
        assert ap == 1.0

    def test_ap_interleaved_relevant(self):
        """Test AP with interleaved relevant items"""
        predictions = ["a", "x", "b", "y", "c"]
        ground_truth = ["a", "b", "c"]
        ap = compute_average_precision(predictions, ground_truth)
        # P@1=1/1=1, P@3=2/3, P@5=3/5 -> AP = (1 + 2/3 + 3/5)/3
        expected = (1 + 2/3 + 3/5) / 3
        assert ap == pytest.approx(expected)

    def test_ap_no_relevant(self):
        """Test AP when no predictions match"""
        predictions = ["x", "y", "z"]
        ground_truth = ["a", "b"]
        ap = compute_average_precision(predictions, ground_truth)
        assert ap == 0.0

    def test_ap_empty_predictions(self):
        """Test AP with empty predictions"""
        ap = compute_average_precision([], ["a", "b"])
        assert ap == 0.0

    def test_ap_empty_ground_truth(self):
        """Test AP with empty ground truth"""
        ap = compute_average_precision(["a", "b"], [])
        assert ap == 0.0


# ============================================
# Integration Tests
# ============================================

class TestMetricsIntegration:
    """Integration tests for ranking metrics"""

    def test_consistent_ordering(self):
        """Test that better rankings give higher scores for all metrics"""
        ground_truth = ["a", "b", "c"]

        # Perfect ranking
        perfect = ["a", "b", "c", "x", "y"]

        # Good ranking (relevant items in top 3, but not in order)
        good = ["b", "a", "c", "x", "y"]

        # Bad ranking (relevant items scattered)
        bad = ["x", "a", "y", "b", "c"]

        # All metrics should prefer perfect >= good > bad
        assert compute_ndcg_at_k(perfect, ground_truth, k=5) >= compute_ndcg_at_k(good, ground_truth, k=5)
        assert compute_ndcg_at_k(good, ground_truth, k=5) > compute_ndcg_at_k(bad, ground_truth, k=5)

        assert compute_precision_at_k(perfect, ground_truth, k=3) >= compute_precision_at_k(bad, ground_truth, k=3)

        assert compute_mrr(perfect, ground_truth) >= compute_mrr(good, ground_truth)
        assert compute_mrr(good, ground_truth) >= compute_mrr(bad, ground_truth)

    def test_metrics_bounded_0_to_1(self):
        """Test that all metrics are bounded between 0 and 1"""
        predictions = ["a", "x", "b", "y", "c"]
        ground_truth = ["a", "b", "c"]

        ndcg = compute_ndcg_at_k(predictions, ground_truth, k=5)
        precision = compute_precision_at_k(predictions, ground_truth, k=5)
        recall = compute_recall_at_k(predictions, ground_truth, k=5)
        mrr = compute_mrr(predictions, ground_truth)
        ap = compute_average_precision(predictions, ground_truth)

        for metric in [ndcg, precision, recall, mrr, ap]:
            assert 0.0 <= metric <= 1.0
