"""
Offline evaluation metrics for ranking quality assessment.

This module provides standard IR metrics (NDCG, Precision@K, MRR) to evaluate
ranking model quality using historical click data as ground truth.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from src.db.storage import DatabaseManager
from src.util.logger import logger


def compute_dcg_at_k(relevances: List[int], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at K.

    Args:
        relevances: List of relevance scores (1 for relevant, 0 for not)
        k: Cutoff position

    Returns:
        DCG@K score
    """
    relevances = np.array(relevances[:k])
    if len(relevances) == 0:
        return 0.0

    # DCG = sum(rel_i / log2(i + 2)) for i in range(k)
    discounts = np.log2(np.arange(len(relevances)) + 2)
    return np.sum(relevances / discounts)


def compute_ndcg_at_k(predictions: List[str], ground_truth: List[str], k: int = 5) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K.

    NDCG measures ranking quality by comparing the predicted ranking against
    an ideal ranking where all relevant items appear at the top.

    Args:
        predictions: List of photo_ids in predicted rank order
        ground_truth: List of photo_ids that are relevant (clicked)
        k: Cutoff position for evaluation

    Returns:
        NDCG@K score between 0 and 1 (1 = perfect ranking)
    """
    if not predictions or not ground_truth:
        return 0.0

    ground_truth_set = set(ground_truth)

    # Create relevance vector for predicted ranking
    relevances = [1 if pid in ground_truth_set else 0 for pid in predictions[:k]]

    # Compute actual DCG
    dcg = compute_dcg_at_k(relevances, k)

    # Compute ideal DCG (all relevant items at top)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = compute_dcg_at_k(ideal_relevances, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_precision_at_k(predictions: List[str], ground_truth: List[str], k: int = 5) -> float:
    """
    Compute Precision at K - fraction of top-K predictions that are relevant.

    Args:
        predictions: List of photo_ids in predicted rank order
        ground_truth: List of photo_ids that are relevant (clicked)
        k: Cutoff position for evaluation

    Returns:
        Precision@K score between 0 and 1
    """
    if not predictions or not ground_truth:
        return 0.0

    ground_truth_set = set(ground_truth)
    top_k = predictions[:k]

    relevant_in_top_k = sum(1 for pid in top_k if pid in ground_truth_set)
    return relevant_in_top_k / k


def compute_recall_at_k(predictions: List[str], ground_truth: List[str], k: int = 5) -> float:
    """
    Compute Recall at K - fraction of relevant items found in top-K.

    Args:
        predictions: List of photo_ids in predicted rank order
        ground_truth: List of photo_ids that are relevant (clicked)
        k: Cutoff position for evaluation

    Returns:
        Recall@K score between 0 and 1
    """
    if not predictions or not ground_truth:
        return 0.0

    ground_truth_set = set(ground_truth)
    top_k = predictions[:k]

    relevant_in_top_k = sum(1 for pid in top_k if pid in ground_truth_set)
    return relevant_in_top_k / len(ground_truth_set)


def compute_mrr(predictions: List[str], ground_truth: List[str]) -> float:
    """
    Compute Mean Reciprocal Rank - inverse of position of first relevant result.

    MRR measures how quickly a user finds a relevant result. Higher is better.

    Args:
        predictions: List of photo_ids in predicted rank order
        ground_truth: List of photo_ids that are relevant (clicked)

    Returns:
        MRR score between 0 and 1 (1 = first result is relevant)
    """
    if not predictions or not ground_truth:
        return 0.0

    ground_truth_set = set(ground_truth)

    for i, pid in enumerate(predictions):
        if pid in ground_truth_set:
            return 1.0 / (i + 1)

    return 0.0


def compute_average_precision(predictions: List[str], ground_truth: List[str]) -> float:
    """
    Compute Average Precision for a single query.

    AP summarizes the precision-recall curve by averaging precision at each
    relevant item's position.

    Args:
        predictions: List of photo_ids in predicted rank order
        ground_truth: List of photo_ids that are relevant (clicked)

    Returns:
        AP score between 0 and 1
    """
    if not predictions or not ground_truth:
        return 0.0

    ground_truth_set = set(ground_truth)
    relevant_count = 0
    precision_sum = 0.0

    for i, pid in enumerate(predictions):
        if pid in ground_truth_set:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i

    if relevant_count == 0:
        return 0.0

    return precision_sum / len(ground_truth_set)


class OfflineEvaluator:
    """
    Evaluator for computing offline ranking metrics from historical data.

    Uses search_interactions (clicks) as ground truth to evaluate ranking quality.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    def get_session_data(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve session data with impressions and clicks for evaluation.

        Args:
            model_name: Optional filter for specific ranking model

        Returns:
            List of session dicts with impressions and clicks
        """
        try:
            query = """
                SELECT
                    sh.session_id,
                    sh.ranking_model,
                    GROUP_CONCAT(DISTINCT si.photo_id || ':' || si.position) as impressions,
                    GROUP_CONCAT(DISTINCT CASE WHEN sint.label = 1 THEN sint.photo_id END) as clicks
                FROM search_history sh
                JOIN search_impressions si ON sh.session_id = si.session_id
                LEFT JOIN search_interactions sint ON sh.session_id = sint.session_id AND sint.label = 1
                WHERE 1=1
            """
            params = []

            if model_name:
                query += " AND sh.ranking_model = ?"
                params.append(model_name)

            query += " GROUP BY sh.session_id HAVING impressions IS NOT NULL"

            self.db.cursor.execute(query, params)
            rows = self.db.cursor.fetchall()

            sessions = []
            for row in rows:
                # Parse impressions (photo_id:position pairs)
                impressions_str = row[2] or ""
                impression_pairs = [p.split(":") for p in impressions_str.split(",") if ":" in p]

                # Sort by position to get ranked order
                impression_pairs.sort(key=lambda x: int(x[1]))
                ranked_predictions = [p[0] for p in impression_pairs]

                # Parse clicks
                clicks_str = row[3] or ""
                clicks = [c for c in clicks_str.split(",") if c]

                if ranked_predictions:  # Only include sessions with impressions
                    sessions.append({
                        "session_id": row[0],
                        "model": row[1],
                        "predictions": ranked_predictions,
                        "ground_truth": clicks
                    })

            return sessions
        except Exception as e:
            logger.error(f"Failed to get session data: {e}")
            return []

    def evaluate_model(self, model_name: str, k: int = 5) -> Dict[str, float]:
        """
        Run full offline evaluation suite for a specific model.

        Args:
            model_name: Name of the ranking model to evaluate
            k: Cutoff position for @K metrics

        Returns:
            Dict with NDCG@K, Precision@K, Recall@K, MRR, MAP
        """
        sessions = self.get_session_data(model_name)

        if not sessions:
            return {
                f"ndcg@{k}": 0.0,
                f"precision@{k}": 0.0,
                f"recall@{k}": 0.0,
                "mrr": 0.0,
                "map": 0.0,
                "sessions_evaluated": 0
            }

        # Filter to sessions with at least one click
        sessions_with_clicks = [s for s in sessions if s["ground_truth"]]

        if not sessions_with_clicks:
            return {
                f"ndcg@{k}": 0.0,
                f"precision@{k}": 0.0,
                f"recall@{k}": 0.0,
                "mrr": 0.0,
                "map": 0.0,
                "sessions_evaluated": 0
            }

        # Compute metrics for each session
        ndcg_scores = []
        precision_scores = []
        recall_scores = []
        mrr_scores = []
        ap_scores = []

        for session in sessions_with_clicks:
            preds = session["predictions"]
            truth = session["ground_truth"]

            ndcg_scores.append(compute_ndcg_at_k(preds, truth, k))
            precision_scores.append(compute_precision_at_k(preds, truth, k))
            recall_scores.append(compute_recall_at_k(preds, truth, k))
            mrr_scores.append(compute_mrr(preds, truth))
            ap_scores.append(compute_average_precision(preds, truth))

        return {
            f"ndcg@{k}": round(np.mean(ndcg_scores), 3),
            f"precision@{k}": round(np.mean(precision_scores), 3),
            f"recall@{k}": round(np.mean(recall_scores), 3),
            "mrr": round(np.mean(mrr_scores), 3),
            "map": round(np.mean(ap_scores), 3),
            "sessions_evaluated": len(sessions_with_clicks)
        }

    def run_offline_evaluation(self, k: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Run full offline evaluation for all models and return comparison.

        Args:
            k: Cutoff position for @K metrics

        Returns:
            Dict mapping model names to their metrics
        """
        results = {}

        # Get unique models
        try:
            self.db.cursor.execute("SELECT DISTINCT ranking_model FROM search_history WHERE ranking_model IS NOT NULL")
            models = [row[0] for row in self.db.cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            models = ["xgboost", "heuristic"]

        for model in models:
            results[model] = self.evaluate_model(model, k)
            logger.info(f"Evaluated {model}: {results[model]}")

        return results

    def get_comparison_summary(self, k: int = 5) -> Dict[str, Any]:
        """
        Get a portfolio-ready comparison summary between models.

        Args:
            k: Cutoff position for @K metrics

        Returns:
            Dict with model metrics and lift calculations
        """
        metrics = self.run_offline_evaluation(k)

        summary = {"models": metrics, "k": k}

        # Calculate lift if we have both models
        if "xgboost" in metrics and "heuristic" in metrics:
            xgb = metrics["xgboost"]
            heur = metrics["heuristic"]

            lifts = {}
            for metric_name in [f"ndcg@{k}", f"precision@{k}", "mrr", "map"]:
                if heur.get(metric_name, 0) > 0:
                    lift = ((xgb.get(metric_name, 0) - heur.get(metric_name, 0)) / heur.get(metric_name, 0)) * 100
                    lifts[f"{metric_name}_lift"] = round(lift, 1)

            summary["lifts"] = lifts

        return summary
