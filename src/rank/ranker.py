import torch
import numpy as np
from typing import List, Dict, Any, Optional
from src.common.types import ImageAnalysisResult
from src.pose.pose import Pose
from src.util.logger import logger
from .base import BaseRankingStrategy, RankingResult
from .heuristic_ranker import HeuristicStrategy

class SearchResultRanker:
    def __init__(self, strategy: Optional[BaseRankingStrategy] = None):
        self.strategy: BaseRankingStrategy = strategy if strategy else HeuristicStrategy()
        logger.info(f"init ranking strategy: {strategy}")
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def set_strategy(self, strategy: BaseRankingStrategy) -> None:
        logger.info(f"🔄 Switching Ranker Strategy to {type(strategy).__name__}")
        self.strategy = strategy

    def process(
        self,
        results: List[ImageAnalysisResult],
        semantic_scores: Dict[str, float],
        caption_scores: Optional[Dict[str, float]] = None,
        target_name: Optional[str] = None,
        lambda_param: float = 0.6,
        top_k: int = 30,
        pose: Optional[Pose] = None
    ) -> RankingResult:
        """
        Apply strategy ranking + MMR diversity.

        Args:
            results: Candidate images to rank
            semantic_scores: CLIP similarity scores
            caption_scores: Caption similarity scores
            target_name: Person name for face quality boost
            lambda_param: MMR balance (1.0=relevance only, 0.0=diversity only)
            top_k: Number of results to return
            pose: Pose filter

        Returns:
            RankingResult with diversified top-k results
        """

        if not results:
            logger.warning("No results to rank")
            return RankingResult(
                ranked_results=[],
                display_metrics={},
                training_features={}
            )

        if caption_scores is None:
            logger.warning("Caption sore is none for ranking")
            caption_scores = {}

        ranking_result = self.strategy.score_candidates(
            results, semantic_scores, caption_scores, target_name, pose
        )

        logger.info(f"{type(self.strategy).__name__} ranked {len(ranking_result.ranked_results)} candidates")

        diversified_results = self._apply_mmr(
            ranking_result.ranked_results,
            lambda_param,
            top_k,
            ranking_result.display_metrics
        )

        top_k_paths = {r.display_path for r in diversified_results}

        filtered_display_metrics = {
            path: metrics
            for path, metrics in ranking_result.display_metrics.items()
            if path in top_k_paths
        }

        filtered_training_features = {
            path: features
            for path, features in ranking_result.training_features.items()
            if path in top_k_paths
        }

        return RankingResult(
            ranked_results=diversified_results,
            display_metrics=filtered_display_metrics,
            training_features=filtered_training_features
        )

    def _apply_mmr(
        self,
        items: List[ImageAnalysisResult],
        lambda_param: float,
        top_k: int,
        display_metrics: Dict[str, Dict[str, Any]]
    ) -> List[ImageAnalysisResult]:
        """
        Maximal Marginal Relevance for diversity.

        Balances relevance (from strategy ranking) with diversity
        (semantic dissimilarity to already-selected results).

        Args:
            items: Pre-ranked results from strategy
            lambda_param: Relevance vs diversity weight
            top_k: Number to select
            display_metrics: Metrics dict (modified in-place to add mmr_rank)

        Returns:
            Diversified list of top-k results
        """
        num_candidates = len(items)
        if num_candidates == 0:
            logger.error(f"No results to rank. Returning empty list.")
            return []

        logger.info(f"Performing MMR ranking on {num_candidates} candidates with lambda={lambda_param}")

        embeddings = np.array([c.semantic_vector for c in items])
        c_tensor = torch.tensor(embeddings, dtype=torch.float32, device=self.device)

        scores = []
        for item in items:
            metrics = display_metrics.get(item.display_path, {})

            # Try different score keys depending on strategy
            score = (
                    metrics.get("final_relevance") or  # Heuristic ranker
                    metrics.get("xgboost_score") or  # XGBoost ranker
                    0.0  # Fallback
            )
            scores.append(score)

        scores = np.array(scores, dtype=np.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32, device=self.device)

        if num_candidates > 1:
            s_min, s_max = torch.min(scores_tensor), torch.max(scores_tensor)
            if s_max > s_min:
                scores_tensor = (scores_tensor - s_min) / (s_max - s_min)
                logger.info(f"Normalized score before MMR ranking. Top score: {scores_tensor.max().item()}, Min: score {scores_tensor.min().item()}")
            else:
                # All scores are identical - fall back to rank position
                logger.warning("All relevance scores identical, using rank position")
                scores_tensor = torch.linspace(1.0, 0.0, num_candidates, device=self.device)

        # Similarity Matrix
        c_norm = torch.nn.functional.normalize(c_tensor, p=2, dim=1)
        sim_matrix = torch.mm(c_norm, c_norm.T)

        selected_indices = []
        candidate_mask = torch.ones(num_candidates, dtype=torch.bool, device=self.device)
        max_sim_to_selected = torch.zeros(num_candidates, device=self.device)

        logger.info(f"Starting MMR Loop to select top {top_k} candidates")
        for i in range(min(top_k, num_candidates)):
            # Calculate MMR values for all candidates
            mmr_vals = (lambda_param * scores_tensor) - ((1 - lambda_param) * max_sim_to_selected)

            # Mask out already selected
            mmr_vals[~candidate_mask] = -float('inf')

            best_idx = int(torch.argmax(mmr_vals).item())
            selected_indices.append(best_idx)
            candidate_mask[best_idx] = False

            # Update Similarity Penalty
            # Find max similarity between current candidate and the one just selected
            max_sim_to_selected = torch.max(max_sim_to_selected, sim_matrix[:, best_idx])

            # Update Metrics with Rank
            item_path = items[best_idx].display_path
            if item_path in display_metrics:
                display_metrics[item_path]["mmr_rank"] = i + 1

        logger.info(f"MMR selected {len(selected_indices)} diverse results")
        return [items[i] for i in selected_indices]