import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.common.types import ImageAnalysisResult
from src.pose.pose import Pose
from src.util.logger import logger
from .base import BaseRankingStrategy
from .heuristic_ranker import HeuristicStrategy
from src.rank.rank_metrics import PictureRankMetrics

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
        target_name: Optional[str] = None,
        lambda_param: float = 0.6,
        top_k: int = 50,
        pose: Optional[Pose] = None
    ) -> Tuple[List[ImageAnalysisResult], Dict[str, Any], Optional[Dict[str, Any]]]:

        if not results:
            logger.error(f"No results to rank. Returning empty list.")
            return [], {}, None

        scored_candidates = self.strategy.score_candidates(
            results, semantic_scores, target_name, pose
        )

        logger.info(f"Received {len(scored_candidates)} candidates from {self.strategy} ranker")

        # Separate scores and metrics
        items = [x[0] for x in scored_candidates] # Items
        scores = [x[1] for x in scored_candidates] # Final score by the strategy

        # Create metrics dict: Path -> metric dict return by the strategy
        all_metrics: Dict[str, Any] = {
            x[0].display_path: x[2] for x in scored_candidates
        }

        photo_rank_metrics: Dict[str, Any] = {
            x[0].display_path: x[3] for x in scored_candidates
        }

        final_results = self._apply_mmr(items, scores, lambda_param, top_k, all_metrics)

        return final_results, all_metrics, photo_rank_metrics

    def _apply_mmr(
        self,
        items: List[ImageAnalysisResult],
        scores_list: List[float],
        lambda_param: float,
        top_k: int,
        metric_store: Dict[str, Any]
    ) -> List[ImageAnalysisResult]:

        num_candidates = len(items)
        if num_candidates == 0:
            logger.error(f"No results to rank. Returning empty list.")
            return []

        logger.info(f"Performing MMR ranking on {num_candidates} candidates with lambda={lambda_param}")

        embeddings = np.array([c.semantic_vector for c in items])
        c_tensor = torch.tensor(embeddings, dtype=torch.float32, device=self.device)

        scores = torch.tensor(scores_list, dtype=torch.float32, device=self.device)

        if num_candidates > 1:
            s_min, s_max = torch.min(scores), torch.max(scores)
            if s_max > s_min:
                scores = (scores - s_min) / (s_max - s_min)

        # Similarity Matrix
        c_norm = torch.nn.functional.normalize(c_tensor, p=2, dim=1)
        sim_matrix = torch.mm(c_norm, c_norm.T)

        selected_indices = []
        candidate_mask = torch.ones(num_candidates, dtype=torch.bool, device=self.device)
        max_sim_to_selected = torch.zeros(num_candidates, device=self.device)

        logger.info(f"Starting MMR Loop to select top {top_k} candidates")
        for i in range(min(top_k, num_candidates)):
            # Calculate MMR values for all candidates
            mmr_vals = (lambda_param * scores) - ((1 - lambda_param) * max_sim_to_selected)

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
            if item_path in metric_store:
                metric_store[item_path]["mmr_rank"] = i + 1

        return [items[i] for i in selected_indices]