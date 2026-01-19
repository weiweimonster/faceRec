from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from src.common.types import ImageAnalysisResult
from src.pose.pose import Pose

@dataclass
class RankingResult:
    """
    Container for ranking pipeline outputs.

    Attributes:
        ranked_results: ImageAnalysisResult objects in ranked order (best first)
        display_metrics: Dict[path -> metrics] for UI display
                        Contains weighted/normalized scores (e.g., {"semantic": 0.85, "g_aesthetic": 0.9})
        training_features: Dict[path -> features] for database logging and XGBoost training
                          Contains raw feature values (e.g., {"aesthetic_score": 4.5, "g_blur": 120.3})

    Example:
        result = RankingResult(
            ranked_results=[img1, img2, img3],
            display_metrics={
                "path/to/img1.jpg": {"final_relevance": 0.92, "semantic": 0.88, "g_aesthetic": 0.95},
                "path/to/img2.jpg": {"final_relevance": 0.87, "semantic": 0.82, "g_aesthetic": 0.91}
            },
            training_features={
                "path/to/img1.jpg": {"semantic_score": 0.88, "aesthetic_score": 4.8, "g_blur": 156.2},
                "path/to/img2.jpg": {"semantic_score": 0.82, "aesthetic_score": 4.6, "g_blur": 142.1}
            }
        )
    """
    ranked_results: List[ImageAnalysisResult]
    display_metrics: Dict[str, Dict[str, Any]]
    training_features: Dict[str, dict[str, float]]

class BaseRankingStrategy(ABC):
    """
    Abstract Base Class for scoring strategies.
    Responsible ONLY for calculating the score of each item.
    """

    @abstractmethod
    def score_candidates(
        self,
        results: List[ImageAnalysisResult],
        semantic_scores: Dict[str, float],
        target_name: Optional[str] = None,
        pose: Optional[Pose] = None
    ) -> RankingResult:
        """
        Calculates scores for a list of candidates.

        Args:
            results: List of image analysis objects.
            semantic_scores: Dictionary mapping file paths to vector similarity scores.
            target_name: (Optional) The name of the person being searched for.
            pose: (Optional) The specific pose requested.

        Returns:
            RankingResult with sorted results and metrics
        """
        pass