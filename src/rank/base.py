from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from src.common.types import ImageAnalysisResult
from src.pose.pose import Pose

ScoredCandidate = Tuple[ImageAnalysisResult, float, Dict[str, Any]]

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
    ) -> List[ScoredCandidate]:
        """
        Calculates scores for a list of candidates.

        Args:
            results: List of image analysis objects.
            semantic_scores: Dictionary mapping file paths to vector similarity scores.
            target_name: (Optional) The name of the person being searched for.
            pose: (Optional) The specific pose requested.

        Returns:
            List of tuples: (ImageAnalysisResult, Final_Score, Debug_Metrics_Dict)
        """
        pass