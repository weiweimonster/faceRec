import xgboost as xgb
import pandas as pd
import os

from typing import List, Dict, Optional
from src.common.types import ImageAnalysisResult
from src.pose.pose import Pose
from .base import BaseRankingStrategy, RankingResult
from src.util.logger import logger
from src.features.container import FeatureExtractor

class XGBoostRanker(BaseRankingStrategy):
    def __init__(self, model_path="ltr_model.json"):
        self.model = None
        self.feature_cols = []
        self.extractor = FeatureExtractor()

        if os.path.exists(model_path):
            logger.info(f"Loading XGBoost model from {model_path}")
            self.model = xgb.Booster()
            self.model.load_model(model_path)
            # Use feature names from the model as source of truth
            self.feature_cols = self.model.feature_names
            logger.info(f"Model uses {len(self.feature_cols)} features: {self.feature_cols}")
        else:
            logger.error(f"Model not found at {model_path}. XGBoost ranker disabled.")

    def score_candidates(
            self,
            results: List[ImageAnalysisResult],
            semantic_scores: Dict[str, float],
            target_name: Optional[str] = None,
            pose: Optional[Pose] = None
    ) -> RankingResult:
        """
        Score candidates using trained XGBoost model.

        Returns:
            RankingResult with:
            - ranked_results: Sorted by XGBoost prediction
            - display_metrics: Just XGBoost score for UI
            - training_features: Full feature dict for logging
        """

        if not self.model or not results:
            logger.warning("XGBoost model not loaded or no results. Returning unranked.")
            return RankingResult(
                ranked_results=results,
                display_metrics={r.display_path: {} for r in results},
                training_features={r.display_path: {} for r in results}
            )

        logger.info(f"Extracting {len(self.feature_cols)} features from {len(results)} candidates")

        # Extract features for all candidates
        training_features: Dict[str, Dict[str, float]] = {}
        rows: List[Dict[str, float]] = []

        for item in results:
            context = {"semantic_score": semantic_scores.get(item.display_path, 0.0)}
            features = self.extractor.extract_from_result(
                result=item,
                context=context,
                target_face_name=target_name,
                feature_subset=self.feature_cols
            )
            rows.append(features)
            training_features[item.display_path] = features

        # Predict
        # Convert to DataFrame to ensure column order matches feature_names
        logger.info(f"Loading {len(rows)} into data frame")
        df = pd.DataFrame(rows)
        dtest = xgb.DMatrix(df[self.feature_cols])

        logger.info(f"Ranking {len(results)} using XGBoost")
        # XGBoost predict returns a numpy array of floats
        scores = self.model.predict(dtest)

        # 3. Return Format
        scored_items = list(zip(results, scores))
        scored_items.sort(key=lambda x: x[1], reverse=True)

        ranked_results = [item for item, score in scored_items]

        # Build display metrics (simple XGBoost score for UI)
        display_metrics = {
            item.display_path: {"xgboost_score": round(float(score), 4)}
            for item, score in scored_items
        }

        logger.info(f"XGBoost ranked {len(ranked_results)} results. Top score: {scores.max():.4f}, Min score: {scores.min(): .4f}")

        return RankingResult(
            ranked_results=ranked_results,
            display_metrics=display_metrics,
            training_features=training_features
        )




