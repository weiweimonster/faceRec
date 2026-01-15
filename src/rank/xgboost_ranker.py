import xgboost as xgb
import pandas as pd
import numpy as np
import os

from sympy import false
from typing import List, Dict, Optional, Any
from src.common.types import ImageAnalysisResult
from src.pose.pose import Pose
from .base import BaseRankingStrategy, ScoredCandidate
from src.util.logger import logger

class XGBoostRanker(BaseRankingStrategy):
    def __init__(self, model_path="ltr_model.json"):
        self.model = None
        self.feature_cols = ["g_contrast", "f_conf", "meta_year"]
        self.feature_cols_map = {
            "g_contrast": "global_contrast",
            "g_brightness": "global_brightness",
            "g_aesthetic": "aesthetic_score",
            "f_blur": "global_blur",  # Example: Mapping global blur if face blur isn't available

            # --- Calculated Features (Lambdas) ---
            "meta_year": lambda item: int(str(item.timestamp)[:4]) if item.timestamp and len(str(item.timestamp)) >= 4 else -1.0,

            "meta_iso": lambda item: float(item.iso) if item.iso is not None else 0.0,

            # Example: Handle complex Face Logic (assuming item.faces exists)
            # If no face, return 0.0. If faces, return max confidence.
            "f_conf": lambda item: max([f.confidence for f in item.faces]) if getattr(item, 'faces', None) else 0.0
        }

        if os.path.exists(model_path):
            logger.info(f"🧠 Loading XGBoost Ranker from {model_path}...")
            self.model = xgb.Booster()
            self.model.load_model(model_path)
        else:
            logger.error(f"⚠️ Warning: Model not found at {model_path}. Ranker will be skipped.")

    def score_candidates(
            self,
            results: List[ImageAnalysisResult],
            semantic_scores: Dict[str, float],
            target_name: Optional[str] = None,
            pose: Optional[Pose] = None
    ) -> List[ScoredCandidate]:

        if not self.model or not results:
            logger.error("No model initialized")
            return [(r, semantic_scores.get(r.display_path, 0.0), {}, None) for r in results]

        logger.info(f"Extracting features from {len(results)}")
        # Extract features
        rows: List[Dict[str, float]] = []
        for item in results:
            row: Dict[str, float] = {}
            for col in self.feature_cols:
                rule = self.feature_cols_map.get(col)
                try:
                    val = 0.0
                    # Case A: Lambda
                    if callable(rule):
                        val = rule(item)
                    # Case B: Attribute mapping
                    elif isinstance(rule, str):
                        val = getattr(item, rule, 0.0)
                    # Case C: No mapping, assume attribute name matches model feature
                    else:
                        val = getattr(item, col, 0.0)
                    row[col] = float(val) if val is not None else 0.0
                except Exception as e:
                    logger.debug(f"Feature extraction failed for {col}: {e}")
                    row[col] = 0.0
            rows.append(row)

        # Predict
        # Convert to DataFrame to ensure column order matches feature_names
        logger.info(f"Loading {len(rows)} into data frame")
        df = pd.DataFrame(rows)
        dtest = xgb.DMatrix(df[self.feature_cols])

        logger.info(f"Ranking {len(results)} using XGBoost")
        # XGBoost predict returns a numpy array of floats
        scores = self.model.predict(dtest)

        # 3. Return Format
        scored_items: List[ScoredCandidate] = []
        for i, item in enumerate(results):
            score_val = float(scores[i])
            metrics = {"xgboost_score": round(score_val, 4)}
            scored_items.append((item, score_val, metrics, None))

        return scored_items




