from src.util.logger import logger
from src.common.types import FaceData, ImageAnalysisResult
from typing import Optional, List, Dict, Any, Tuple
import torch
import numpy as np

class SearchResultRanker:
    # Tunable Hyperparameters
    # Adjust these to change how quality is calculated
    QUALITY_BOUNDS = {
        "blur": {"min": 100, "max": 700},
        "height": {"min": 100, "max": 600},
        "width": {"min": 120, "max": 700},
        "brightness_target": 120,
        # TODO: Think if this would hurt, since if we ask to retrieve side faces, then all of the penalties will be applied
        "angle_limit": 45  # Degrees: penalty starts after this
    }

    # Weights for the final score components
    WEIGHTS = {
        "semantic": 0.9,
        "quality": 0.1,
        "q_blur": 0.4,
        "q_size": 0.2,        # Combined height/width
        "q_brightness": 0.2,
        "q_orientation": 0.2, # Looking at camera
        "aesthetic_score": 0.2
    }

    def __init__(self, results: List[ImageAnalysisResult], semantic_scores: Dict[str, float]):
        logger.info(f"Initializing search results ranker with {len(results)}")
        self.results = results
        self.semantic_scores = semantic_scores
        # TODO: Make this an environmental variables
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Metrics of each photo: Path -> Metric
        self.metrics: Dict[str, Dict[str, Any]]  = {}

    def _normalize(self, value: float, min_v: float, max_v: float) -> float:
        if value < min_v: return 0.0
        if value > max_v: return 1.0
        return (value - min_v) / (max_v - min_v)

    def calculate_face_quality(self, result: ImageAnalysisResult, target_name: str):
        """Calculates a quality score using class-member bounds and weights."""
        if not result.faces or len(result.faces) == 0:
            logger.error(f"Faces are empty or no faces found")
            return 0,0

        face = [f for f in result.faces if f.name.lower() == target_name.lower()]
        if not face:
            logger.warning("No face detected, skip calculating face quality.....")
            return 0.0
        elif len(face) > 1:
            logger.error(f"{len(face)} detected with {target_name} in one photos.")
            logger.error(f"Skip calculating face quality.....")
            return 0.0

        best_face = face[0]
        cfg = self.QUALITY_BOUNDS
        wts = self.WEIGHTS

        # 1. Sharpness (Blur)
        s_blur = self._normalize(best_face.blur_score or 0, cfg["blur"]["min"], cfg["blur"]["max"])

        # 2. Size (Resolution)
        h_score = self._normalize(best_face.bbox[3] if best_face.bbox else 0, cfg["height"]["min"], cfg["height"]["max"])
        w_score = self._normalize(best_face.bbox[2] if best_face.bbox else 0, cfg["width"]["min"], cfg["width"]["max"])
        s_size = (h_score + w_score) / 2

        # 3. Lighting
        dist_b = abs((best_face.brightness or cfg["brightness_target"]) - cfg["brightness_target"])
        s_bright = 1.0 - (dist_b / cfg["brightness_target"])

        # 4. Orientation (Gaze)
        yaw, pitch = abs(best_face.yaw or 0), abs(best_face.pitch or 0)
        s_orient = 1.0 - (self._normalize(max(yaw, pitch), 0, cfg["angle_limit"]))

        # Weighted Sum calculation
        quality_score = (
                (wts["q_blur"] * s_blur) +
                (wts["q_size"] * s_size) +
                (wts["q_brightness"] * s_bright) +
                (wts["q_orientation"] * s_orient)
        )

        # Detailed breakdown for self.metrics
        breakdown = {
            "norm_blur": round(s_blur, 3),
            "norm_size": round(s_size, 3),
            "norm_brightness": round(s_bright, 3),
            "norm_orientation": round(s_orient, 3),
            "total_quality_raw": round(quality_score, 3)
        }

        return quality_score * (best_face.confidence or 1.0), breakdown

    # TODO: Add parameter for each filtering mechanism, so user can choose which type of mechanism to use
    def rank(self, target_name: str = None, lambda_param: float = 0.6, top_k: int = 20) -> Tuple[List[ImageAnalysisResult], Dict[str, Any]]:
        """Main entry point for ranking logic."""
        if not self.results:
            logger.error("Empty results to rank. Returning nothing")
            return [], {}

        # Quality-Boosted Semantic Scoring
        logger.info(f"Calculating Final Rank Score for {target_name}")
        for item in self.results:
            path = item.display_path
            semantic_sim = self.semantic_scores.get(path, 0.0)

            quality_boost, face_metrics = 0.0, {}
            if target_name:
                quality_boost, face_metrics = self.calculate_face_quality(item, target_name)

            final_relevance = (self.WEIGHTS["semantic"] * semantic_sim) + (self.WEIGHTS["quality"] * quality_boost)

            self.metrics[path] = {
                "semantic_sim": round(semantic_sim, 3),
                "quality_boost": round(quality_boost, 3),
                "final_relevance": round(final_relevance, 3),
                **face_metrics
            }

        return self._apply_mmr(lambda_param, top_k), self.metrics

    def _apply_mmr(self, lambda_param: float, top_k: int) -> List[ImageAnalysisResult]:
        """
        Maximal Marginal Relevance.
        Now pulls relevance scores directly from self.metrics using display_path.
        """
        logger.info("Applying MMR for deduplication")
        num_candidates = len(self.results)
        if num_candidates == 0: return []

        # 1. Prepare Tensors
        embeddings = np.array([c.semantic_vector for c in self.results])
        c_tensor = torch.tensor(embeddings, dtype=torch.float32, device=self.device)

        # Pull scores directly from self.metrics using the path as the key
        raw_scores = [self.metrics[c.display_path]["final_relevance"] for c in self.results]
        scores = torch.tensor(raw_scores, dtype=torch.float32, device=self.device)

        # 2. Normalize Relevance Scores
        s_min, s_max = torch.min(scores), torch.max(scores)
        norm_scores = (scores - s_min) / (s_max - s_min + 1e-6) if s_max > s_min else scores

        # 3. Calculate Similarity Matrix
        c_norm = torch.nn.functional.normalize(c_tensor, p=2, dim=1)
        logger.info("Calculating Similarity Matric for MMR")
        sim_matrix = torch.mm(c_norm, c_norm.T)

        # 4. Iterative Selection
        selected_indices = []
        candidate_mask = torch.ones(num_candidates, dtype=torch.bool, device=self.device)
        max_sim_to_selected = torch.zeros(num_candidates, device=self.device)

        for i in range(min(top_k, num_candidates)):
            # MMR formula
            mmr_vals = (lambda_param * norm_scores) - ((1 - lambda_param) * max_sim_to_selected)
            mmr_vals[~candidate_mask] = -float('inf')

            best_idx = torch.argmax(mmr_vals).item()
            selected_indices.append(best_idx)
            candidate_mask[best_idx] = False

            # Update redundancy penalty
            max_sim_to_selected = torch.max(max_sim_to_selected, sim_matrix[:, best_idx])

            # Update debug log
            path = self.results[best_idx].display_path
            self.metrics[path]["mmr_rank"] = i + 1

        return [self.results[i] for i in selected_indices]