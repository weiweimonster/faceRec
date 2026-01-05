from src.pose.pose import Pose
from src.util.logger import logger
from src.common.types import FaceData, ImageAnalysisResult
from typing import Optional, List, Dict, Any, Tuple
import torch
import numpy as np
import math

class SearchResultRanker:
    # Tunable Hyperparameters
    # Adjust these to change how quality is calculated
    QUALITY_BOUNDS = {
        # Face Metrics
        "face_blur": {"min": 30, "max": 1200},
        "face_size": {"min": 50, "max": 550},
        "angle_limit": 45,

        # Global Metrics (New)
        "global_blur": {"min": 21.0, "max": 1400.0}, # Laplacian Variance
        "iso": {"min": 40, "max": 1600},            # Log Scale used below
        "contrast": {"min": 35.0, "max": 80.0},     # Std Dev
        "aesthetic": {"min": 4.0, "max": 5.5}       # CLIP Aesthetic Score
    }

    # Weights determine importance (Sum doesn't strictly have to be 1.0)
    WEIGHTS = {
        "semantic": 0.85,      # High relevance to the text query

        # Face Quality (Only applied if searching for a person)
        "face_quality": 0.1,

        # Global Quality (Applied to everything)
        "global_quality": 0.05,

        # Sub-weights for Global Score
        "w_aesthetic": 0.4,
        "w_sharpness": 0.3,
        "w_iso": 0.2,
        "w_contrast": 0.1
    }

    def __init__(self, results: List[ImageAnalysisResult], semantic_scores: Dict[str, float]):
        logger.info(f"Initializing Ranker with {len(results)} items")
        self.results = results
        self.semantic_scores = semantic_scores
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.metrics: Dict[str, Dict[str, Any]] = {}

    def _normalize(self, value: float, min_v: float, max_v: float) -> float:
        """Linear normalization to 0.0 - 1.0"""
        if value is None: return 0.0
        if value < min_v: return 0.0
        if value > max_v: return 1.0
        return (value - min_v) / (max_v - min_v)

    def _calculate_iso_score(self, iso_val: Optional[int]) -> float:
        """
        Calculates score based on ISO. Lower is better.
        Uses Log2 scale because ISO doubles at every stop.
        ISO 100 -> Score 1.0
        ISO 3200 -> Score 0.0
        """
        if iso_val is None: return 0.5 # Neutral if unknown
        if iso_val <= 100: return 1.0

        # Logarithmic normalization
        # log2(100) ~ 6.64, log2(3200) ~ 11.64
        min_log = math.log2(self.QUALITY_BOUNDS["iso"]["min"])
        max_log = math.log2(self.QUALITY_BOUNDS["iso"]["max"])

        curr_log = math.log2(iso_val)
        norm = (curr_log - min_log) / (max_log - min_log)

        # Invert because High ISO = Bad Quality
        score = 1.0 - max(0.0, min(1.0, norm))
        return score

    def _calculate_global_quality(self, item: ImageAnalysisResult) -> Tuple[float, Dict[str, float]]:
        """
        Scores the technical quality of the whole image (Scenery, Composition, Noise).
        """
        cfg = self.QUALITY_BOUNDS
        wts = self.WEIGHTS

        # 1. Aesthetics (The "Beauty" AI Score)
        s_aes = self._normalize(item.aesthetic_score, cfg["aesthetic"]["min"], cfg["aesthetic"]["max"])

        # 2. Global Sharpness
        s_blur = self._normalize(item.global_blur, cfg["global_blur"]["min"], cfg["global_blur"]["max"])

        # 3. Noise (ISO)
        s_iso = self._calculate_iso_score(item.iso)

        # 4. Contrast (Dynamic Range)
        s_con = self._normalize(item.global_contrast, cfg["contrast"]["min"], cfg["contrast"]["max"])

        # Weighted Sum
        score = (
                (s_aes * wts["w_aesthetic"]) +
                (s_blur * wts["w_sharpness"]) +
                (s_iso * wts["w_iso"]) +
                (s_con * wts["w_contrast"])
        )

        metrics = {
            "g_aesthetic": round(s_aes, 2),
            "g_sharpness": round(s_blur, 2),
            "g_iso": round(s_iso, 2),
            "g_contrast": round(s_con, 2),
            "global_score": round(score, 3)
        }
        return score, metrics

    def _calculate_face_quality(self, item: ImageAnalysisResult, target_name: str, requested_pose: Optional[Pose] = None) -> Tuple[float, Dict[str, float]]:
        if not item.faces: return 0.0, {}

        # 1. Fast Lookup: Just find the first face matching the name
        # We use next() to stop iterating immediately after finding the person
        target_face = next((f for f in item.faces if f.name.lower() == target_name.lower()), None)

        if not target_face:
            return 0.0, {}

        cfg = self.QUALITY_BOUNDS

        # 2. Correct Size Calculation (x2 - x1)
        # Bbox is [x1, y1, x2, y2]
        if target_face.bbox and len(target_face.bbox) == 4:
            width_px = target_face.bbox[2] - target_face.bbox[0]
            height_px = target_face.bbox[3] - target_face.bbox[1]
        else:
            width_px, height_px = 0, 0

        # 3. Score Calculations

        # Blur (Laplacian Variance of the face crop)
        s_blur = self._normalize(target_face.blur_score, cfg["face_blur"]["min"], cfg["face_blur"]["max"])

        # Size (Resolution relative to our expectations)
        h_score = self._normalize(height_px, cfg["face_size"]["min"], cfg["face_size"]["max"])
        w_score = self._normalize(width_px, cfg["face_size"]["min"], cfg["face_size"]["max"])
        s_size = (h_score + w_score) / 2

        # Orientation (Yaw/Pitch)
        s_orient = self._calculate_orientation_score(target_face, requested_pose)

        # Weighted Sum
        score = (s_blur * 0.4) + (s_size * 0.3) + (s_orient * 0.3)

        metrics = {
            "f_blur": round(s_blur, 2),
            "f_size": round(s_size, 2),
            "f_orient": round(s_orient, 2),
            "f_score": round(score, 3)
        }

        return score * target_face.confidence, metrics

    def _calculate_orientation_score(self, target_face: FaceData, requested_pose: Optional[Pose]) -> float:
        """
        Calculates a continuous score [0.0 - 1.0] based on how close
        the face is to the requested intent bullseye.
        """
        # 1. Determine the target anchor
        # If user didn't specify, we usually want Frontal shots
        target_pose = requested_pose if requested_pose else Pose.FRONT
        target_yaw, target_pitch = target_pose.anchors

        # 2. Calculate the Angular Distance
        # We use Euclidean distance in 2D angle space
        yaw_diff = target_face.yaw - target_yaw
        pitch_diff = target_face.pitch - target_pitch

        # Distance formula: sqrt(a² + b²)
        distance = math.sqrt(yaw_diff**2 + pitch_diff**2)

        # 3. Normalize the distance
        # We define a 'Radius of Acceptance'.
        # For example, if you are more than 60 degrees away, the score is 0.
        max_acceptable_distance = 60.0

        score = 1.0 - (distance / max_acceptable_distance)
        return max(0.0, score)

    def rank(self, target_name: str = None, lambda_param: float = 0.6, top_k: int = 50, pose: Optional[Pose] = None) -> Tuple[List[ImageAnalysisResult], Dict[str, Any]]:
        if not self.results: return [], {}

        logger.info(f"Ranking {len(self.results)} items. Target: {target_name or 'None'}, Pose: {pose or 'None'}")

        for item in self.results:
            path = item.display_path

            # A. Semantic Score (The Base)
            semantic_sim = self.semantic_scores.get(path, 0.0)

            # B. Global Quality (Applies to everyone)
            global_q, g_metrics = self._calculate_global_quality(item)

            # C. Face Quality (Conditional)
            face_q, f_metrics = 0.0, {}
            if target_name:
                face_q, f_metrics = self._calculate_face_quality(item, target_name, pose)

            # --- FINAL FORMULA ---
            # Score = (Semantic * 0.85) + (Face * 0.1) + (Global * 0.05)
            w = self.WEIGHTS

            final_score = (semantic_sim * w["semantic"]) + (global_q * w["global_quality"])

            if target_name:
                final_score += (face_q * w["face_quality"])

            # Store for Debugging / Frontend display
            self.metrics[path] = {
                "final_relevance": round(final_score, 4),
                "semantic": round(semantic_sim, 3),
                **g_metrics,
                **f_metrics
            }

        return self._apply_mmr(lambda_param, top_k), self.metrics

    def _apply_mmr(self, lambda_param: float, top_k: int) -> List[ImageAnalysisResult]:
        """
        Standard MMR for diversity.
        """
        num_candidates = len(self.results)
        if num_candidates == 0: return []

        # 1. Load Vectors & Scores
        embeddings = np.array([c.semantic_vector for c in self.results])
        c_tensor = torch.tensor(embeddings, dtype=torch.float32, device=self.device)

        scores_list = [self.metrics[c.display_path]["final_relevance"] for c in self.results]
        scores = torch.tensor(scores_list, dtype=torch.float32, device=self.device)

        # 2. Normalize Scores to [0, 1] relative to this batch
        # This is crucial so that '0.8' relevance compares meaningfully to '0.8' similarity
        if num_candidates > 1:
            s_min, s_max = torch.min(scores), torch.max(scores)
            if s_max > s_min:
                scores = (scores - s_min) / (s_max - s_min)

        # 3. Similarity Matrix
        c_norm = torch.nn.functional.normalize(c_tensor, p=2, dim=1)
        sim_matrix = torch.mm(c_norm, c_norm.T)

        # 4. Selection Loop
        selected_indices = []
        candidate_mask = torch.ones(num_candidates, dtype=torch.bool, device=self.device)
        max_sim_to_selected = torch.zeros(num_candidates, device=self.device)

        for i in range(min(top_k, num_candidates)):
            # MMR = (Lambda * Relevance) - ((1-Lambda) * SimilarityPenalty)
            mmr_vals = (lambda_param * scores) - ((1 - lambda_param) * max_sim_to_selected)
            mmr_vals[~candidate_mask] = -float('inf')

            best_idx = torch.argmax(mmr_vals).item()
            selected_indices.append(best_idx)
            candidate_mask[best_idx] = False

            # Update similarities for next round
            max_sim_to_selected = torch.max(max_sim_to_selected, sim_matrix[:, best_idx])

            # Log rank
            self.metrics[self.results[best_idx].display_path]["mmr_rank"] = i + 1

        return [self.results[i] for i in selected_indices]