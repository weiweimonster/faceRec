import math
from typing import List, Dict, Any, Tuple, Optional
from src.common.types import ImageAnalysisResult, FaceData
from src.pose.pose import Pose
from .base import BaseRankingStrategy, ScoredCandidate

class HeuristicStrategy(BaseRankingStrategy):
    # Tunable Hyperparameters (Moved from original class)
    QUALITY_BOUNDS = {
        "face_blur": {"min": 30, "max": 1200},
        "face_size": {"min": 50, "max": 550},
        "global_blur": {"min": 21.0, "max": 1400.0},
        "iso": {"min": 40, "max": 1600},
        "contrast": {"min": 35.0, "max": 80.0},
        "aesthetic": {"min": 4.0, "max": 5.5}
    }

    WEIGHTS = {
        "semantic": 0.85,
        "face_quality": 0.1,
        "global_quality": 0.05,
        "w_aesthetic": 0.4,
        "w_sharpness": 0.3,
        "w_iso": 0.2,
        "w_contrast": 0.1
    }

    def score_candidates(
            self,
            results: List[ImageAnalysisResult],
            semantic_scores: Dict[str, float],
            target_name: Optional[str] = None,
            pose: Optional[Pose] = None
    ) -> List[ScoredCandidate]:

        # Initialize empty results
        scored_items: List[ScoredCandidate] = []

        # Iterate though all the result and calculate metrics
        for item in results:
            path = item.display_path

            # get the semantic score
            semantic_sim = semantic_scores[path]

            # Calculate the global quality scores
            global_q, g_metrics = self._calculate_global_quality(item)

            face_q: float = 0.0
            f_metrics: Dict[str, float] = {}

            if target_name:
                # Calculate the face quality score
                face_q, f_metrics = self._calculate_face_quality(item, target_name, pose)

            w = self.WEIGHTS

            # Combine semantic score and global score
            final_score = (semantic_sim * w["semantic"]) + (global_q * w["global_quality"])

            # If this is person related, then combine face quality score
            if target_name:
                final_score += (face_q * w["face_quality"])

            metrics: Dict[str, Any] = {
                "final_relevance": round(final_score, 4),
                "semantic": round(semantic_sim, 3),
                **g_metrics,
                **f_metrics
            }

            scored_items.append((item, final_score, metrics))

        return scored_items

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


