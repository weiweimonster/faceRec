import math
from typing import List, Dict, Any, Tuple, Optional
from src.common.types import ImageAnalysisResult, FaceData
from src.pose.pose import Pose
from .base import BaseRankingStrategy, RankingResult
from src.util.image_util import parse_date_components
from src.features.registry import get_normalization_bounds
from src.features.container import FeatureExtractor
from src.util.logger import logger

class HeuristicStrategy(BaseRankingStrategy):

    WEIGHTS = {
        "semantic": 0.6,
        "caption": 0.25,
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
            caption_scores: Dict[str, float],
            target_name: Optional[str] = None,
            pose: Optional[Pose] = None
    ) -> RankingResult:

        scored_items: List[Tuple[ImageAnalysisResult, float]] = []
        display_metrics: Dict[str, Dict[str, Any]] = {}
        training_features: Dict[str, Dict[str, float]] = {}

        # Iterate though all the result and calculate metrics
        for item in results:
            path = item.display_path

            # get the semantic score
            semantic_sim = semantic_scores[path]
            caption_sim = caption_scores.get(path, 0.0)

            # Calculate the global quality scores
            global_q, g_display_metrics = self._calculate_global_quality(item)
            face_q, f_display_metrics = 0.0, {}

            if target_name:
                face_q, f_display_metrics = self._calculate_face_quality(item, target_name, pose)

            w = self.WEIGHTS

            # Combine semantic score, caption score, and global score
            final_score = (
                (semantic_sim * w["semantic"]) +
                (caption_sim * w["caption"]) +
                (global_q * w["global_quality"])
            )

            # If this is person related, then combine face quality score
            if target_name:
                final_score += (face_q * w["face_quality"])

            display_metrics[path]: Dict[str, Any] = {
                "final_relevance": round(final_score, 4),
                "semantic": round(semantic_sim, 3),
                "caption": round(caption_sim, 3),
                **g_display_metrics,
                **f_display_metrics
            }

            context = {
                "semantic_score": semantic_sim,
                "caption_score": caption_sim
            }
            training_features[path] = FeatureExtractor.extract_from_result(
                result=item,
                context=context,
                target_face_name=target_name
            )

            scored_items.append((item, final_score))

        scored_items.sort(key=lambda x: x[1], reverse=True)
        ranked_results = [item for item, score in scored_items]

        logger.info(f"Heuristic ranker scored {len(ranked_results)} candidates")

        return RankingResult(
            ranked_results=ranked_results,
            display_metrics=display_metrics,
            training_features=training_features
        )

    def _normalize(self, feature_name: str, value: float) -> float:
        """Linear normalization to 0.0 - 1.0"""
        if value is None:
            return 0.0

        bounds = get_normalization_bounds(feature_name)
        if not bounds:
            logger.warning(f"No normalization bounds for {feature_name}. Retuning 0.0")
            return 0.0

        min_v, max_v = bounds["min"], bounds["max"]

        if value < min_v:
            return 0.0
        if value > max_v:
            return 1.0
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

        bounds = get_normalization_bounds("g_iso")
        min_log = math.log2(bounds["min"])
        max_log = math.log2(bounds["max"])
        curr_log = math.log2(iso_val)
        norm = (curr_log - min_log) / (max_log - min_log)

        # Invert because high ISO = worse quality
        score = 1.0 - max(0.0, min(1.0, norm))
        return score

    def _calculate_global_quality(self, item: ImageAnalysisResult) ->  Tuple[float, Dict[str, float]]:
        """
        Scores the technical quality of the whole image (Scenery, Composition, Noise).
        """

        wts = self.WEIGHTS

        # 1. Aesthetics (The "Beauty" AI Score)
        s_aes = self._normalize("aesthetic_score", item.aesthetic_score)
        s_blur = self._normalize("g_blur", item.global_blur)
        s_iso = self._calculate_iso_score(item.iso)
        s_con = self._normalize("g_contrast", item.global_contrast)

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

    def _calculate_face_quality(
            self,
            item: ImageAnalysisResult,
            target_name: str,
            requested_pose: Optional[Pose] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate face quality score for target person.

        Returns:
            (combined_score, display_metrics_dict)
        """
        if not item.faces:
            logger.error(f"Provided picture doesn't have any faces. Returning zeros")
            return 0.0, {}

        # Find target face by name
        target_face = next((f for f in item.faces if f.name.lower() == target_name.lower()), None)

        if not target_face:
            logger.error(f"No face found with name '{target_name}'. Returning zeros")
            return 0.0, {}

        if target_face.bbox and len(target_face.bbox) == 4:
            width_px = target_face.bbox[2] - target_face.bbox[0]
            height_px = target_face.bbox[3] - target_face.bbox[1]
        else:
            width_px, height_px = 0, 0

        # Normalize using registry bounds
        s_blur = self._normalize("f_blur", target_face.blur_score)
        h_score = self._normalize("f_height", height_px)
        w_score = self._normalize("f_width", width_px)
        s_size = (h_score + w_score) / 2
        s_orient = self._calculate_orientation_score(target_face, requested_pose)

        # Weighted combination
        score = (s_blur * 0.4) + (s_size * 0.3) + (s_orient * 0.3)

        # Display metrics
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


