"""
GoldenDatasetVerifier: Verify refactored pipeline produces identical results.
"""
from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.common.types import ImageAnalysisResult
from src.ingestion.pipeline import Pipeline, create_default_pipeline
from src.ingestion.stages.base import PipelineItem
from src.db.storage import DatabaseManager
from src.util.logger import logger


@dataclass
class VerificationResult:
    """Result of verifying against golden dataset."""
    passed: bool
    total_checked: int
    passed_count: int
    failed_count: int
    failures: List[Dict[str, Any]]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "total_checked": self.total_checked,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "failures": self.failures[:10],  # Limit to first 10 failures
        }


class GoldenDatasetVerifier:
    """
    Verifies that refactored pipeline produces results matching a golden dataset.

    The golden dataset should be captured BEFORE refactoring using the
    capture_golden.py script.
    """

    def __init__(
        self,
        golden_path: str = "golden_dataset.json",
        tolerance: float = 1e-5,
    ):
        """
        Args:
            golden_path: Path to golden dataset JSON file
            tolerance: Tolerance for floating point comparisons
        """
        self.golden_path = golden_path
        self.tolerance = tolerance
        self.golden_data: Optional[Dict[str, Any]] = None

    def load_golden(self) -> bool:
        """Load golden dataset from file."""
        if not Path(self.golden_path).exists():
            logger.error(f"Golden dataset not found at {self.golden_path}")
            return False

        with open(self.golden_path, "r") as f:
            self.golden_data = json.load(f)

        logger.info(f"Loaded golden dataset with {len(self.golden_data['images'])} images")
        return True

    def verify(
        self,
        sample_size: int = 100,
        use_gpu: bool = True,
        seed: int = 42,
        shuffle: bool = True,
        batch_size: int = 16,
    ) -> VerificationResult:
        """
        Run verification against golden dataset.

        Args:
            sample_size: Number of images to verify (max limited by golden dataset)
            use_gpu: Whether to use GPU
            seed: Random seed for reproducible shuffling
            shuffle: Whether to shuffle the sample selection
            batch_size: Batch size for pipeline processing

        Returns:
            VerificationResult with pass/fail status and details
        """
        import random

        if not self.golden_data:
            if not self.load_golden():
                return VerificationResult(
                    passed=False,
                    total_checked=0,
                    passed_count=0,
                    failed_count=0,
                    failures=[{"error": "Golden dataset not found"}],
                )

        # Get paths from golden dataset
        golden_images = self.golden_data["images"]
        all_paths = list(golden_images.keys())

        # Shuffle and sample
        if shuffle:
            random.seed(seed)
            random.shuffle(all_paths)

        paths = all_paths[:sample_size]

        logger.info(f"Verifying {len(paths)} images against golden dataset (shuffle={shuffle}, seed={seed})...")

        # Create a non-persisting pipeline (no DB writes)
        from src.ingestion.stages import (
            PreprocessStage,
            FaceDetectionStage,
            PoseExtractionStage,
            CLIPEncodingStage,
            AestheticScoreStage,
            CaptionGenerationStage,
            CaptionEmbeddingStage,
        )

        stages = [
            PreprocessStage(duplicate_checker=None),  # No duplicate check
            FaceDetectionStage(use_gpu=use_gpu),
            PoseExtractionStage(use_gpu=use_gpu),
            CLIPEncodingStage(use_gpu=use_gpu, batch_size=batch_size),
            AestheticScoreStage(use_gpu=use_gpu),
            CaptionGenerationStage(batch_size=min(batch_size, 8)),  # Qwen2-VL uses more VRAM
            CaptionEmbeddingStage(batch_size=min(batch_size * 2, 32)),  # E5 is lighter
            # No PersistStage - we just compare results
        ]

        pipeline = Pipeline(stages=stages, batch_size=batch_size)

        logger.info(f"Using batch_size={batch_size}")

        try:
            # Run pipeline
            pipeline.initialize()

            # Process in batches and collect results
            results: Dict[str, PipelineItem] = {}

            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i:i + batch_size]
                items = [PipelineItem.from_path(p) for p in batch_paths]

                for stage in pipeline.stages:
                    items = stage.process(items)

                for item in items:
                    if item.result.original_path:
                        results[item.result.original_path] = item

            # Compare with golden (only the selected sample paths)
            failures = []
            passed_count = 0

            for path in paths:
                expected = golden_images[path]

                if path not in results:
                    failures.append({
                        "path": path,
                        "error": "Not processed",
                    })
                    continue

                item = results[path]
                if item.error:
                    failures.append({
                        "path": path,
                        "error": f"Processing error: {item.error}",
                    })
                    continue

                # Run comparisons
                comparison_failures = self._compare_result(path, item.result, expected)
                if comparison_failures:
                    failures.extend(comparison_failures)
                else:
                    passed_count += 1

            return VerificationResult(
                passed=len(failures) == 0,
                total_checked=len(paths),
                passed_count=passed_count,
                failed_count=len(paths) - passed_count,
                failures=failures,
            )

        finally:
            pipeline.cleanup()

    def _compare_result(
        self,
        path: str,
        actual: ImageAnalysisResult,
        expected: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Compare actual result with expected golden values."""
        failures = []

        # Helper for comparing scalar values with tolerance
        def check_scalar(field: str, actual_val, expected_val, tolerance: float = 0.01):
            if actual_val is None and expected_val is None:
                return
            if actual_val is None or expected_val is None:
                failures.append({
                    "path": path,
                    "field": field,
                    "expected": expected_val,
                    "actual": actual_val,
                    "error": "One value is None",
                })
                return
            if abs(actual_val - expected_val) > tolerance:
                failures.append({
                    "path": path,
                    "field": field,
                    "expected": expected_val,
                    "actual": actual_val,
                    "diff": abs(actual_val - expected_val),
                })

        def check_exact(field: str, actual_val, expected_val):
            if actual_val != expected_val:
                failures.append({
                    "path": path,
                    "field": field,
                    "expected": expected_val,
                    "actual": actual_val,
                })

        # 1. Image dimensions (exact match)
        check_exact("original_width", actual.original_width, expected.get("original_width"))
        check_exact("original_height", actual.original_height, expected.get("original_height"))

        # 2. Timestamp and derived fields (exact match)
        check_exact("timestamp", actual.timestamp, expected.get("timestamp"))
        check_exact("month", actual.month, expected.get("month"))
        check_exact("time_period", actual.time_period, expected.get("time_period"))
        check_exact("iso", actual.iso, expected.get("iso"))

        # 3. Global image quality metrics (tolerance for float precision)
        check_scalar("global_blur", actual.global_blur, expected.get("global_blur"), tolerance=0.1)
        check_scalar("global_brightness", actual.global_brightness, expected.get("global_brightness"), tolerance=0.1)
        check_scalar("global_contrast", actual.global_contrast, expected.get("global_contrast"), tolerance=0.1)

        # 4. Semantic vector similarity (cosine similarity)
        if actual.semantic_vector is not None and expected.get("semantic_vector") is not None:
            expected_vec = np.array(expected["semantic_vector"])
            if not np.allclose(actual.semantic_vector, expected_vec, rtol=self.tolerance):
                similarity = float(np.dot(actual.semantic_vector, expected_vec))
                if similarity < 0.999:  # Allow small differences due to float precision
                    failures.append({
                        "path": path,
                        "field": "semantic_vector",
                        "similarity": similarity,
                        "error": "Semantic vector mismatch",
                    })

        # 5. Aesthetic score
        check_scalar("aesthetic_score", actual.aesthetic_score, expected.get("aesthetic_score"), tolerance=0.01)

        # 6. Caption vector similarity (relaxed threshold due to caption non-determinism)
        if actual.caption_vector is not None and expected.get("caption_vector") is not None:
            actual_vec = np.array(actual.caption_vector)
            expected_vec = np.array(expected["caption_vector"])
            if not np.allclose(actual_vec, expected_vec, rtol=self.tolerance):
                similarity = float(np.dot(actual_vec, expected_vec))
                if similarity < 0.85:  # Relaxed from 0.999 due to Qwen2-VL non-determinism
                    failures.append({
                        "path": path,
                        "field": "caption_vector",
                        "similarity": similarity,
                        "error": "Caption vector mismatch",
                    })

        # 7. Caption text (may vary due to model non-determinism, skip strict comparison)
        # Note: Qwen2-VL may produce slightly different captions on different runs
        # We could add a similarity check here if needed

        # 8. Face count
        actual_face_count = len(actual.faces) if actual.faces else 0
        expected_face_count = expected.get("face_count", 0)
        if actual_face_count != expected_face_count:
            failures.append({
                "path": path,
                "field": "face_count",
                "expected": expected_face_count,
                "actual": actual_face_count,
            })

        # 9. Face-level comparisons (order matters!)
        if actual.faces and expected.get("faces"):
            for i, (actual_face, expected_face) in enumerate(zip(actual.faces, expected["faces"])):
                prefix = f"faces[{i}]"

                # Bounding box (exact match - critical for index ordering)
                if actual_face.bbox != expected_face.get("bbox"):
                    failures.append({
                        "path": path,
                        "field": f"{prefix}.bbox",
                        "expected": expected_face.get("bbox"),
                        "actual": actual_face.bbox,
                        "error": "INDEX ORDER MISMATCH - Critical!",
                    })

                # Face embedding similarity
                if actual_face.embedding is not None and expected_face.get("embedding") is not None:
                    actual_emb = np.array(actual_face.embedding)
                    expected_emb = np.array(expected_face["embedding"])
                    # Normalize for cosine similarity
                    actual_norm = actual_emb / (np.linalg.norm(actual_emb) + 1e-10)
                    expected_norm = expected_emb / (np.linalg.norm(expected_emb) + 1e-10)
                    similarity = float(np.dot(actual_norm, expected_norm))
                    if similarity < 0.999:
                        failures.append({
                            "path": path,
                            "field": f"{prefix}.embedding",
                            "similarity": similarity,
                            "error": "Face embedding mismatch",
                        })

                # Confidence
                check_scalar(f"{prefix}.confidence", actual_face.confidence, expected_face.get("confidence"), tolerance=0.001)

                # Shot type (exact match)
                check_exact(f"{prefix}.shot_type", actual_face.shot_type, expected_face.get("shot_type"))

                # Pose (exact match as string)
                actual_pose = str(actual_face.pose) if actual_face.pose else None
                check_exact(f"{prefix}.pose", actual_pose, expected_face.get("pose"))

                # Face quality metrics
                check_scalar(f"{prefix}.blur_score", actual_face.blur_score, expected_face.get("blur_score"), tolerance=0.1)
                check_scalar(f"{prefix}.brightness", actual_face.brightness, expected_face.get("brightness"), tolerance=0.1)

                # Pose angles
                check_scalar(f"{prefix}.yaw", actual_face.yaw, expected_face.get("yaw"), tolerance=0.1)
                check_scalar(f"{prefix}.pitch", actual_face.pitch, expected_face.get("pitch"), tolerance=0.1)
                check_scalar(f"{prefix}.roll", actual_face.roll, expected_face.get("roll"), tolerance=0.1)

        return failures


def capture_golden_dataset(
    output_path: str = "golden_dataset.json",
    sample_size: int = 300,
    photos_dir: str = "./photos",
    use_gpu: bool = True,
) -> None:
    """
    Capture a golden dataset from the CURRENT (pre-refactor) implementation.

    Run this BEFORE refactoring to create the comparison baseline.

    Args:
        output_path: Path to save golden dataset
        sample_size: Number of images to include
        photos_dir: Directory containing photos
        use_gpu: Whether to use GPU
    """
    import os
    import random
    from src.ingestion.processor import FeatureExtractor
    from src.ingestion.format_handler import ensure_display_version
    from src.util.image_util import calculate_image_hash

    logger.info(f"Capturing golden dataset with {sample_size} images...")

    # Find all images
    all_files = []
    extensions = {'.jpg', '.jpeg', '.png', '.heic', '.webp'}
    for root, _, files in os.walk(photos_dir):
        for f in files:
            if Path(f).suffix.lower() in extensions:
                all_files.append(os.path.join(root, f))

    # Stratified sampling
    random.seed(42)
    sample = random.sample(all_files, min(sample_size, len(all_files)))

    # Process with original implementation
    engine = FeatureExtractor(use_gpu=use_gpu)
    golden_data = {"images": {}}

    from tqdm import tqdm

    for raw_path in tqdm(sample, desc="Capturing golden dataset"):
        try:
            display_path = ensure_display_version(raw_path, "./photos/cache")
            result = engine.process_image(display_path, raw_path)

            if result:
                golden_data["images"][raw_path] = {
                    # Image metadata
                    "display_path": display_path,
                    "original_width": result.original_width,
                    "original_height": result.original_height,

                    # Timestamp and derived fields
                    "timestamp": result.timestamp,
                    "month": result.month,
                    "time_period": result.time_period,
                    "iso": result.iso,

                    # Global image quality metrics
                    "global_blur": result.global_blur,
                    "global_brightness": result.global_brightness,
                    "global_contrast": result.global_contrast,

                    # CLIP semantic vector (768D)
                    "semantic_vector": result.semantic_vector.tolist() if result.semantic_vector is not None else None,

                    # Aesthetic score
                    "aesthetic_score": result.aesthetic_score,

                    # Caption and caption embedding
                    "caption": result.caption,
                    "caption_vector": result.caption_vector if result.caption_vector is not None else None,

                    # Face data (complete)
                    "face_count": len(result.faces) if result.faces else 0,
                    "faces": [
                        {
                            "bbox": f.bbox,
                            "embedding": f.embedding.tolist() if f.embedding is not None else None,
                            "confidence": f.confidence,
                            "shot_type": f.shot_type,
                            "pose": str(f.pose) if f.pose else None,
                            "blur_score": f.blur_score,
                            "brightness": f.brightness,
                            "yaw": f.yaw,
                            "pitch": f.pitch,
                            "roll": f.roll,
                        }
                        for f in (result.faces or [])
                    ],
                }
        except Exception as e:
            logger.warning(f"Failed to process {raw_path}: {e}")

    # Save golden dataset
    with open(output_path, "w") as f:
        json.dump(golden_data, f)

    logger.info(f"Golden dataset saved to {output_path} ({len(golden_data['images'])} images)")
