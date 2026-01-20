"""
PreprocessStage: Hash calculation, duplicate check, HEIC conversion, image loading, EXIF extraction.

This is a CPU-bound stage that uses ThreadPoolExecutor for parallel I/O operations.
"""
from __future__ import annotations

from typing import List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
from PIL import Image

from src.ingestion.stages.base import ProcessingStage, PipelineItem
from src.ingestion.format_handler import ensure_display_version
from src.util.image_util import (
    calculate_image_hash,
    compute_global_visual_stats,
    get_exif_timestamp,
    get_exif_iso,
    get_disk_timestamp,
    extract_time_features,
)
from src.util.logger import logger


class PreprocessStage(ProcessingStage):
    """
    First stage of the pipeline: prepares images for processing.

    Responsibilities:
    - Calculate file hash for deduplication
    - Check if image already exists in database
    - Convert HEIC to JPG if needed
    - Load image as OpenCV (BGR) and PIL (RGB)
    - Extract global visual stats (blur, brightness, contrast)
    - Extract EXIF metadata (timestamp, ISO)
    """

    def __init__(
        self,
        cache_dir: str = "./photos/cache",
        duplicate_checker: Optional[Callable[[str], bool]] = None,
        max_workers: int = 4,
    ):
        """
        Args:
            cache_dir: Directory to store converted HEIC files
            duplicate_checker: Function that takes a file hash and returns True if duplicate
            max_workers: Number of threads for parallel I/O
        """
        self.cache_dir = cache_dir
        self.duplicate_checker = duplicate_checker
        self.max_workers = max_workers

    @property
    def name(self) -> str:
        return "preprocess"

    @property
    def supports_gpu_batching(self) -> bool:
        return False  # CPU-bound I/O operations

    def process(self, items: List[PipelineItem]) -> List[PipelineItem]:
        """Process items in parallel using thread pool."""
        if not items:
            return items

        # Process in parallel for I/O bound operations
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_single, item): i
                for i, item in enumerate(items)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    # Result is updated in-place, but catch any exceptions
                    future.result()
                except Exception as e:
                    items[idx].mark_error(f"Preprocess error: {e}")
                    logger.error(f"Preprocess failed for {items[idx].result.original_path}: {e}")

        return items

    def _process_single(self, item: PipelineItem) -> None:
        """Process a single item (runs in thread pool)."""
        if not item.is_processable():
            return

        raw_path = item.result.original_path

        try:
            # 1. Calculate hash
            item.file_hash = calculate_image_hash(raw_path)
        except Exception as e:
            item.mark_error(f"Hash calculation failed: {e}")
            return

        # 2. Check for duplicates
        if self.duplicate_checker and self.duplicate_checker(item.file_hash):
            item.mark_skipped("Duplicate image")
            logger.debug(f"Skipping duplicate: {raw_path}")
            return

        try:
            # 3. Convert HEIC if needed, get display path
            display_path = ensure_display_version(raw_path, self.cache_dir)
            item.result.display_path = display_path
        except Exception as e:
            item.mark_error(f"Format conversion failed: {e}")
            return

        try:
            # 4. Load image with OpenCV
            cv_img = cv2.imread(display_path)
            if cv_img is None:
                item.mark_error(f"Failed to load image: {display_path}")
                return

            item.cv_image = cv_img
            h, w = cv_img.shape[:2]
            item.result.original_width = w
            item.result.original_height = h

            # 5. Compute global visual stats
            global_stats = compute_global_visual_stats(cv_img)
            item.result.global_blur = global_stats["global_blur"]
            item.result.global_brightness = global_stats["global_brightness"]
            item.result.global_contrast = global_stats["global_contrast"]

        except Exception as e:
            item.mark_error(f"Image loading failed: {e}")
            return

        try:
            # 6. Load as PIL for CLIP (will be used later)
            item.pil_image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.warning(f"PIL conversion failed for {raw_path}: {e}")
            # Not fatal - CLIP stage will handle this

        # 7. Extract EXIF data
        try:
            timestamp = get_exif_timestamp(display_path)
            if not timestamp:
                timestamp = get_disk_timestamp(raw_path)

            item.result.timestamp = timestamp

            if timestamp:
                month, time_period = extract_time_features(timestamp)
                item.result.month = month
                item.result.time_period = time_period

            item.result.iso = get_exif_iso(display_path)

        except Exception as e:
            logger.warning(f"EXIF extraction failed for {raw_path}: {e}")
            # Not fatal - continue without EXIF data
