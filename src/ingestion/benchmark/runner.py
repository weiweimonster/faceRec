"""
BenchmarkRunner: Compare sequential vs parallel pipeline performance.
"""
from __future__ import annotations

import os
import json
import time
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path

from src.ingestion.pipeline import Pipeline, PipelineStats, create_default_pipeline
from src.db.storage import DatabaseManager
from src.util.logger import logger


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    mode: str  # "sequential" or "parallel"
    batch_size: int
    sample_size: int
    stats: PipelineStats

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "batch_size": self.batch_size,
            "sample_size": self.sample_size,
            **self.stats.to_dict(),
        }


class BenchmarkRunner:
    """
    Runs benchmarks comparing sequential vs parallel pipeline performance.

    Uses a separate test database to avoid affecting production data.
    """

    def __init__(
        self,
        photos_dir: str = "./photos",
        test_db_dir: str = ".db/benchmark",
        use_gpu: bool = True,
    ):
        """
        Args:
            photos_dir: Directory containing photos to benchmark
            test_db_dir: Directory for test database (will be created fresh)
            use_gpu: Whether to use GPU for inference
        """
        self.photos_dir = photos_dir
        self.test_db_dir = test_db_dir
        self.use_gpu = use_gpu

    def select_sample(
        self,
        size: int,
        stratified: bool = True,
        seed: Optional[int] = None,
    ) -> List[str]:
        """
        Select a sample of images for benchmarking.

        Args:
            size: Number of images to select
            stratified: Whether to stratify by file extension
            seed: Random seed for reproducibility

        Returns:
            List of file paths
        """
        if seed is not None:
            random.seed(seed)

        # Find all image files
        all_files: List[str] = []
        extensions = {'.jpg', '.jpeg', '.png', '.heic', '.webp'}

        for root, _, files in os.walk(self.photos_dir):
            for f in files:
                if Path(f).suffix.lower() in extensions:
                    all_files.append(os.path.join(root, f))

        if len(all_files) < size:
            logger.warning(f"Only {len(all_files)} images available, requested {size}")
            return all_files

        if not stratified:
            return random.sample(all_files, size)

        # Stratified sampling by extension
        by_ext: Dict[str, List[str]] = {}
        for f in all_files:
            ext = Path(f).suffix.lower()
            by_ext.setdefault(ext, []).append(f)

        # Calculate samples per extension
        sample = []
        per_ext = size // len(by_ext)
        remainder = size % len(by_ext)

        for i, (ext, files) in enumerate(by_ext.items()):
            n = per_ext + (1 if i < remainder else 0)
            n = min(n, len(files))
            sample.extend(random.sample(files, n))

        # Fill remaining if needed
        if len(sample) < size:
            remaining = [f for f in all_files if f not in sample]
            sample.extend(random.sample(remaining, min(size - len(sample), len(remaining))))

        return sample[:size]

    def _create_test_db(self, run_id: str = None) -> DatabaseManager:
        """Create a fresh test database with unique path to avoid ChromaDB caching issues."""
        import shutil
        import uuid

        # Use unique subdirectory for each run to avoid ChromaDB 1.4.0 caching issues
        if run_id is None:
            run_id = str(uuid.uuid4())[:8]

        run_dir = os.path.join(self.test_db_dir, run_id)
        sql_path = os.path.join(run_dir, "sqlite/test.db")
        chroma_path = os.path.join(run_dir, "chroma")

        # Clean up if exists
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)

        # Create fresh directories
        os.makedirs(os.path.dirname(sql_path), exist_ok=True)
        os.makedirs(chroma_path, exist_ok=True)

        return DatabaseManager(sql_path=sql_path, chroma_path=chroma_path)

    def run_single_benchmark(
        self,
        sample: List[str],
        batch_size: int,
    ) -> BenchmarkResult:
        """
        Run a single benchmark with the given batch size.

        Args:
            sample: List of file paths to process
            batch_size: Batch size (1=sequential, >1=parallel)

        Returns:
            BenchmarkResult with timing statistics
        """
        mode = "sequential" if batch_size == 1 else "parallel"
        logger.info(f"Running {mode} benchmark (batch_size={batch_size}, samples={len(sample)})...")

        # Create fresh test DB for this run with unique ID to avoid ChromaDB caching issues
        run_id = f"{mode}_{batch_size}"
        db = self._create_test_db(run_id=run_id)

        try:
            pipeline = create_default_pipeline(
                db=db,
                batch_size=batch_size,
                use_gpu=self.use_gpu,
            )

            stats = pipeline.run(sample)

            return BenchmarkResult(
                mode=mode,
                batch_size=batch_size,
                sample_size=len(sample),
                stats=stats,
            )

        finally:
            db.close()
            try:
                pipeline.cleanup()
            except:
                pass

    def run_benchmark(
        self,
        sample_sizes: List[int] = [50, 100, 200],
        parallel_batch_size: int = 16,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Run full benchmark comparing sequential vs parallel.

        Args:
            sample_sizes: List of sample sizes to test
            parallel_batch_size: Batch size for parallel mode
            seed: Random seed for reproducibility

        Returns:
            Dict with benchmark results
        """
        results = {}

        for size in sample_sizes:
            logger.info(f"\n{'='*50}")
            logger.info(f"Benchmark with {size} images")
            logger.info('='*50)

            sample = self.select_sample(size, stratified=True, seed=seed)

            # Sequential (batch_size=1)
            sequential_result = self.run_single_benchmark(sample, batch_size=1)

            # Parallel (batch_size=parallel_batch_size)
            parallel_result = self.run_single_benchmark(sample, batch_size=parallel_batch_size)

            # Calculate speedup
            speedup = 1.0
            if sequential_result.stats.images_per_second > 0:
                speedup = parallel_result.stats.images_per_second / sequential_result.stats.images_per_second

            results[size] = {
                "sequential": sequential_result.to_dict(),
                "parallel": parallel_result.to_dict(),
                "speedup": round(speedup, 2),
            }

            logger.info(f"\nResults for {size} images:")
            logger.info(f"  Sequential: {sequential_result.stats.images_per_second:.2f} img/s")
            logger.info(f"  Parallel:   {parallel_result.stats.images_per_second:.2f} img/s")
            logger.info(f"  Speedup:    {speedup:.2f}x")

        # Create summary
        if results:
            avg_speedup = sum(r["speedup"] for r in results.values()) / len(results)
            # Use largest sample size for summary stats
            largest = max(results.keys())
            summary = {
                "speedup_factor": round(avg_speedup, 2),
                "sequential_imgs_per_sec": results[largest]["sequential"]["images_per_second"],
                "parallel_imgs_per_sec": results[largest]["parallel"]["images_per_second"],
            }
        else:
            summary = {}

        return {
            "summary": summary,
            "results_by_sample_size": results,
        }

    def cleanup(self) -> None:
        """Remove the test database directory after benchmark completes."""
        import shutil

        if os.path.exists(self.test_db_dir):
            shutil.rmtree(self.test_db_dir)
            logger.info(f"Cleaned up test database at {self.test_db_dir}")

    def save_report(self, results: Dict[str, Any], output_path: str = "benchmark_report.json"):
        """Save benchmark results to JSON file."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Benchmark report saved to {output_path}")
