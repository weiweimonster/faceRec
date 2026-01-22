import os
import argparse
from typing import List
from src.db.storage import DatabaseManager
from src.clustering.service import ClusteringService
from src.util.logger import logger

RAW_PHOTOS_DIR = "./photos"
CACHE_DIR = "./photos/cache"
DB_SQL_PATH = ".db/sqlite/photos.db"
DB_CHROMA_PATH = "./db/chroma"


def run_ingestion(mode: str = "parallel", batch_size: int = None) -> None:
    """
    Scans the raw_photos directory, processes image with AI models
    and saves the embeddings in the database.

    Args:
        mode: "parallel" (batch_size=16) or "sequential" (batch_size=1)
        batch_size: Override batch size (if None, uses mode-based default)
    """
    logger.info(f"Starting Ingestion Pipeline (mode={mode}, batch_size={batch_size})...")

    # Initialize Database
    db = DatabaseManager(sql_path=DB_SQL_PATH, chroma_path=DB_CHROMA_PATH)

    # Find Files
    all_files: List[str] = []
    for dp, _, filenames in os.walk(RAW_PHOTOS_DIR):
        for f in filenames:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic', '.webp')):
                all_files.append(os.path.join(dp, f))

    logger.info(f"Found {len(all_files)} photos.")

    _run_pipeline_ingestion(db, all_files, mode, batch_size)

    db.close()
    logger.info("Ingestion Complete.")


def _run_pipeline_ingestion(db: DatabaseManager, all_files: List[str], mode: str, batch_size: int = None) -> None:
    """New pipeline-based ingestion."""
    from src.ingestion.pipeline import create_default_pipeline

    # Determine batch size: explicit > mode-based default
    if batch_size is None:
        batch_size = 16 if mode == "parallel" else 1
    logger.info(f"Using pipeline with batch_size={batch_size}")

    pipeline = create_default_pipeline(
        db=db,
        batch_size=batch_size,
        use_gpu=True,
        cache_dir=CACHE_DIR,
    )

    try:
        stats = pipeline.run(all_files)

        # Log results
        logger.info(f"\nPipeline Statistics:")
        logger.info(f"  Total images: {stats.total_images}")
        logger.info(f"  Processed: {stats.processed}")
        logger.info(f"  Skipped: {stats.skipped}")
        logger.info(f"  Errors: {stats.errors}")
        logger.info(f"  Total time: {stats.total_time_s:.2f}s")
        logger.info(f"  Speed: {stats.images_per_second:.2f} images/second")

        # Stage breakdown
        if stats.stage_times:
            logger.info("\nStage timing breakdown:")
            for stage, time_s in stats.stage_times.items():
                logger.info(f"  {stage}: {time_s:.2f}s")

    finally:
        pipeline.cleanup()


def run_clustering() -> None:
    """
    Loads all unclustered faces from the database, groups them by identity,
    and updates the database records
    """
    logger.info("Starting Clustering Pipeline...")

    db = DatabaseManager(sql_path=DB_SQL_PATH, chroma_path=DB_CHROMA_PATH)
    service = ClusteringService(db)
    service.run()
    db.close()


def run_benchmark(sample_size: int = 100, batch_size: int = 16) -> None:
    """
    Run benchmark comparing sequential vs parallel ingestion.

    Args:
        sample_size: Number of images to test with
        batch_size: Batch size for parallel mode (default: 16)
    """
    from src.ingestion.benchmark import BenchmarkRunner, BenchmarkReport

    logger.info(f"Starting Benchmark (sample_size={sample_size}, batch_size={batch_size})...")

    runner = BenchmarkRunner(
        photos_dir=RAW_PHOTOS_DIR,
        test_db_dir=".db/benchmark",
        use_gpu=True,
    )

    try:
        results = runner.run_benchmark(
            sample_sizes=[sample_size],
            parallel_batch_size=batch_size,
            seed=42,
        )

        # Save report
        report = BenchmarkReport.from_dict(results)
        report.save("benchmark_report.json")

        # Print text report
        print(report.format_text())
    finally:
        # Clean up test database
        runner.cleanup()


def run_verify(sample_size: int = 100, seed: int = 42, no_shuffle: bool = False, batch_size: int = 16) -> None:
    """
    Verify pipeline results against golden dataset.

    Args:
        sample_size: Number of images to verify (max limited by golden dataset)
        seed: Random seed for reproducible shuffling
        no_shuffle: If True, disable shuffling (take first N images)
        batch_size: Batch size for pipeline processing
    """
    from src.ingestion.benchmark import GoldenDatasetVerifier

    logger.info(f"Running verification (sample_size={sample_size}, seed={seed}, shuffle={not no_shuffle}, batch_size={batch_size})...")

    verifier = GoldenDatasetVerifier(
        golden_path="golden_dataset.json",
        tolerance=1e-5,
    )

    result = verifier.verify(
        sample_size=sample_size,
        use_gpu=True,
        seed=seed,
        shuffle=not no_shuffle,
        batch_size=batch_size,
    )

    print(f"\nVerification Results:")
    print(f"  Status: {'PASSED' if result.passed else 'FAILED'}")
    print(f"  Total checked: {result.total_checked}")
    print(f"  Passed: {result.passed_count}")
    print(f"  Failed: {result.failed_count}")

    if result.failures:
        print(f"\nFailures (first 10):")
        for failure in result.failures[:10]:
            print(f"  - {failure}")


def run_capture_golden(count: int = 300) -> None:
    """
    Capture golden dataset for verification.

    Run this BEFORE refactoring to create the comparison baseline.

    Args:
        count: Number of images to include in golden dataset
    """
    from src.ingestion.benchmark.verification import capture_golden_dataset

    logger.info(f"Capturing golden dataset ({count} images)...")
    capture_golden_dataset(
        output_path="golden_dataset.json",
        sample_size=count,
        photos_dir=RAW_PHOTOS_DIR,
        use_gpu=True,
    )
    logger.info("Golden dataset captured successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Photo RAG Pipeline Manager")
    parser.add_argument(
        "mode",
        choices=["ingest", "cluster", "benchmark", "verify", "capture-golden"],
        help="Select operation mode."
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential processing (batch_size=1) for ingest mode."
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing (batch_size=16) for ingest mode."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=100,
        help="Sample size for benchmark/verify modes (default: 100, max: 200 for verify)."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=300,
        help="Number of images for capture-golden mode (default: 300)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling in verify mode (default: 42)."
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling in verify mode (take first N images)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for pipeline processing (default: 16 for parallel, 1 for sequential)."
    )

    args = parser.parse_args()

    if args.mode == "ingest":
        if args.sequential:
            run_ingestion(mode="sequential", batch_size=args.batch_size)
        else:
            # Default to parallel
            run_ingestion(mode="parallel", batch_size=args.batch_size)
    elif args.mode == "cluster":
        run_clustering()
    elif args.mode == "benchmark":
        batch_size = args.batch_size if args.batch_size else 16
        run_benchmark(sample_size=args.sample, batch_size=batch_size)
    elif args.mode == "verify":
        sample = min(args.sample, 200)  # Max 200 for verify
        batch_size = args.batch_size if args.batch_size else 16
        run_verify(sample_size=sample, seed=args.seed, no_shuffle=args.no_shuffle, batch_size=batch_size)
    elif args.mode == "capture-golden":
        run_capture_golden(count=args.count)
