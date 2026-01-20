#!/usr/bin/env python3
"""
Capture Golden Dataset Script

Run this BEFORE refactoring to create a verification baseline.

Usage:
    python -m scripts.capture_golden --count 300 --output golden_dataset.json

The golden dataset captures results from the CURRENT implementation,
which can then be used to verify the refactored implementation produces
identical results.
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.benchmark.verification import capture_golden_dataset
from src.util.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="Capture golden dataset for verification"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=300,
        help="Number of images to include (default: 300)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="golden_dataset.json",
        help="Output file path (default: golden_dataset.json)"
    )
    parser.add_argument(
        "--photos-dir",
        type=str,
        default="./photos",
        help="Directory containing photos (default: ./photos)"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU (use CPU only)"
    )

    args = parser.parse_args()

    logger.info(f"Capturing golden dataset with {args.count} images...")
    logger.info(f"Output: {args.output}")
    logger.info(f"Photos directory: {args.photos_dir}")
    logger.info(f"GPU: {'disabled' if args.no_gpu else 'enabled'}")

    capture_golden_dataset(
        output_path=args.output,
        sample_size=args.count,
        photos_dir=args.photos_dir,
        use_gpu=not args.no_gpu,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
