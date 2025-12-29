import os
import argparse
from tqdm import tqdm
from typing import List
from src.db.storage import DatabaseManager
from src.ingestion.processor import FeatureExtractor
from src.ingestion.format_handler import ensure_display_version
from src.clustering.service import ClusteringService
from src.utils import calculate_image_hash

RAW_PHOTOS_DIR = "./photos"
CACHE_DIR = "./photos/cache"
DB_SQL_PATH = ".db/sqlite/photos.db"
DB_CHROMA_PATH = "./db/chroma"

def run_ingestion() -> None:
    """
    Scans the raw_photos directory, processes image with AI models
    and saves the embeddings in the database.
    """

    print("🚀 Starting Ingestion Pipeline...")

    # Initialize Tools
    # We initialize here to avoid loading heavy models if we are only running clustering
    db = DatabaseManager(sql_path=DB_SQL_PATH, chroma_path=DB_CHROMA_PATH)
    engine = FeatureExtractor(use_gpu=True)

    # Find Files
    all_files: List[str] = []
    for dp, _, filenames in os.walk(RAW_PHOTOS_DIR):
        for f in filenames:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic', '.webp')):
                all_files.append(os.path.join(dp, f))

    print(f"📸 Found {len(all_files)} photos.")

    # Process loop
    for raw_path in tqdm(all_files, desc="Ingesting"):
        try:
            img_hash = calculate_image_hash(raw_path)
        except Exception as e:
            print(f"Error calculating hash for {raw_path}: {e}. Skipping.")
            continue
        try:
            # Optimization: Skip if already in DB
            if db.photo_exists(img_hash):
                print(f"Skipping duplicate: {raw_path}")
                continue

            # Convert/Validate Image (Handle HEIC)
            display_path = ensure_display_version(raw_path, CACHE_DIR)

            # Extract Features
            result = engine.process_image(display_path)

            if result:
                # Save to DB
                db.save_result(result, raw_path, display_path, img_hash)
            else:
                print(f"⚠️  Skipping invalid file: {raw_path}")
        except Exception as e:
            print(f"❌ Error processing {raw_path}: {e}")

    db.close()
    print("✅ Ingestion Complete.")

def run_clustering() -> None:
    """
    Loads all unclustered faces from the database, groups them by identity,
    and updates the database records
    """
    print("🧠 Starting Clustering Pipeline...")

    # We don't need the FeatureExtractor here (saves RAM)
    db = DatabaseManager(sql_path=DB_SQL_PATH, chroma_path=DB_CHROMA_PATH)
    service = ClusteringService(db)
    service.run()
    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Photo RAG Pipeline Manager")
    parser.add_argument(
        "mode",
        choices=["ingest", "cluster"],
        help="Select 'ingest' to process new photos or 'cluster' to organize faces."
    )

    args = parser.parse_args()
    if args.mode == "ingest":
        run_ingestion()
    elif args.mode == "cluster":
        run_clustering()
