import pytest
import numpy as np
import sqlite3
import json
from src.db.storage import DatabaseManager
from src.ingestion.processor import ImageAnalysisResult, FaceData


@pytest.fixture
def db_manager(tmp_path):
    # Use :memory: for SQLite speed, and tmp_path for Chroma Isolation
    chroma_dir = tmp_path / "chroma_db"
    manager = DatabaseManager(sql_path=":memory:", chroma_path=str(chroma_dir))
    yield manager
    manager.close()


@pytest.fixture
def sample_data():
    return ImageAnalysisResult(
        semantic_vector=np.random.rand(512).astype(np.float32),
        faces=[
            FaceData(
                bbox=[0, 0, 100, 100],
                embedding=np.random.rand(512).astype(np.float32),
                confidence=0.95
                # Note: Add 'det_score' here if your FaceData class requires it
            )
        ],
        original_width=1920,
        original_height=1080
    )


def test_init_creates_tables(db_manager):
    """
    Check if tables exist after init.
    """
    cursor = db_manager.cursor
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    assert "photos" in tables
    assert "people" in tables
    assert "photo_faces" in tables


def test_save_result_inserts_data(db_manager, sample_data):
    # UPDATED: Added 'dummy_hash_1' argument
    pid = db_manager.save_result(sample_data, "/orig/a.jpg", "/disp/a.jpg", "dummy_hash_1")

    # 1. Verify Photo Record
    db_manager.cursor.execute("SELECT * from photos WHERE photo_id=?", (pid,))
    photo_row = db_manager.cursor.fetchone()
    assert photo_row is not None
    assert photo_row[1] == "/orig/a.jpg"  # original_path

    # Verify the hash was stored (assuming it's column index 3 or by name)
    db_manager.cursor.execute("SELECT file_hash FROM photos WHERE photo_id=?", (pid,))
    assert db_manager.cursor.fetchone()[0] == "dummy_hash_1"

    # 2. Verify Face Record
    db_manager.cursor.execute("SELECT * from photo_faces WHERE photo_id=?", (pid,))
    face_row = db_manager.cursor.fetchone()
    assert face_row is not None
    assert face_row[2] == -1  # person_id default
    assert isinstance(face_row[3], bytes)  # embedding blob

    # 3. Verify Chroma Vector
    vec_data = db_manager.vector_collection.get(ids=[pid])
    assert len(vec_data['ids']) == 1
    assert vec_data['metadatas'][0]['path'] == "/disp/a.jpg"


def test_cascade_delete(db_manager, sample_data):
    """
    Test that deleting a photo deletes its faces.
    """
    # UPDATED: Added 'dummy_hash_2'
    pid = db_manager.save_result(sample_data, "p1", "d1", "dummy_hash_2")

    db_manager.cursor.execute("SELECT count(*) FROM photo_faces")
    assert db_manager.cursor.fetchone()[0] == 1

    # Delete the photo
    db_manager.cursor.execute("DELETE FROM photos WHERE photo_id=?", (pid,))
    db_manager.conn.commit()

    # Verify face is gone
    db_manager.cursor.execute("SELECT count(*) FROM photo_faces")
    assert db_manager.cursor.fetchone()[0] == 0


def test_duplicate_hash_prevention(db_manager, sample_data):
    """
    NEW TEST: Verifies that saving the same hash twice triggers an error.
    """
    unique_hash = "shared_hash_123"

    # First save: Should succeed
    db_manager.save_result(sample_data, "path/A.jpg", "disp/A.jpg", unique_hash)

    # Second save: Different path, same hash -> Should Fail
    # This proves your UNIQUE constraint is working
    with pytest.raises(sqlite3.IntegrityError):
        db_manager.save_result(sample_data, "path/B.jpg", "disp/B.jpg", unique_hash)