import pytest
import numpy as np
import sqlite3
import json
from src.db.storage import DatabaseManager
from src.common.types import ImageAnalysisResult, FaceData
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_chroma():
    with patch('src.db.storage.chromadb.PersistentClient') as mock_client_cls:
        mock_client_instance = mock_client_cls.return_value
        mock_collection = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection

        yield {
            "client_cls": mock_client_cls,
            "client_instance": mock_client_instance,
            "collection": mock_collection
        }

@pytest.fixture
def db_manager(tmp_path, mock_chroma):
    # Use :memory: for SQLite speed, and tmp_path for Chroma Isolation
    chroma_dir = tmp_path / "chroma_db"
    manager = DatabaseManager(sql_path=":memory:", chroma_path=str(chroma_dir))
    yield manager
    manager.close()


@pytest.fixture
def sample_result():
    return ImageAnalysisResult(
        semantic_vector=np.random.rand(512).astype(np.float32),
        faces=[
            FaceData(
                bbox=[100, 100, 200, 200],
                embedding=np.random.rand(512).astype(np.float32),
                confidence=0.98,
                shot_type="Medium-Shot",
                blur_score=0.5,
                brightness=100.0,
                yaw=0.0,
                pitch=0.0,
                roll=0.0
            )
        ],
        original_width=1920,
        original_height=1080,
        timestamp="2025-01-01 10:00:00"
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


def test_save_result_inserts_data(db_manager, mock_chroma, sample_result):
    pid = db_manager.save_result(sample_result, "/orig/a.jpg", "/disp/a.jpg", "dummy_hash_1")

    assert isinstance(pid, str)
    assert len(pid) > 0

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
    mock_chroma["collection"].add.assert_called_once()
    call_args = mock_chroma["collection"].add.call_args[1]

    assert call_args["ids"] == [pid]
    assert len(call_args["embeddings"][0]) == 512
    assert call_args["metadatas"][0]["path"] == "/disp/a.jpg"


def test_cascade_delete(db_manager, sample_result):
    """
    Test that deleting a photo deletes its faces.
    """
    # UPDATED: Added 'dummy_hash_2'
    pid = db_manager.save_result(sample_result, "p1", "d1", "dummy_hash_2")

    db_manager.cursor.execute("SELECT count(*) FROM photo_faces")
    assert db_manager.cursor.fetchone()[0] == 1

    # Delete the photo
    db_manager.cursor.execute("DELETE FROM photos WHERE photo_id=?", (pid,))
    db_manager.conn.commit()

    # Verify face is gone
    db_manager.cursor.execute("SELECT count(*) FROM photo_faces")
    assert db_manager.cursor.fetchone()[0] == 0

def test_save_result_rollback_on_error(db_manager, mock_chroma, sample_result):
    """
    Test Transaction Atomicity.
    If ChromaDB fails, the SQLite record should be removed (rolled back).
    """
    # Force Chroma to raise an error
    mock_chroma["collection"].add.side_effect = Exception("Chroma Connection Error")

    # Attempt to save - should raise exception
    with pytest.raises(Exception) as exc:
        db_manager.save_result(sample_result, "path", "display", "hash_fail")

    assert "Chroma Connection Error" in str(exc.value)

    # Verify SQLite is empty (Rollback worked)
    db_manager.cursor.execute("SELECT count(*) FROM photos")
    count = db_manager.cursor.fetchone()[0]
    assert count == 0

def test_duplicate_hash_prevention(db_manager, mock_chroma, sample_result):
    """
    NEW TEST: Verifies that saving the same hash twice triggers an error.
    """
    unique_hash = "shared_hash_123"

    # First save: Should succeed
    db_manager.save_result(sample_result, "path/A.jpg", "disp/A.jpg", unique_hash)

    # Second save: Different path, same hash -> Should Fail
    # This proves your UNIQUE constraint is working
    with pytest.raises(sqlite3.IntegrityError):
        db_manager.save_result(sample_result, "path/B.jpg", "disp/B.jpg", unique_hash)


def test_photo_exists(db_manager, sample_result):
    """Test the deduplication logic."""
    file_hash = "unique_hash_123"

    # 1. Should be False initially
    assert db_manager.photo_exists(file_hash) is False

    # 2. Add the photo
    db_manager.save_result(sample_result, "path", "display", file_hash)

    # 3. Should be True now
    assert db_manager.photo_exists(file_hash) is True


def test_person_management(db_manager):
    """Test updating and retrieving person names."""

    # Check default behavior
    assert db_manager.get_person_name(99) == "Person 99"

    # Set a name
    db_manager.update_person_name(99, "Alice")
    assert db_manager.get_person_name(99) == "Alice"

    # Update the name (Upsert logic)
    db_manager.update_person_name(99, "Alice Cooper")
    assert db_manager.get_person_name(99) == "Alice Cooper"