import pytest
import numpy as np
import sqlite3
import json
from src.db.storage import DatabaseManager
from src.common.types import ImageAnalysisResult, FaceData
from src.pose.pose import Pose
from unittest.mock import MagicMock, patch

from src.util.search_config import SearchFilters


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
        # Essential keys
        photo_id=None,
        original_path="/orig/a.jpg",
        display_path="/disp/a.jpg",
        semantic_vector=np.random.rand(512).astype(np.float32),
        # Metadata
        original_width=1920,
        original_height=1080,
        timestamp="2025-01-01 10:00:00",
        aesthetic_score=7.5,  # NEW FIELD
        faces=[
            FaceData(
                bbox=[100, 100, 200, 200],
                embedding=np.random.rand(512).astype(np.float32),
                confidence=0.98,
                shot_type="Medium-Shot",
                blur_score=0.5,
                brightness=100.0,
                yaw=10.5,   # NEW FIELD
                pitch=-5.2, # NEW FIELD
                roll=1.1,   # NEW FIELD
                pose=Pose.FRONT
            )
        ]
    )

@pytest.fixture
def mock_filters():
    """Creates a mock SearchFilters object with default behaviors."""
    filters = MagicMock()
    filters.fn_name = "search"
    filters.is_person_search = False
    filters.people = []
    filters.year = None
    filters.pose = None
    filters.semantic_query = None
    return filters

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

    # 1. Verify Photo Record including aesthetic_score
    db_manager.cursor.execute("SELECT aesthetic_score, width, height FROM photos WHERE photo_id=?", (pid,))
    row = db_manager.cursor.fetchone()
    assert row[0] == 7.5
    assert row[1] == 1920

    # 2. Verify Face Record including Pose and Shot Type
    db_manager.cursor.execute("""
                              SELECT shot_type, yaw, pitch, roll, pose
                              FROM photo_faces WHERE photo_id=?
                              """, (pid,))
    face_row = db_manager.cursor.fetchone()
    assert face_row[0] == "Medium-Shot"
    assert face_row[1] == 10.5
    assert face_row[2] == -5.2
    assert face_row[4] == "Front"

    # 3. Verify Photo Record
    db_manager.cursor.execute("SELECT * from photos WHERE photo_id=?", (pid,))
    photo_row = db_manager.cursor.fetchone()
    assert photo_row is not None
    assert photo_row[1] == "/orig/a.jpg"  # original_path

    # 4. Verify the hash was stored (assuming it's column index 3 or by name)
    db_manager.cursor.execute("SELECT file_hash FROM photos WHERE photo_id=?", (pid,))
    assert db_manager.cursor.fetchone()[0] == "dummy_hash_1"

    # 5. Verify Face Record
    db_manager.cursor.execute("SELECT * from photo_faces WHERE photo_id=?", (pid,))
    face_row = db_manager.cursor.fetchone()
    assert face_row is not None
    assert face_row[2] == -1  # person_id default
    assert isinstance(face_row[3], bytes)  # embedding blob

    # 6. Verify Chroma Vector
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

def test_get_candidate_path_scene_mode(db_manager, mock_filters):
    """Verify simple SELECT when not in person search mode."""
    db_manager.cursor.execute("INSERT INTO photos (photo_id, display_path) VALUES ('1', 'd/1.jpg')")

    mock_filters.is_person_search = False
    paths = db_manager.get_candidate_path(mock_filters)

    assert paths == ["d/1.jpg"]

def test_get_candidate_path_person_intersection(db_manager, mock_filters):
    """Verify the GROUP BY / HAVING logic for multiple people."""
    # Setup: 1 photo with 2 people
    db_manager.cursor.execute("INSERT INTO photos (photo_id, display_path) VALUES ('p1', 'path/1.jpg')")
    db_manager.cursor.execute("INSERT INTO people (person_id, name) VALUES (1, 'Alice'), (2, 'Bob')")
    db_manager.cursor.execute("INSERT INTO photo_faces (photo_id, person_id, confidence) VALUES ('p1', 1, 0.9), ('p1', 2, 0.9)")
    db_manager.conn.commit()

    mock_filters.is_person_search = True
    mock_filters.people = ["Alice", "Bob"]

    paths = db_manager.get_candidate_path(mock_filters)
    assert len(paths) == 1
    assert paths[0] == "path/1.jpg"

def test_fetch_metadata_batch_hydration_logic(db_manager, sample_result):
    """Verify that 'all' correctly maps DB rows to ImageAnalysisResult objects."""
    # Ingest a real sample
    db_manager.save_result(sample_result, "orig/1.jpg", "disp/1.jpg", "hash1")

    # Run hydration
    results = db_manager.fetch_metadata_batch(["disp/1.jpg"], fields="all")

    assert len(results) == 1
    res = results[0]
    assert isinstance(res, ImageAnalysisResult)
    assert res.aesthetic_score == sample_result.aesthetic_score
    assert len(res.faces) == 1
    assert res.faces[0].yaw == sample_result.faces[0].yaw

def test_fetch_metadata_batch_invalid_path(db_manager):
    """Ensure querying non-existent paths returns an empty list safely."""
    results = db_manager.fetch_metadata_batch(["non_existent.jpg"], fields="all")
    assert results == []

def test_get_semantic_candidates_math(db_manager, mock_chroma):
    """Verify distance to similarity conversion logic."""
    # Mock ChromaDB response
    mock_chroma["collection"].query.return_value = {
        'metadatas': [[{'path': 'a.jpg'}, {'path': 'b.jpg'}]],
        'distances': [[0.1, 1.9]] # 0.1 is very similar, 1.9 is opposite
    }

    dummy_vec = [0.0] * 512
    scores = db_manager.get_semantic_candidates(dummy_vec, allowed_paths=[], limit=2)

    assert scores["a.jpg"] == pytest.approx(0.9) # 1.0 - 0.1
    assert scores["b.jpg"] == pytest.approx(0.0) # max(0, 1.0 - 1.9)