import pytest
import sqlite3
import torch
from unittest.mock import MagicMock, patch
from src.retrieval.engine import SearchEngine


# --- Fixtures ---

@pytest.fixture
def mock_db():
    """
    Creates a temporary in-memory SQLite DB with some dummy data.
    """
    # 1. Create the REAL connection (to hold the data)
    real_conn = sqlite3.connect(":memory:")
    cursor = real_conn.cursor()

    # Create Tables & Data
    cursor.execute("CREATE TABLE people (person_id INTEGER PRIMARY KEY, name TEXT)")
    cursor.execute("CREATE TABLE photo_faces (photo_id TEXT, person_id INTEGER)")
    cursor.execute("INSERT INTO people VALUES (0, 'Ethan')")
    cursor.execute("INSERT INTO people VALUES (1, 'Jessica')")
    cursor.execute("INSERT INTO photo_faces VALUES ('photo_A', 0)")
    cursor.execute("INSERT INTO photo_faces VALUES ('photo_B', 0)")
    real_conn.commit()

    # 2. Create a WRAPPER (The Mock)
    # "wraps=real_conn" tells the Mock: "If I don't specify a behavior, just pass the call to the real connection."
    mock_wrapper = MagicMock(wraps=real_conn)

    # 3. Stub out .close() on the WRAPPER (This is allowed!)
    # Now, when the app calls .close(), it hits this fake method instead of the real one.
    mock_wrapper.close = MagicMock()

    # 4. Return the wrapper to the app
    yield mock_wrapper

    # 5. Clean up the REAL connection after test finishes
    real_conn.close()


@pytest.fixture
def engine(mock_db):
    """
    Initialize SearchEngine with mocked generic components
    but a REAL (in-memory) SQLite connection for logic testing.
    """
    # 1. Mock Chroma
    mock_chroma_client = MagicMock()
    mock_collection = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection

    # 2. Mock CLIP (Critical to avoid loading 500MB model during tests)
    with patch("src.retrieval.engine.clip.load") as mock_load, \
            patch("src.retrieval.engine.clip.tokenize") as mock_tokenize, \
            patch("chromadb.PersistentClient", return_value=mock_chroma_client), \
            patch("sqlite3.connect", return_value=mock_db):  # Hijack SQL connect to use our in-memory DB

        # Setup Fake CLIP Model
        mock_model = MagicMock()
        # Return a random tensor when encode_text is called
        mock_model.encode_text.return_value = torch.rand(1, 512)
        mock_load.return_value = (mock_model, None)

        # Initialize Engine
        engine = SearchEngine("dummy_sql_path", "dummy_chroma_path")

        # Attach mocks to the instance so we can spy on them in tests
        engine.collection = mock_collection
        engine.mock_model = mock_model

        yield engine


# --- Tests ---

def test_detect_person_in_query(engine):
    """
    Does it correctly find 'Ethan' in a sentence?
    """
    # Exact match
    pid = engine._get_person_id_from_name("Ethan")
    assert pid == 0

    # Embedded in sentence
    pid = engine._get_person_id_from_name("Show me photos of Ethan at the beach")
    assert pid == 0

    # Case insensitivity
    pid = engine._get_person_id_from_name("ethan is cool")
    assert pid == 0

    # Non-existent person
    pid = engine._get_person_id_from_name("John at the park")
    assert pid is None


def test_hybrid_search_flow(engine):
    """
    Scenario: User asks for 'Ethan at beach'.
    Expectation: The Chroma query MUST include a filter for Ethan's Photo IDs.
    """
    # Setup the mock return for Chroma
    engine.collection.query.return_value = {
        'ids': [['photo_A']],
        'metadatas': [[{'path': '/data/photo_A.jpg'}]]
    }

    results = engine.search("Ethan at beach")

    # 1. Verify CLIP was called
    engine.mock_model.encode_text.assert_called_once()

    # 2. Verify the 'Handshake': Did we filter by ID?
    # Arguments passed to collection.query(...)
    call_kwargs = engine.collection.query.call_args[1]

    # We expect 'where': {'id': {'$in': ['photo_A', 'photo_B']}}
    assert "where" in call_kwargs
    assert "id" in call_kwargs["where"]
    assert "$in" in call_kwargs["where"]["id"]

    # It should match the IDs from our mock_db fixture
    allowed_ids = call_kwargs["where"]["id"]["$in"]
    assert "photo_A" in allowed_ids
    assert "photo_B" in allowed_ids


def test_pure_semantic_search_flow(engine):
    """
    Scenario: User asks for 'A sunny beach' (No name mentioned).
    Expectation: Chroma query should NOT have a 'where' filter.
    """
    engine.collection.query.return_value = {'metadatas': [[]]}

    engine.search("A sunny beach")

    call_kwargs = engine.collection.query.call_args[1]

    # Crucial check: 'where' should not exist or be None
    assert "where" not in call_kwargs or call_kwargs["where"] is None


def test_person_with_no_photos(engine):
    """
    Scenario: User asks for 'Jessica', but Jessica has 0 photos.
    Expectation: Should return empty list immediately without hitting Chroma.
    """
    results = engine.search("Jessica at the park")

    # Should return empty list
    assert results == []

    # Should NOT have called Chroma (Efficiency check)
    engine.collection.query.assert_not_called()


def test_chroma_error_handling(engine):
    """
    Scenario: ChromaDB crashes/fails.
    Expectation: Return empty list, don't crash the app.
    """
    # Make Chroma raise an exception
    engine.collection.query.side_effect = Exception("DB Connection Lost")

    results = engine.search("Sunset")
    assert results == []