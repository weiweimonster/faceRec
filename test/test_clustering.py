import pytest
import numpy as np
from src.clustering.face_clusterer import  FaceClusterer, FaceRecord
from src.clustering.service import ClusteringService

def test_face_clusterer_grouping():
    """
    Create synthetic data:
    - 5 vectors for Person A (very similar)
    - 5 vectors for Person B (very similar)
    - 1 noise vector
    """
    person_a = np.random.rand(512).astype(np.float32)
    person_b = -person_a  # Opposite direction = max distance

    records = []

    # Create Person A group
    for i in range(5):
        vec = person_a + np.random.normal(0, 0.01, 512).astype(np.float32)
        records.append(FaceRecord(face_id=i, embedding=vec, original_file_path=""))

    for i in range(5, 10):
        vec = person_b + np.random.normal(0, 0.01, 512).astype(np.float32)
        records.append(FaceRecord(face_id=i, embedding=vec, original_file_path=""))

    clusterer = FaceClusterer(eps=0.5, min_samples=2)
    results = clusterer.run(records)
    # Should have 2 clusters (A and B)
    assert len(results) == 2

    # Verify members
    # Get the cluster ID for the first person
    first_cluster_id = None
    for cid, res in results.items():
        if 0 in res.member_ids:
            first_cluster_id = cid
            break
    assert first_cluster_id is not None
    # Ensure all Person A are in this cluster
    assert set(results[first_cluster_id].member_ids).issuperset({0, 1, 2, 3, 4})
    assert 5 not in results[first_cluster_id].member_ids

def test_clustering_service_saves_to_db(tmp_path):
    from src.db.storage import DatabaseManager
    # Set up real db in temp file
    db = DatabaseManager(sql_path=str(tmp_path / "test.db"), chroma_path=str(tmp_path / "chroma"))
    service = ClusteringService(db)

    # Insert Fake Data manually into DB
    fake_vec = np.random.rand(512).astype(np.float32)
    db.cursor.execute("INSERT INTO photos (photo_id) VALUES ('p1')")
    db.cursor.execute("INSERT INTO photos (photo_id) VALUES ('p2')")

    # Insert 3 faces that are identical (Zero Vector) -> Should cluster together
    db.cursor.execute("INSERT INTO photo_faces (photo_id, embedding_blob) VALUES ('p1', ?)", (fake_vec,))
    db.cursor.execute("INSERT INTO photo_faces (photo_id, embedding_blob) VALUES ('p1', ?)", (fake_vec,))
    db.cursor.execute("INSERT INTO photo_faces (photo_id, embedding_blob) VALUES ('p2', ?)", (fake_vec,))
    db.conn.commit()

    service.run()
    db.cursor.execute("SELECT person_id FROM photo_faces")
    pids = [r[0] for r in db.cursor.fetchall()]

    # DBSCAN label starts at 0. So we expect [0, 0, 0]
    assert len(pids) == 3
    assert pids[0] != -1
    assert pids[0] == pids[1] == pids[2] # All same person

    # Verify People Table
    db.cursor.execute("SELECT count(*) FROM people")
    assert db.cursor.fetchone()[0] == 1


