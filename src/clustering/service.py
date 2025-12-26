from __future__ import annotations
import sqlite3
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple

from src.clustering.face_clusterer import FaceRecord, FaceClusterer, ClusterResult
from src.db.storage import DatabaseManager

class ClusteringService:
    """
    Orchestrator that pulls unclustered faces from the database, runs the
    DBSCAN algorithm, and updates the database with the new identities.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        """
        Initialize the service with a database connection.
        Args:
            db_manager (DatabaseManager): Database connection manager.
        """

        self.db = db_manager
        # Initialize the math engine (DBSCAN)
        # eps = 0.4 is the distance threshold for InsightFace embeddings
        self.clusterer = FaceClusterer(eps=0.4, min_samples=3)

    def load_face_from_db(self) -> List[FaceRecord]:
        """
        Fetch all faces from SQLite to prepare for clustering
        Returns:
            List[FaceRecord]: List of FaceRecord objects.
        """
        # Join with photos table to get the full file path for debugging and visualization
        query = """
            SELECT pf.id, pf.embedding_blob, p.display_path 
            FROM photo_faces pf
            JOIN photos p ON pf.photo_id = p.photo_id
        """

        self.db.cursor.execute(query)
        rows = self.db.cursor.fetchall()

        records: List[FaceRecord] = []
        for row in rows:
            face_id, blob, path = row

            # Critical: Convert binary blob back into np.float32 array
            # matches the .tobytes() call during ingestion
            embedding = np.frombuffer(blob, dtype=np.float32)
            records.append(
                FaceRecord(
                    face_id=face_id,
                    embedding=embedding,
                    original_file_path=path
                )
            )
        return records

    def run(self) -> None:
        """
        Main execution method
        1. Load faces
        2. Runs clustering
        3. Saves results to DB
        """
        print("Loading faces...")
        faces = self.load_face_from_db()

        if not faces:
            print("No faces found to cluster.")
            return
        print(f"Clustering {len(faces)} faces...")

        results: Dict[int, ClusterResult] = self.clusterer.run(faces)

        print(f"Saving {len(results)} unique people to DB...")
        self.save_to_db(results)

    def save_to_db(self, clusters: Dict[int, ClusterResult]) -> None:
        """
        Writes the clustering results back to SQLite.
        Args:
            clusters (Dict[int, ClusterResult]): A dictionary mapping cluster_id to ClusterResult.
        """

        try:
            # Reset everyone to unknown to ensure clean state
            # This handles cases where a person was deleted or merged
            self.db.cursor.execute("UPDATE photo_faces SET person_id = -1")

            # Update members of each cluster
            # Prepare a batch list for faster execution
            update_queue: List[Tuple[int, int]] = []

            for c_id, res in clusters.items():
                for face_id in res.member_ids:
                    update_queue.append((c_id, face_id))
            self.db.cursor.executemany(
                "UPDATE photo_faces SET person_id = ? WHERE id = ?",
                update_queue
            )
            # Update 'people' table with the representative face
            # This table is used by the UI to show a thumbnail for "Person 5"
            for c_id, res in clusters.items():
                self.db.cursor.execute("""
                   INSERT INTO people (person_id, representative_face_id)
                   VALUES (?, ?) ON CONFLICT(person_id) DO
                   UPDATE SET representative_face_id = ?
                   """, (c_id, res.representative_face_id, res.representative_face_id))
            self.db.conn.commit()
            print("Clustering complete!")
        except Exception as e:
            print(f"Failed to save clusters to DB: {e}")
            self.db.conn.rollback()