from __future__ import annotations

import sqlite3
import chromadb
import numpy as np
import uuid
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Any, Dict

# Import the data structures (ensure these exist in your src/ingestion/processor.py)
from src.ingestion.processor import ImageAnalysisResult


class DatabaseManager:
    """
    Manages the dual-database system:
    1. SQLite: For structured metadata (paths, timestamps, face linkages, identities).
    2. ChromaDB: For semantic vector search (CLIP embeddings).
    """

    def __init__(self, sql_path: str = "db/sqlite/photos.db", chroma_path: str = "db/chroma/"):
        # Ensure directories exist
        Path(sql_path).parent.mkdir(parents=True, exist_ok=True)
        Path(chroma_path).mkdir(parents=True, exist_ok=True)

        # 1. Setup SQLite
        self.conn = sqlite3.connect(sql_path)
        self.cursor = self.conn.cursor()
        self._init_sql_tables()

        # 2. Setup ChromaDB (Persistent)
        # We explicitly set the tenant/db names if needed, but defaults work fine for local
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)

        # Get or create the collection.
        # "cosine" distance is CRITICAL for normalized CLIP embeddings.
        self.vector_collection = self.chroma_client.get_or_create_collection(
            name="photo_gallery",
            metadata={"hnsw:space": "cosine"}
        )

    def _init_sql_tables(self):
        """
        Creates the necessary SQL tables if they don't exist.
        Enables Foreign Key support for SQLite.
        """
        self.cursor.execute("PRAGMA foreign_keys = ON;")

        # Table: Photos (The Master Record)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS photos(
                photo_id TEXT PRIMARY KEY,
                original_path TEXT,
                file_hash TEXT UNIQUE,
                display_path TEXT,
                timestamp DATETIME,
                width INTEGER,
                height INTEGER,
                meta_tags TEXT -- JSON list of auto-generated tags
            )
        """)

        # Table: People (The Identity Map)
        # Stores the "Human Readable" names for clusters
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS people(
                person_id INTEGER PRIMARY KEY,
                name TEXT DEFAULT 'Unknown',
                representative_face_id INTEGER
            )
        """)

        # Table: Photo_Faces (Links Photos <-> People)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS photo_faces(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id TEXT,
                person_id INTEGER DEFAULT - 1,    -- -1 means 'Unknown/Unclustered'
                embedding_blob BLOB,
                bounding_box TEXT, -- JSON [x, y, w, h]
                confidence REAL,
                FOREIGN KEY(photo_id) REFERENCES photos(photo_id) ON DELETE CASCADE
            )
        """)

        # Create indices for faster SQL lookups
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_person_id ON photo_faces(person_id);")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON photos(timestamp);")

        self.conn.commit()

    def photo_exists(self, file_hash: str) -> bool:
        """
        Checks if a photo has already been ingested to avoid duplicates.
        """
        self.cursor.execute("SELECT 1 FROM photos WHERE file_hash = ?", (file_hash,))
        return self.cursor.fetchone() is not None

    def save_result(self, result: ImageAnalysisResult, original_path: str, display_path: str, file_hash: str) -> str:
        """
        Saves the analysis result to both databases transactionally.

        Args:
            result: The output from FeatureExtractor.
            original_path: Where the file lives on disk.
            display_path: The web-friendly version (JPG).
            file_hash: The SHA-256 hash of the file content (for deduplication).

        Returns:
            str: The generated photo_id (UUID).
        """
        photo_id = str(uuid.uuid4())

        try:
            # --- Step A: Insert into SQLite ---

            # 1. Insert Photo Record
            # UPDATED: Now includes 'file_hash'
            self.cursor.execute("""
                        INSERT INTO photos (photo_id, original_path, display_path, file_hash, width, height)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """, (photo_id, original_path, display_path, file_hash, result.original_width,
                              result.original_height))

            # 2. Insert Face Records
            for face in result.faces:
                # FIX: Force numpy conversion to handle cases where InsightFace returns lists
                # This prevents the "'list' object has no attribute 'astype'" error
                emb_array = np.array(face.embedding, dtype=np.float32)
                emb_bytes = emb_array.tobytes()

                # Ensure bbox is JSON serializable
                bbox_json = json.dumps(face.bbox)

                self.cursor.execute("""
                        INSERT INTO photo_faces (photo_id, person_id, embedding_blob, bounding_box, confidence)
                        VALUES (?, -1, ?, ?, ?)
                        """, (photo_id, emb_bytes, bbox_json, face.confidence))

            self.conn.commit()

            # --- Step B: Insert into ChromaDB (Vector Store) ---

            metadata = {
                "path": display_path,
                "face_count": len(result.faces)
            }

            self.vector_collection.add(
                ids=[photo_id],
                embeddings=[result.semantic_vector.tolist()],  # Chroma expects list
                metadatas=[metadata]
            )

            return photo_id

        except Exception as e:
            self.conn.rollback()
            print(f"❌ Database Transaction Failed for {original_path}: {e}")
            raise e

    def get_person_name(self, person_id: int) -> str:
        """Helper to get a name from an ID."""
        self.cursor.execute("SELECT name FROM people WHERE person_id = ?", (person_id,))
        row = self.cursor.fetchone()
        return row[0] if row else f"Person {person_id}"

    def update_person_name(self, person_id: int, new_name: str):
        """Allows the UI to label 'Person 5' as 'Ethan'."""
        self.cursor.execute("""
            INSERT INTO people (person_id, name)
            VALUES (?, ?) ON CONFLICT(person_id) DO
            UPDATE SET name = ?
            """, (person_id, new_name, new_name))
        self.conn.commit()

    def close(self):
        self.conn.close()
        # Chroma client handles its own cleanup typically