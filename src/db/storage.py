from __future__ import annotations

import pickle
import sqlite3
import chromadb
import numpy as np
import uuid
import json
from pathlib import Path
from typing import List, Optional, Any, Dict, Tuple
from src.util.logger import logger
from src.ingestion.processor import FaceData, ImageAnalysisResult
from src.util.search_config import SearchFilters


class DatabaseManager:
    """
    Manages the dual-database system:
    1. SQLite: For structured metadata (paths, timestamps, face linkages, identities).
    2. ChromaDB: For semantic vector search (CLIP embeddings).
    """

    def __init__(self, sql_path: str = ".db/sqlite/photos.db", chroma_path: str = "db/chroma/"):
        # Ensure directories exist
        Path(sql_path).parent.mkdir(parents=True, exist_ok=True)
        Path(chroma_path).mkdir(parents=True, exist_ok=True)

        # 1. Setup SQLite
        # Use False for streamlit multithread
        self.conn = sqlite3.connect(sql_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_sql_tables()
        self.init_ltr_tables()

        # 2. Setup ChromaDB (Persistent)
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)

        # Get or create the collections.
        # "cosine" distance is CRITICAL for normalized CLIP embeddings.
        self.vector_collection = self.chroma_client.get_or_create_collection(
            name="photo_gallery",
            metadata={"hnsw:space": "cosine"}
        )

        self.caption_collection = self.chroma_client.get_or_create_collection(
            name="caption_gallery",
            metadata={"hnsw:space": "cosine"}
        )

    def init_ltr_tables(self):
        """
        Creates tables for Learning-to-Rank data collection.
        1. search_history: Maps session_id -> Query/Intent + ranking model used
        2. search_interactions: Maps session_id + photo_id -> Label + Feature Snapshot + position
        3. search_impressions: Tracks all photos shown in search results
        """
        # Table 1: The Search Session (Context)
        self.cursor.execute("""
                            CREATE TABLE IF NOT EXISTS search_history
                            (
                                session_id          TEXT PRIMARY KEY,
                                raw_query           TEXT,
                                parsed_filters_json TEXT,
                                ranking_model       TEXT DEFAULT 'heuristic',
                                timestamp           DATETIME DEFAULT CURRENT_TIMESTAMP
                            )
                            """)

        # Table 2: User interactions (clicks/selections)
        self.cursor.execute("""
                            CREATE TABLE IF NOT EXISTS search_interactions
                            (
                                interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                session_id     TEXT,
                                photo_id       TEXT,
                                label          INTEGER  DEFAULT 0,
                                position       INTEGER  DEFAULT -1,
                                features_json  TEXT,
                                timestamp      DATETIME DEFAULT CURRENT_TIMESTAMP,
                                FOREIGN KEY (session_id) REFERENCES search_history (session_id),
                                FOREIGN KEY (photo_id) REFERENCES photos (photo_id)
                            )
                            """)

        # Table 3: Impressions (all photos shown in search results)
        self.cursor.execute("""
                            CREATE TABLE IF NOT EXISTS search_impressions
                            (
                                impression_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                session_id    TEXT NOT NULL,
                                photo_id      TEXT NOT NULL,
                                position      INTEGER NOT NULL,
                                timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP,
                                FOREIGN KEY (session_id) REFERENCES search_history (session_id),
                                UNIQUE(session_id, photo_id)
                            )
                            """)

        # Create indices for faster lookups
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_imp_session ON search_impressions(session_id);")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_imp_timestamp ON search_impressions(timestamp);")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_session ON search_interactions(session_id);")

        # Add new columns to existing tables if they don't exist (for migration)
        self._migrate_ltr_tables()

        self.conn.commit()

    def _migrate_ltr_tables(self):
        """Add new columns to existing tables for backwards compatibility."""
        # Check and add ranking_model to search_history
        self.cursor.execute("PRAGMA table_info(search_history)")
        columns = [col[1] for col in self.cursor.fetchall()]
        if 'ranking_model' not in columns:
            self.cursor.execute("ALTER TABLE search_history ADD COLUMN ranking_model TEXT DEFAULT 'heuristic'")
            logger.info("Migrated search_history: added ranking_model column")

        # Check and add position to search_interactions
        self.cursor.execute("PRAGMA table_info(search_interactions)")
        columns = [col[1] for col in self.cursor.fetchall()]
        if 'position' not in columns:
            self.cursor.execute("ALTER TABLE search_interactions ADD COLUMN position INTEGER DEFAULT -1")
            logger.info("Migrated search_interactions: added position column")

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
                month INTEGER,
                time_period TEXT,
                width INTEGER,
                height INTEGER,
                aesthetic_score REAL,
                iso INTEGER,
                global_blur REAL,
                global_brightness REAL,
                global_contrast REAL,
                meta_tags TEXT, -- JSON list of auto-generated tags
                caption_text TEXT -- Generated image caption
            )
        """)

        # Table: People (The Identity Map)ra
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
                pose TEXT,
                shot_type TEXT,
                blur_score REAL,
                brightness REAL,
                face_height REAL,
                yaw REAL,
                pitch REAL,
                roll REAL,
                FOREIGN KEY(photo_id) REFERENCES photos(photo_id) ON DELETE CASCADE
            )
        """)

        # Create indices for faster SQL lookups
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_person_id ON photo_faces(person_id);")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON photos(timestamp);")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_time_meta ON photos(month, time_period);")

        # Migration: Add caption_text column to existing photos table
        self.cursor.execute("PRAGMA table_info(photos)")
        columns = [col[1] for col in self.cursor.fetchall()]
        if 'caption_text' not in columns:
            self.cursor.execute("ALTER TABLE photos ADD COLUMN caption_text TEXT")
            logger.info("Migrated photos: added caption_text column")

        self.conn.commit()
    def log_search_query(self, session_id: str, query: str, filters: Any, ranking_model: str = "heuristic"):
        """Saves the context (intent) of the search including the ranking model used."""
        try:
            filters_dict = filters.to_dict() if hasattr(filters, "to_dict") else filters
            self.cursor.execute("""
                    INSERT OR REPLACE INTO search_history (session_id, raw_query, parsed_filters_json, ranking_model)
                    VALUES (?, ?, ?, ?)
                """, (session_id, query, json.dumps(filters_dict), ranking_model))
            self.conn.commit()
            logger.info(f"📝 Logged Session: {session_id[:8]}... | Query: {query} | Model: {ranking_model}")
        except Exception as e:
            logger.error(f"Failed to log search query: {e}")

    def log_impressions(self, session_id: str, photo_ids_with_positions: List[Tuple[str, int]]) -> int:
        """
        Log all photos shown in search results with their positions.

        Args:
            session_id: The search session ID
            photo_ids_with_positions: List of (photo_id, position) tuples

        Returns:
            Number of impressions logged
        """
        if not photo_ids_with_positions:
            return 0

        try:
            # Use INSERT OR IGNORE to handle duplicate impressions gracefully
            self.cursor.executemany("""
                INSERT OR IGNORE INTO search_impressions (session_id, photo_id, position)
                VALUES (?, ?, ?)
            """, [(session_id, photo_id, pos) for photo_id, pos in photo_ids_with_positions])
            self.conn.commit()
            count = self.cursor.rowcount
            logger.info(f"📊 Logged {count} impressions for session {session_id[:8]}...")
            return count
        except Exception as e:
            logger.error(f"Failed to log impressions: {e}")
            return 0

    def log_interaction_from_features(
            self,
            result: ImageAnalysisResult,
            session_id: str,
            features: Dict[str, float],
            label: int = 1,
            position: int = -1
    ):
        """
        Log user interaction with pre-extracted features.

        Args:
            result: The ImageAnalysisResult that was clicked/skipped
            session_id: Unique search session ID
            features: Pre-extracted feature dict from ranker
            label: 1 for click (positive), 0 for skip (negative)
            position: Position in search results (0-indexed), -1 if unknown

        Example:
            features = {
                "semantic_score": 0.88,
                "aesthetic_score": 4.5,
                "g_blur": 120.3,
                "f_conf": 0.95,
                ...
            }
            db.log_interaction_from_features(result, session_id, features, label=1, position=3)
        """
        try:
            self.cursor.execute("""
                                INSERT INTO search_interactions (session_id, photo_id, label, position, features_json)
                                VALUES (?, ?, ?, ?, ?)
                                """, (session_id, result.photo_id, label, position, json.dumps(features)))

            self.conn.commit()
            logger.info(f"✅ Logged {'click' if label else 'skip'} for {result.photo_id} at position {position} ({len(features)} features)")

        except Exception as e:
            logger.error(f"Failed to log interaction for {result.photo_id}: {e}")

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
            # 1. Insert Photo Record
            self.cursor.execute("""
                                INSERT INTO photos (
                                    photo_id, original_path, display_path, file_hash,
                                    width, height, timestamp, month, time_period,
                                    aesthetic_score, iso, global_blur, global_brightness, global_contrast,
                                    caption_text
                                )
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    photo_id, original_path, display_path, file_hash,
                                    result.original_width, result.original_height,
                                    result.timestamp, result.month, result.time_period,
                                    result.aesthetic_score, result.iso, result.global_blur,
                                    result.global_brightness, result.global_contrast,
                                    result.caption
                                ))

            # 2. Insert Face Records
            for face in result.faces:
                # FIX: Force numpy conversion to handle cases where InsightFace returns lists
                # This prevents the "'list' object has no attribute 'astype'" error
                emb_array = np.array(face.embedding, dtype=np.float32)
                emb_bytes = emb_array.tobytes()

                # Ensure bbox is JSON serializable
                bbox_json = json.dumps(face.bbox)

                shot_type = face.shot_type if face.shot_type else "Unknown"
                blur_score = face.blur_score if face.blur_score else 0.0
                brightness = face.brightness if face.brightness else 0.0
                yaw = face.yaw if face.yaw else 0.0
                pitch = face.pitch if face.pitch else 0.0
                roll = face.roll if face.roll else 0.0
                pose = str(face.pose) if face.pose else "Unknown"

                self.cursor.execute(
                    """INSERT INTO photo_faces (photo_id, person_id, embedding_blob, bounding_box, confidence,
                        shot_type, blur_score, brightness, yaw, pitch, roll, pose)
                        VALUES (?, -1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (photo_id, emb_bytes, bbox_json, face.confidence, shot_type, blur_score,
                              brightness, yaw, pitch, roll, pose))

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

            # Only add to caption collection if caption_vector exists
            if result.caption_vector is not None:
                self.caption_collection.add(
                    ids=[photo_id],
                    embeddings=[result.caption_vector],
                    metadatas=[metadata]
                )
            else:
                logger.warning(f"No caption vector for {photo_id}, skipping caption collection")

            self.conn.commit()

            return photo_id

        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ Database Transaction Failed for {original_path}: {e}")
            raise e

    def save_result_batch(
        self,
        results: list,
        original_paths: list,
        display_paths: list,
        file_hashes: list,
    ) -> list:
        """
        Saves multiple analysis results to both databases in a single transaction.

        More efficient than calling save_result() multiple times as it batches
        the commits.

        Args:
            results: List of ImageAnalysisResult objects
            original_paths: List of original file paths
            display_paths: List of web-friendly paths
            file_hashes: List of SHA-256 hashes

        Returns:
            List of generated photo_ids (UUIDs)
        """
        if len(results) != len(original_paths) != len(display_paths) != len(file_hashes):
            raise ValueError("All input lists must have the same length")

        photo_ids = []

        try:
            for result, original_path, display_path, file_hash in zip(
                results, original_paths, display_paths, file_hashes
            ):
                photo_id = str(uuid.uuid4())
                photo_ids.append(photo_id)

                # 1. Insert Photo Record
                self.cursor.execute("""
                    INSERT INTO photos (
                        photo_id, original_path, display_path, file_hash,
                        width, height, timestamp, month, time_period,
                        aesthetic_score, iso, global_blur, global_brightness, global_contrast,
                        caption_text
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    photo_id, original_path, display_path, file_hash,
                    result.original_width, result.original_height,
                    result.timestamp, result.month, result.time_period,
                    result.aesthetic_score, result.iso, result.global_blur,
                    result.global_brightness, result.global_contrast,
                    result.caption
                ))

                # 2. Insert Face Records
                for face in (result.faces or []):
                    emb_array = np.array(face.embedding, dtype=np.float32)
                    emb_bytes = emb_array.tobytes()
                    bbox_json = json.dumps(face.bbox)

                    shot_type = face.shot_type if face.shot_type else "Unknown"
                    blur_score = face.blur_score if face.blur_score else 0.0
                    brightness = face.brightness if face.brightness else 0.0
                    yaw = face.yaw if face.yaw else 0.0
                    pitch = face.pitch if face.pitch else 0.0
                    roll = face.roll if face.roll else 0.0
                    pose = str(face.pose) if face.pose else "Unknown"

                    self.cursor.execute("""
                        INSERT INTO photo_faces (
                            photo_id, person_id, embedding_blob, bounding_box, confidence,
                            shot_type, blur_score, brightness, yaw, pitch, roll, pose
                        )
                        VALUES (?, -1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        photo_id, emb_bytes, bbox_json, face.confidence, shot_type,
                        blur_score, brightness, yaw, pitch, roll, pose
                    ))

                # 3. Insert into ChromaDB
                metadata = {
                    "path": display_path,
                    "face_count": len(result.faces) if result.faces else 0
                }

                self.vector_collection.add(
                    ids=[photo_id],
                    embeddings=[result.semantic_vector.tolist()],
                    metadatas=[metadata]
                )

                if result.caption_vector is not None:
                    self.caption_collection.add(
                        ids=[photo_id],
                        embeddings=[result.caption_vector],
                        metadatas=[metadata]
                    )

            # Single commit for entire batch
            self.conn.commit()
            logger.info(f"✅ Batch saved {len(photo_ids)} photos to database")

            return photo_ids

        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ Batch save failed: {e}")
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
        # Release ChromaDB resources
        if hasattr(self, 'chroma_client') and self.chroma_client:
            self.vector_collection = None
            self.caption_collection = None
            self.chroma_client = None

    def get_candidate_path(self, filters: SearchFilters) -> List[str]:
        params = []

        # This ensures it exists for month, time_period, and year filters.
        p = "ph." if filters.is_person_search else ""

        # Determine base query based on search mode
        if filters.is_person_search:
            logger.info("Mode: Person Search")
            query = """
                    SELECT DISTINCT ph.display_path
                    FROM photo_faces pf
                             JOIN photos ph ON pf.photo_id = ph.photo_id
                             LEFT JOIN people p ON pf.person_id = p.person_id
                    WHERE 1 = 1 \
                    """
        else:
            logger.info("Mode: Scene Search (Targeting all photos)")
            query = "SELECT display_path FROM photos WHERE 1=1"

        if filters.pose:
            logger.info(f"Pose Filter added to SQL Query: {str(filters.pose)}")
            query += " AND pf.pose LIKE ?"
            params.append(str(filters.pose))

        if filters.year:
            logger.info(f"Year Filter added to SQL Query: {str(filters.year)}")
            # You can now safely use the 'p' defined at the top
            query += f" AND strftime('%Y', {p}timestamp) = ?"
            params.append(str(filters.year))

        if filters.month:
            logger.info(f"Month Filter: {filters.month}")
            query += f" AND {p}month = ?"
            params.append(filters.month)

        if filters.time_period:
            logger.info(f"Time Period Filter: {filters.time_period}")
            query += f" AND {p}time_period = ?"
            params.append(filters.time_period.lower())

        if filters.people:
            logger.info("People filter detected, parsing names.....")
            name_conditions = ["p.name LIKE ?" for _ in filters.people]
            for name in filters.people:
                logger.info(f"    Adding {name} to SQL Query")
                params.append(f"%{name}%")

            query += " AND (" + " OR ".join(name_conditions) + ")"
            query += " GROUP BY ph.photo_id HAVING COUNT(DISTINCT p.name) >= ?"
            params.append(len(filters.people))

        try:
            logger.info(f"Query: {str(query)} with Params: {str(params)}")
            self.cursor.execute(query, tuple(params))
            return [row[0] for row in self.cursor.fetchall()]
        except Exception as e:
            logger.error(f"SQL Candidate Search Error: {e}")
            return []

    def get_semantic_candidates(
            self,
            query_vector: List[float],
            allowed_paths: List[str],
            limit: int,
            collection: str = "visual"
    ) -> Dict[str, float]:
        if collection == "caption":
            target_col = self.caption_collection
        else:
            target_col = self.vector_collection
        # Note: Directly taking query_vector to prevent loading heavy model like CLIP
        logger.info(f"Performing Vector Search with {collection}")

        search_params = {
            "query_embeddings": [query_vector],
            "n_results": limit,
            "include": ["metadatas", "distances", "embeddings"]
        }

        if allowed_paths:
            search_params["where"] = {"path": {"$in": allowed_paths}}

        try:
            results = target_col.query(**search_params)
            # Dict: Path -> Semantic Score
            output = {}
            if results['metadatas']:
                metas = results['metadatas'][0]
                dists = results['distances'][0]
                vecs = results['embeddings'][0]

                for i, meta in enumerate(metas):
                    path = meta.get('path')

                    # Convert Distance to Similarity (1 - distance)
                    sim = max(0.0, 1.0 - dists[i])

                    # Convert List -> Numpy Array (Float32 is standard for Torch)
                    vector = np.array(vecs[i], dtype=np.float32)

                    output[path] = (sim, vector)
            return output
        except Exception as e:
            logger.error(f"Chroma Search Error: {e}")
            return {}

    def fetch_metadata_batch(self, paths: List[str], fields: str | List[str] = "all") -> List[ImageAnalysisResult]:
        """
        Fetches metadata for a list of photos and hydrates them into ImageAnalysisResult objects.

        Args:
            paths: List of display_paths to fetch.
            fields: A list of specific metadata keys or "all" to fetch everything.
        """
        if not paths:
            return []

        # 1. Map requested fields to their actual SQL columns
        valid_map = {
            "timestamp": "ph.timestamp",
            "yaw": "pf.yaw",
            "pitch": "pf.pitch",
            "roll": "pf.roll",
            "bbox": "pf.bounding_box",
            "confidence": "pf.confidence",
            "pose": "pf.pose",
            "blur": "pf.blur_score",
            "brightness": "pf.brightness",
            "face_height": "pf.face_height",
            "shot_type": "pf.shot_type",
            "embedding": "pf.embedding_blob",
            "name": "p.name",
            "original_path": "ph.original_path",
            "photo_id": "ph.photo_id",
            "aesthetic_score": "ph.aesthetic_score",
            "width": "ph.width",
            "height": "ph.height",
            "iso": "ph.iso",
            "global_blur": "ph.global_blur",
            "global_brightness": "ph.global_brightness",
            "global_contrast": "ph.global_contrast",
            "month": "ph.month",
            "time_period": "ph.time_period",
            "caption_text": "ph.caption_text",
        }

        # Handle the "all" logic
        target_fields = list(valid_map.keys()) if fields == "all" else fields
        if not isinstance(target_fields, list):
            target_fields = [target_fields]

        # 2. Build Query
        base_cols = ["ph.display_path", "ph.photo_id"]
        requested_sql_cols = [valid_map[f] for f in target_fields if f in valid_map]

        # Deduplicate columns and create string
        cols_str = ", ".join(list(dict.fromkeys(base_cols + requested_sql_cols)))
        placeholders = ",".join(["?"] * len(paths))

        query = f"""
            SELECT {cols_str} FROM photos ph
            LEFT JOIN photo_faces pf ON ph.photo_id = pf.photo_id
            LEFT JOIN people p ON pf.person_id = p.person_id
            WHERE ph.display_path IN ({placeholders})
        """

        logger.info(f"Executing query: {str(query)} to fetch metadata for {len(paths)}")
        self.cursor.execute(query, paths)
        rows = self.cursor.fetchall()
        logger.info(f"Fetched {len(rows)} from database")

        # 3. Process Rows into Objects
        results_map: Dict[str, ImageAnalysisResult] = {}
        col_names = [d[0] for d in self.cursor.description]

        logger.info(f"Debug: cols names: {col_names}")

        for i, row in enumerate(rows):
            row_dict = dict(zip(col_names, row))
            p_id = row_dict['photo_id']

            # Initialize the ImageAnalysisResult if not already in map
            if p_id not in results_map:
                results_map[p_id] = ImageAnalysisResult(
                    photo_id=p_id,
                    display_path=row_dict['display_path'],
                    original_path=row_dict.get('original_path'),
                    timestamp=row_dict.get('timestamp'),
                    aesthetic_score=row_dict.get('aesthetic_score'),
                    original_width=row_dict.get('width'),
                    original_height=row_dict.get('height'),
                    iso=row_dict.get('iso'), # Can be None
                    global_blur=row_dict.get('global_blur'),
                    global_brightness=row_dict.get('global_brightness'),
                    global_contrast=row_dict.get('global_contrast'),
                    time_period=row_dict.get('time_period'),
                    month=row_dict.get('month'),
                    caption=row_dict.get('caption_text') or "",
                    faces=[]
                )
            # 4. Extract Face Data (only if confidence exists, meaning a face was joined)
            if row_dict.get('confidence') is not None:
                embedding = None
                if 'embedding_blob' in row_dict and row_dict['embedding_blob']:
                    embedding = np.frombuffer(row_dict['embedding_blob'], dtype=np.float32)

                face = FaceData(
                    # If we haven't labeled it, then name field will be NULL'
                    # TODO: Figure out a better way to handle this gracefully
                    name=row_dict.get('name', 'Unknown') or 'Unknown', # Added name back in
                    bbox=json.loads(row_dict['bounding_box']) if row_dict.get('bounding_box') else None,
                    confidence=row_dict['confidence'],
                    yaw=row_dict.get('yaw', -1.0),
                    pitch=row_dict.get('pitch', -1.0),
                    roll=row_dict.get('roll', -1.0),
                    shot_type=row_dict.get('shot_type'),
                    blur_score=row_dict.get('blur_score'), # Fixed key to match valid_map
                    brightness=row_dict.get('brightness'),
                    pose=row_dict.get('pose'),
                    embedding=embedding,
                )
                results_map[p_id].faces.append(face)

        return list(results_map.values())

    def get_people_clusters(self) -> List[Dict[str, Any]]:
        """
        Returns a list of people for the sidebar: (person_id, name, display_path, bbox_json)
        """
        query = """
                SELECT p.person_id, p.name, ph.display_path, pf.bounding_box
                FROM people p
                         JOIN photo_faces pf ON p.representative_face_id = pf.id
                         JOIN photos ph ON pf.photo_id = ph.photo_id
                ORDER BY p.person_id ASC \
                """
        try:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()

            # Convert raw tuples to friendly dictionaries
            clusters = []
            for r in rows:
                clusters.append({
                    "id": r[0],
                    "name": r[1],
                    "face_path": r[2],
                    "bbox": r[3]
                })
            return clusters
        except Exception as e:
            logger.error(f"Failed to fetch clusters: {e}")
            return []


    def merge_identity(self, old_pid: int, new_name: str) -> str:
        """
        Renames a person. If 'new_name' already exists, merges the old ID into the new ID.
        Returns a status message for the UI.
        """
        new_name = new_name.strip()

        # 1. Check if the target name already exists
        self.cursor.execute("SELECT person_id FROM people WHERE name = ? AND person_id != ?", (new_name, old_pid))
        row = self.cursor.fetchone()

        if row:
            # MERGE: Target exists, move all faces to target and delete old person
            target_pid = row[0]
            self.cursor.execute("UPDATE photo_faces SET person_id = ? WHERE person_id = ?", (target_pid, old_pid))
            self.cursor.execute("DELETE FROM people WHERE person_id = ?", (old_pid,))
            self.conn.commit()
            return f"Merged 'ID {old_pid}' into existing '{new_name}' (ID {target_pid})."
        else:
            # RENAME: Just update the name
            self.cursor.execute("UPDATE people SET name = ? WHERE person_id = ?", (new_name, old_pid))
            self.conn.commit()
            return f"Renamed ID {old_pid} to {new_name}"

    # =========================================================================
    # CTR & Analytics Methods
    # =========================================================================

    def get_overall_ctr(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Get overall CTR metrics for a date range.

        Args:
            start_date: Optional start date (YYYY-MM-DD format)
            end_date: Optional end date (YYYY-MM-DD format)

        Returns:
            Dict with impressions, clicks, ctr, and unique_sessions
        """
        try:
            # Build date filter
            date_filter = ""
            params = []
            if start_date:
                date_filter += " AND i.timestamp >= ?"
                params.append(start_date)
            if end_date:
                date_filter += " AND i.timestamp <= ?"
                params.append(end_date + " 23:59:59")

            # Count impressions
            self.cursor.execute(f"""
                SELECT COUNT(*) FROM search_impressions i
                WHERE 1=1 {date_filter}
            """, params)
            impressions = self.cursor.fetchone()[0] or 0

            # Count clicks (label=1 interactions)
            self.cursor.execute(f"""
                SELECT COUNT(*) FROM search_interactions si
                WHERE si.label = 1 {date_filter.replace('i.', 'si.')}
            """, params)
            clicks = self.cursor.fetchone()[0] or 0

            # Count unique sessions
            self.cursor.execute(f"""
                SELECT COUNT(DISTINCT session_id) FROM search_impressions i
                WHERE 1=1 {date_filter}
            """, params)
            unique_sessions = self.cursor.fetchone()[0] or 0

            ctr = (clicks / impressions * 100) if impressions > 0 else 0.0

            return {
                "impressions": impressions,
                "clicks": clicks,
                "ctr": round(ctr, 2),
                "unique_sessions": unique_sessions
            }
        except Exception as e:
            logger.error(f"Failed to get overall CTR: {e}")
            return {"impressions": 0, "clicks": 0, "ctr": 0.0, "unique_sessions": 0}

    def get_ctr_by_model(self) -> Dict[str, Dict[str, Any]]:
        """
        Get CTR breakdown per ranking model.

        Returns:
            Dict mapping model name to {sessions, impressions, clicks, ctr}
        """
        try:
            query = """
                SELECT
                    sh.ranking_model,
                    COUNT(DISTINCT sh.session_id) as sessions,
                    COUNT(DISTINCT si.impression_id) as impressions,
                    SUM(CASE WHEN sint.label = 1 THEN 1 ELSE 0 END) as clicks
                FROM search_history sh
                LEFT JOIN search_impressions si ON sh.session_id = si.session_id
                LEFT JOIN search_interactions sint ON sh.session_id = sint.session_id
                GROUP BY sh.ranking_model
            """
            self.cursor.execute(query)
            rows = self.cursor.fetchall()

            results = {}
            for row in rows:
                model = row[0] or "unknown"
                sessions = row[1] or 0
                impressions = row[2] or 0
                clicks = row[3] or 0
                ctr = (clicks / impressions * 100) if impressions > 0 else 0.0

                results[model] = {
                    "sessions": sessions,
                    "impressions": impressions,
                    "clicks": clicks,
                    "ctr": round(ctr, 2)
                }

            return results
        except Exception as e:
            logger.error(f"Failed to get CTR by model: {e}")
            return {}

    def get_ctr_by_date(self, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """
        Get daily CTR for trend charts.

        Args:
            start_date: Optional start date (YYYY-MM-DD format)
            end_date: Optional end date (YYYY-MM-DD format)

        Returns:
            List of dicts with date, model, impressions, clicks, ctr
        """
        try:
            date_filter = ""
            params = []
            if start_date:
                date_filter += " AND si.timestamp >= ?"
                params.append(start_date)
            if end_date:
                date_filter += " AND si.timestamp <= ?"
                params.append(end_date + " 23:59:59")

            query = f"""
                SELECT
                    DATE(si.timestamp) as date,
                    sh.ranking_model,
                    COUNT(DISTINCT si.impression_id) as impressions,
                    SUM(CASE WHEN sint.label = 1 THEN 1 ELSE 0 END) as clicks
                FROM search_impressions si
                JOIN search_history sh ON si.session_id = sh.session_id
                LEFT JOIN search_interactions sint ON si.session_id = sint.session_id AND si.photo_id = sint.photo_id
                WHERE 1=1 {date_filter}
                GROUP BY DATE(si.timestamp), sh.ranking_model
                ORDER BY date ASC
            """
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()

            results = []
            for row in rows:
                impressions = row[2] or 0
                clicks = row[3] or 0
                ctr = (clicks / impressions * 100) if impressions > 0 else 0.0
                results.append({
                    "date": row[0],
                    "model": row[1] or "unknown",
                    "impressions": impressions,
                    "clicks": clicks,
                    "ctr": round(ctr, 2)
                })

            return results
        except Exception as e:
            logger.error(f"Failed to get CTR by date: {e}")
            return []

    def get_ctr_by_position(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get CTR at each result position for position bias analysis.

        Args:
            limit: Maximum position to analyze

        Returns:
            List of dicts with position, impressions, clicks, ctr
        """
        try:
            query = """
                SELECT
                    si.position,
                    COUNT(*) as impressions,
                    SUM(CASE WHEN sint.label = 1 THEN 1 ELSE 0 END) as clicks
                FROM search_impressions si
                LEFT JOIN search_interactions sint
                    ON si.session_id = sint.session_id
                    AND si.photo_id = sint.photo_id
                WHERE si.position >= 0 AND si.position < ?
                GROUP BY si.position
                ORDER BY si.position ASC
            """
            self.cursor.execute(query, (limit,))
            rows = self.cursor.fetchall()

            results = []
            for row in rows:
                impressions = row[1] or 0
                clicks = row[2] or 0
                ctr = (clicks / impressions * 100) if impressions > 0 else 0.0
                results.append({
                    "position": row[0],
                    "impressions": impressions,
                    "clicks": clicks,
                    "ctr": round(ctr, 2)
                })

            return results
        except Exception as e:
            logger.error(f"Failed to get CTR by position: {e}")
            return []

    def get_model_comparison_summary(self) -> Dict[str, Any]:
        """
        Get side-by-side comparison stats for portfolio display.

        Returns:
            Dict with model comparison metrics including lift percentages
        """
        try:
            model_stats = self.get_ctr_by_model()

            # Calculate average click position per model
            avg_click_pos = {}
            for model in model_stats.keys():
                self.cursor.execute("""
                    SELECT AVG(sint.position)
                    FROM search_interactions sint
                    JOIN search_history sh ON sint.session_id = sh.session_id
                    WHERE sint.label = 1 AND sint.position >= 0 AND sh.ranking_model = ?
                """, (model,))
                result = self.cursor.fetchone()
                avg_click_pos[model] = round(result[0], 1) if result[0] else 0.0

            # Add average click position to stats
            for model in model_stats:
                model_stats[model]["avg_click_position"] = avg_click_pos.get(model, 0.0)

            # Calculate lift if we have both models
            summary = {"models": model_stats}

            if "xgboost" in model_stats and "heuristic" in model_stats:
                xgb = model_stats["xgboost"]
                heur = model_stats["heuristic"]

                if heur["ctr"] > 0:
                    ctr_lift = ((xgb["ctr"] - heur["ctr"]) / heur["ctr"]) * 100
                    summary["ctr_lift"] = round(ctr_lift, 1)

                if heur["avg_click_position"] > 0:
                    pos_lift = ((heur["avg_click_position"] - xgb["avg_click_position"]) / heur["avg_click_position"]) * 100
                    summary["position_lift"] = round(pos_lift, 1)

            # Get date range
            self.cursor.execute("""
                SELECT MIN(timestamp), MAX(timestamp) FROM search_impressions
            """)
            date_range = self.cursor.fetchone()
            summary["date_range"] = {
                "start": date_range[0] if date_range[0] else None,
                "end": date_range[1] if date_range[1] else None
            }

            return summary
        except Exception as e:
            logger.error(f"Failed to get model comparison summary: {e}")
            return {"models": {}, "date_range": {"start": None, "end": None}}

