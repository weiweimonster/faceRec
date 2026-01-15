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
from src.rank.rank_metrics import PictureRankMetrics


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
        # Use False for streamlit multithread
        self.conn = sqlite3.connect(sql_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_sql_tables()
        self.init_ltr_tables()

        # 2. Setup ChromaDB (Persistent)
        # We explicitly set the tenant/db names if needed, but defaults work fine for local
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)

        # Get or create the collection.
        # "cosine" distance is CRITICAL for normalized CLIP embeddings.
        self.vector_collection = self.chroma_client.get_or_create_collection(
            name="photo_gallery",
            metadata={"hnsw:space": "cosine"}
        )

    def init_ltr_tables(self):
        """
        Creates tables for Learning-to-Rank data collection.
        1. search_history: Maps session_id -> Query/Intent
        2. search_interactions: Maps session_id + photo_id -> Label + Feature Snapshot
        """
        # Table 1: The Search Session (Context)
        self.cursor.execute("""
                            CREATE TABLE IF NOT EXISTS search_history
                            (
                                session_id          TEXT PRIMARY KEY,
                                raw_query           TEXT,
                                parsed_filters_json TEXT,
                                timestamp           DATETIME DEFAULT CURRENT_TIMESTAMP
                            )
                            """)

        self.cursor.execute("""
                            CREATE TABLE IF NOT EXISTS search_interactions
                            (
                                interaction_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                                session_id             TEXT,
                                photo_id               TEXT,
                                label                  INTEGER  DEFAULT 0, -- 1 = Positive (Click), 0 = Negative (Skip)
                                features_snapshot_json TEXT,               -- The exact features used for ranking
                                rank_metrics_blob      BLOB,
                                timestamp              DATETIME DEFAULT CURRENT_TIMESTAMP,
                                FOREIGN KEY (session_id) REFERENCES search_history (session_id),
                                FOREIGN KEY (photo_id) REFERENCES photos (photo_id)
                            )
                            """)
        self.conn.commit()

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

        self.conn.commit()
    def log_search_query(self, session_id: str, query: str, filters: Any):
        """Saves the context (intent) of the search."""
        try:
            filters_dict = filters.to_dict() if hasattr(filters, "to_dict") else filters
            self.cursor.execute("""
                    INSERT OR REPLACE INTO search_history (session_id, raw_query, parsed_filters_json)
                    VALUES (?, ?, ?)
                """, (session_id, query, json.dumps(filters_dict)))
            self.conn.commit()
            logger.info(f"📝 Logged Session: {session_id[:8]}... | Query: {query}")
        except Exception as e:
            logger.error(f"Failed to log search query: {e}")

    def log_interaction_from_object(self, result: ImageAnalysisResult, session_id: str, label: int = 1, dynamic_scores: Dict[str, float] = None, rank_metrics: Optional[PictureRankMetrics] = None):
        """
        Saves a training example using the IN-MEMORY object (ImageAnalysisResult).
        Avoids SQL re-reads and handles 'Target Face' logic in Python.
        """
        try:
            scores = dynamic_scores or {}

            photo_metrics_blob = pickle.dumps(rank_metrics) if rank_metrics else None
            # 1. Extract Global Metrics (From the Object)
            # We use getattr/defaults to be safe against missing lazy-loaded fields
            features = {
                "g_aesthetic": result.aesthetic_score if result.aesthetic_score is not None else 0.0,
                "g_brightness": result.global_brightness if result.global_brightness is not None else 0.0,
                "g_contrast": result.global_contrast if result.global_contrast is not None else 0.0,

                "search_semantic": scores.get("semantic", 0.0),
                "search_final_score": scores.get("final_relevance", 0.0),
                "search_mmr_rank": scores.get("mmr_rank", -1),

                "meta_timestamp": result.timestamp,
                "meta_month": result.month if result.month else -1,
                "meta_iso": result.iso if result.iso else -1,
                "meta_time": result.time_period if result.time_period else "unknown"
            }

            # 2. Determine CONTEXT (Who was the target?)
            # We check the search history to see if a person was requested
            history_row = self.cursor.execute(
                "SELECT parsed_filters_json FROM search_history WHERE session_id = ?",
                (session_id,)
            ).fetchone()

            target_person_name = None
            if history_row and history_row[0]:
                try:
                    filters = json.loads(history_row[0])
                    if filters.get("people") and len(filters["people"]) > 0:
                        target_person_name = filters["people"][0]
                except:
                    pass

            # 3. Select the Relevant Face (Python Logic)
            selected_face = None

            if result.faces:
                if target_person_name:
                    # Strategy A: Find specific person
                    for face in result.faces:
                        # Case-insensitive match is safer
                        if face.name and face.name.lower() == target_person_name.lower():
                            logger.info(f"Found face: {face.name} to save into interaction database")
                            selected_face = face
                            break

                # Strategy B: Fallback to largest face if no target found
                if not selected_face:
                    # Sort by area (width * height)
                    logger.error(f"Failed to fine face to save to database for LTR")

            # 4. Merge Face Metrics
            if selected_face:
                features.update({
                    "f_blur": selected_face.blur_score if selected_face.blur_score is not None else -1,
                    "f_yaw": selected_face.yaw if selected_face.yaw is not None else 0,
                    "f_pitch": selected_face.pitch if selected_face.pitch is not None else 0,
                    "f_roll": selected_face.roll if selected_face.roll is not None else 0,
                    "f_conf": selected_face.confidence if selected_face.confidence is not None else 0.0,
                    "has_face": 1
                })
            else:
                features.update({
                    "f_blur": -1, "f_yaw": 0, "f_pitch": 0, "f_roll": 0,
                    "f_conf": 0.0, # Zero confidence for no face
                    "has_face": 0
                })

            # 5. Save Snapshot
            self.cursor.execute("""
                                INSERT INTO search_interactions (session_id, photo_id, label, features_snapshot_json, rank_metrics_blob)
                                VALUES (?, ?, ?, ?, ?)
                                """, (session_id, result.photo_id, label, json.dumps(features), photo_metrics_blob))

            self.conn.commit()

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
                                    aesthetic_score, iso, global_blur, global_brightness, global_contrast
                                )
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    photo_id, original_path, display_path, file_hash,
                                    result.original_width, result.original_height,
                                    result.timestamp, result.month, result.time_period, # UPDATED
                                    result.aesthetic_score, result.iso, result.global_blur,
                                    result.global_brightness, result.global_contrast
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

            self.conn.commit()

            return photo_id

        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ Database Transaction Failed for {original_path}: {e}")
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

    def get_semantic_candidates(self, query_vector: List[float], allowed_paths: List[str], limit: int) -> Dict[str, float]:
        # Note: Directly taking query_vector to prevent loading heavy model like CLIP
        logger.info("Performing Vector Search with Chroma")

        search_params = {
            "query_embeddings": [query_vector],
            "n_results": limit,
            "include": ["metadatas", "distances", "embeddings"]
        }

        if allowed_paths:
            search_params["where"] = {"path": {"$in": allowed_paths}}

        try:
            results = self.vector_collection.query(**search_params)
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

        # if rows and len(rows) > 0:
            # TODO: Remove this after debugging
            # logger.info(f"Debug: first row of metadata fetching: {rows[0]}")

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

