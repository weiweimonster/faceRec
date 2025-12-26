from __future__ import annotations
import chromadb
import sqlite3
import clip
import torch
import numpy as np
from typing import List, Optional, Tuple, Any, Dict

class SearchEngine:
    def __init__(self, sql_path: str, chroma_path: str):
        """
        Initializes the Hybrid Search Engine
        :param sql_path: Path to the SQLite database
        :param chroma_path: Path to the ChromaDB folder
        """
        self.sql_path = sql_path

        # Connect to ChromaDB (Semantic Search)
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_collection("photo_gallery")

        # Load CLIP Model (Text -> Vector)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device} for Search Engine")
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        print("✅ Search Engine Ready.")

    def _get_filtered_paths(self, filters: Dict[str, Any]) -> List[str]:
        """
        Constructs a dynamic SQL query to find file paths matching All filters
        :param filters:
        :return:
        """
        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()

        # Basic Query: Join Photos, Faces, and People
        query_parts = ["""
            SELECT DISTINCT ph.display_path
            FROM photo_faces pf
            JOIN photos ph ON pf.photo_id = ph.photo_id
            LEFT JOIN people p ON pf.person_id = p.person_id
            WHERE 1=1
        """]
        params = []
        # Person Name Filter
        if filters.get("person"):
            query_parts.append("AND p.name LIKE ?")
            params.append(f"%{filters['person']}%")

        # 2. Pose Filter (Side-Left, Front, etc.)
        if filters.get("pose"):
            # We map 'side' generic term to specific DB values if needed
            pose = filters["pose"].lower()
            if pose == "side":
                query_parts.append("AND (pf.pose = 'Side-Left' OR pf.pose = 'Side-Right')")
            elif pose == "front":
                query_parts.append("AND pf.pose = 'Front'")
            else:
                query_parts.append("AND pf.pose = ?")
                params.append(filters["pose"])

        # 3. Shot Type (Close-up, Full-Body)
        if filters.get("shot_type"):
            query_parts.append("AND pf.shot_type = ?")
            params.append(filters["shot_type"])

        # 4. Emotion (Happy, Neutral)
        if filters.get("emotion"):
            query_parts.append("AND pf.emotion = ?")
            params.append(filters["emotion"])

        # 5. Date Range (Optional - Example)
        if filters.get("year"):
           query_parts.append("AND strftime('%Y', ph.timestamp) = ?")
           params.append(str(filters["year"]))

        full_query = " ".join(query_parts)
        cursor.execute(full_query, tuple(params))
        paths = [row[0] for row in cursor.fetchall()]
        conn.close()
        return paths

    def _get_person_id_from_name(self, name_query: str) -> Optional[int]:
        """
        Scans the query string to see if it mentions a known person.
        :param name_query:
        :return: ID for the person
        """
        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()

        # Fetch all known names from the database
        # TODO: use openai API to retrieve the name from the name_query and fall back to hard coded logic
        cursor.execute("SELECT person_id, name FROM people")
        rows = cursor.fetchall()
        conn.close()

        query_lower = name_query.lower()

        # Check if any database name appears in the user's query
        for pid, db_name in rows:
            # Ignore "Unknown" or empty names to avoid false positives
            if db_name and db_name.lower() != "unknown":
                if db_name.lower() in query_lower:
                    return pid
        print(f"No person found for query: {name_query}")
        return None
    def _get_photo_paths_for_person(self, person_id: int) -> List[str]:
        """
        Returns a list of all photo_ids {UUIDs) that contain this person.
        """
        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()

        # We need a JOIN to get the path from the photos table
        cursor.execute("""
                       SELECT DISTINCT ph.display_path
                       FROM photo_faces pf
                                JOIN photos ph ON pf.photo_id = ph.photo_id
                       WHERE pf.person_id = ?
                       """, (person_id,))

        # Extract the list of IDs
        paths: List[str] = [row[0] for row in cursor.fetchall()]
        conn.close()
        return paths

    def search(self, text_query: str, filters: Dict[str, Any] = None, limit: int = 20) -> List[str]:
        """
        Perform Hybrid Search:
        1. SQL: Narrow down candidates based on hard constraints
        2. Chroma: Rank the remaining candidates by semantic similarity
        :param text_query:
        :param limit:
        :return:
        """
        if filters is None: filters = {}

        # STEP 1: Get Candidates from SQL (Metadata Filter)
        # If we have ANY metadata filters, we must run SQL first.
        candidate_paths = []
        has_filters = bool(filters.get("person") or filters.get("pose") or filters.get("shot_type") or filters.get("emotion"))

        if has_filters:
            print(f"Filter detected, running sql search with filters: {filters}")
            candidate_paths = self._get_filtered_paths(filters)

            if not candidate_paths:
                print("❌ No photos match the strict metadata filters.")
                return []

            print(f"📉 Filter reduced search space to {len(candidate_paths)} photos.")
        # Detect Person Filter (SQL)
        # person_id: Optional[int] = self._get_person_id_from_name(text_query)
        # filter_paths: Optional[List[str]] = None

        # STEP 2: Semantic Search (Vector)
        # We only run CLIP if there is a text query
        if not text_query.strip():
            # If user just asked for "Photos of Ethan" (no semantic detail), return SQL results directly
            return candidate_paths[:limit]

        # Embed the query
        text_inputs = clip.tokenize([text_query]).to(self.device)
        with torch.no_grad():
            query_vec = self.model.encode_text(text_inputs)
            query_vec /= query_vec.norm(dim=-1, keepdim=True)
            query_list = query_vec.cpu().numpy().flatten().tolist()

        # Construct Chroma Query
        search_params = {
            "query_embeddings": [query_list],
            "n_results": limit,
            "include": ["metadatas", "distances"]
        }

        # CRITICAL: Restrict Chroma search to only the paths found by SQL
        if has_filters:
            # Chroma "$in" operator lets us whitelist specific paths
            # Note: If list is massive (>2000), this might be slow, but for personal albums it's fine.
            search_params["where"] = {"path": {"$in": candidate_paths}}

        # if person_id is not None:
        #     print(f"🔎 Detected person filter for ID: {person_id}")
        #     filter_paths = self._get_photo_paths_for_person(person_id)
        #
        #     # Optimization: If we know this person has 0 photos, stop immediately
        #     if not filter_paths:
        #         print("❌ This person has no photos yet.")
        #         return []
        #
        # # Tokenize the text (truncate to 77 tokens max)
        # text_inputs = clip.tokenize([text_query]).to(self.device)
        #
        # with torch.no_grad():
        #     # Encode text
        #     query_vector_tensor = self.model.encode_text(text_inputs)
        #
        #     # Normalize the vector (Critical for Cosine Similarity)
        #     query_vector_tensor /= query_vector_tensor.norm(dim=-1, keepdim=True)
        #
        #     # Convert to standard Python list for Chroma
        #     query_vector: List[float] = query_vector_tensor.cpu().numpy().flatten().tolist()
        #
        # # Chroma logic: If we found a person, we only look at THEIR photos.
        # # If no person found, we look at ALL photos
        # search_params = {
        #     "query_embeddings": [query_vector],
        #     "n_results": limit,
        #     "include": ["metadatas", "distances"]
        # }
        #
        # # Apply the ID filter if it exists
        # if filter_paths is not None:
        #     # print(f"Filtering by person IDs: {filter_paths}")
        #     search_params["where"] = {"path": {"$in": filter_paths}}

        try:
            print(f"Running Chroma Search with params")
            results = self.collection.query(**search_params)
        except Exception as e:
            print(f"⚠️ Chroma Search Error: {e}")
            return []

        # Chroma returns a lit of lists (because you can query multiple vectors at once).
        # We only sent one vector, so we take index [0].

        final_paths: List[str] = []

        if results['metadatas'] and len(results['metadatas']) > 0:
            found_metadatas = results['metadatas'][0]

            for meta in found_metadatas:
                # 'path' was stored in metadata during ingestion
                path = meta.get('path')
                if path:
                    final_paths.append(path)
        return final_paths
