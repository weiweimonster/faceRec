from __future__ import annotations
import chromadb
import sqlite3
import clip
import torch
from typing import List, Dict, Any, Tuple
from datetime import datetime
import numpy as np

from src.common.types import ImageAnalysisResult
from src.db.storage import DatabaseManager
from src.retrieval.search_results import SearchResultRanker
from src.util.search_config import SearchFilters, Pose
from src.util.logger import logger

class SearchEngine:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"✅ Search Engine Ready on {self.device}")
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)

    def _encode_text(self, text: str) -> List[float]:
        text_input = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            query_vec = self.model.encode_text(text_input)
            query_vec /= query_vec.norm(dim=-1, keepdim=True)
            return query_vec.cpu().numpy().flatten().tolist()


    def searchv2(self, filters: SearchFilters, limit: int = 20, rank: bool = True) -> Tuple[List[ImageAnalysisResult], Dict[str, Any]]:
        """
        Search pipeline: SQL candidates -> CLIP Encoding -> Chroma Similarity -> Hydration -> Ranker.
        """
        if filters is None:
            logger.error("No filters found. Returning empty.")
            return [], {}

        # 1. SQL Candidate Filtering
        # Filters by People, Year, and Pose to narrow the search space
        logger.info("Performing SQL candidate search")
        candidate_paths = self.db.get_candidate_path(filters)
        logger.info(f"Retrieved {len(candidate_paths)} from SQL query")

        # If filters were provided but no candidates found, exit early
        if not candidate_paths and (filters.people or filters.pose or filters.year):
            logger.info("No candidates found matching SQL filters. Returning empty.")
            return [], {}

        # 2. Text Encoding (Logic resides in SearchEngine)
        if filters.semantic_query:
            query_vector = self._encode_text(filters.semantic_query)
        else:
            logger.error("No semantic query found. Returning empty.")
            return [], {}

        # 3. Vector Similarity Search
        # Note: get_semantic_candidates returns a Dict[path, score]
        logger.info("Performing ChromaDB semantic search")
        semantic_data = self.db.get_semantic_candidates(
            query_vector=query_vector,
            allowed_paths=candidate_paths,
            limit=limit * 3  # Fetch a larger pool for the ranker to filter/diversify
        )

        if not semantic_data:
            return [], {}

        # 4. Metadata Hydration
        # We need the full ImageAnalysisResult objects for the Ranker to work
        ordered_paths = list(semantic_data.keys())
        logger.info(f"Hydrating metadata for {len(ordered_paths)} candidates")

        # Using your 'all' implementation
        hydrated_results = self.db.fetch_metadata_batch(ordered_paths, fields="all")

        # 5. Smart Ranking & MMR Diversity
        # We need to identify the target person for the 'Quality Boost' logic
        target_person = filters.people[0] if (filters.people and len(filters.people) > 0) else None

        logger.info(f"Initializing Ranker. Target person for quality boost: {target_person}")

        semantic_scores_only = {}

        # TODO: Move this into a private function
        for res in hydrated_results:
            if res.display_path in semantic_data:
                score, vector = semantic_data[res.display_path]

                # A. Populate the Dict (for Ranker's lookup if needed)
                semantic_scores_only[res.display_path] = score

                # B. INJECT VECTOR (Fixes the MMR Crash)
                res.semantic_vector = vector

        ranker = SearchResultRanker(hydrated_results, semantic_scores_only)

        # rank() returns (List[ImageAnalysisResult], self.metrics)
        # It also internalizes the MMR logic we discussed
        final_ranked_results, metrics = ranker.rank(
            target_name=target_person,
            lambda_param=0.7,
            top_k=limit
        )

        logger.info(f"✅ Search complete. Returning top {len(final_ranked_results)} results.")
        return final_ranked_results, metrics

    def search_for_generate_v2(self, filters: SearchFilters, limit: int = 20, metadata: List[str] = None) -> Tuple[List[str], Dict[Any, Any]]:
        if filters is None: filters = {}
        if metadata is None: metadata = []

        logger.info("Performing SQL search")
        candidate_paths = self._sql_search(filters)

        logger.info(f"Querying SQL for metadata: {metadata}")
        # Path -> Name -> Data
        metadata_map = self.fetch_metadata_batch(candidate_paths, metadata)

        logger.info("Performing Chroma search")
        raw_results = self._vector_search(filters, candidate_paths, limit * 2)

        logger.info("Re-ranking results")
        # score: Path -> Score
        final_results, score = self.re_ranking_v2(raw_results)

        # Combine score and metadata_map
        logger.info(f"Combining metadata with score")
        for k, v in score.items():
            # k: Path, v: Score
            if k in metadata_map:
                logger.debug(f"Adding final score for {k} to metadata map")
                metadata_map[k]['photo']['final_score'] = v
            else:
                logger.warning(f"Metadata not found for {k}, skipping......")
        logger.info(f"Returning top {limit} results")
        return final_results[:limit], metadata_map

    def re_ranking_v2(self, raw_results: List[Dict]) -> Tuple[List[str], Dict[Any, Any]]:
        raw_results.sort(key=lambda x: x['semantic_score'], reverse=True)

        score_metadata = {}
        for output in raw_results:
            score_metadata[output['path']] = output['semantic_score']

        final_results = [output['path'] for output in raw_results]
        return final_results, score_metadata

    def _sql_search(self, filters: SearchFilters) -> List[str]:
        """
        Constructs a query to find photos containing ALL 'people'
        AND matching ALL 'filters'.
        """
        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()


        params = []

        # Base Join
        query = """
                SELECT DISTINCT ph.display_path
                FROM photo_faces pf
                         JOIN photos ph ON pf.photo_id = ph.photo_id
                         LEFT JOIN people p ON pf.person_id = p.person_id
                WHERE 1 = 1
                """

        if not filters.is_person_search:
            logger.info("⚡ Mode: Scene Search (Targeting all photos)")
            query = "SELECT display_path FROM photos WHERE 1=1"

        # ---------------------------------------------------------
        # 1. POSE FILTER (Enum Logic)
        # ---------------------------------------------------------
        if filters.pose:
            logger.info(f"Pose Filter added to SQL Query: {str(filters.pose)}")
            query += " AND pf.pose LIKE ?"
            params.append(str(filters.pose))


        # ---------------------------------------------------------
        # 3. YEAR FILTER
        # ---------------------------------------------------------
        if filters.year:
            logger.info(f"Year Filter added to SQL Query: {str(filters.year)}")
            if filters.is_person_search:
                query += " AND strftime('%Y', ph.timestamp) = ?"
                params.append(str(filters.year))
            else:
                query += " AND strftime('%Y', timestamp) = ?"
                params.append(str(filters.year))

        # ---------------------------------------------------------
        # 4. PEOPLE FILTER (The "Intersection" Logic)
        # ---------------------------------------------------------
        # Logic: "Find rows matching Name A OR Name B, then group by Photo
        # and keep only groups where Count(Distinct Names) == Total Target People"
        if filters.people:
            logger.info("People filter detected, parsing names.....")
            name_conditions = []
            for name in filters.people:
                # Using LIKE for partial matches (e.g. "Jacob" matches "Jacob S")
                logger.info(f"    Adding {name} to SQL Query")
                name_conditions.append("p.name LIKE ?")
                params.append(f"%{name}%")

            # Add (Name1 OR Name2 OR ...)
            query += " AND (" + " OR ".join(name_conditions) + ")"

            # The Group By enforces that ALL people must be present
            query += " GROUP BY ph.photo_id HAVING COUNT(DISTINCT p.name) >= ?"
            params.append(len(filters.people))

        try:
            logger.info(f"Executing SQL Query: {query} with params: {params}")
            cursor.execute(query, tuple(params))
            paths = [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"SQL Error: {e}")
            paths = []
        conn.close()
        return paths

    def _vector_search(self, filters: SearchFilters, allowed_paths: List[str], limit: int) -> List[Dict]:
        """
        Returns [{'path': str, 'semantic_score': float}]
        """
        text_query = ""
        if filters.semantic_query:
            text_query = filters.semantic_query
        logger.info(f"Tokenizing query: {text_query}")

        text_inputs = clip.tokenize([text_query]).to(self.device)

        with torch.no_grad():
            query_vec = self.model.encode_text(text_inputs)
            query_vec /= query_vec.norm(dim=-1, keepdim=True)
            query_list = query_vec.cpu().numpy().flatten().tolist()

        search_params = {
            "query_embeddings": [query_list],
            "n_results": limit,
            "include": ["metadatas", "distances"]
        }

        if allowed_paths:
            # Chroma "where" clause limits search space
            search_params["where"] = {"path": {"$in": allowed_paths}}
        try:
            logger.info(f"Running Chroma Search with params")
            results = self.collection.query(**search_params)
        except:
            return []

        output = []
        if results['metadatas']:
            for i, meta in enumerate(results['metadatas'][0]):
                path = meta.get('path')
                dist = results['distances'][0][i]
                # Convert Cosine Distance (0..2) to Similarity (0..1)
                # Sim = 1 - (dist / 2) approx for aligned vectors, or just max(0, 1 - dist)
                sim = max(0.0, 1.0 - dist)
                output.append({"path": path, "semantic_score": sim})
        return output

    def fetch_metadata_batch(self, paths: List[str], fields: List[str] = None) -> dict:
        """
        Efficiently fetches metadata for a specific list of photos
        handles the join logic safely so you don't crash on Scene queries
        :param paths:
        :param fields:
        :return:
        """
        if not paths or not fields: return {}
        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()

        # Map requested fields to their actual SQL columns
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
            "photo_id": "ph.photo_id"
        }

        photo_metadata = [
            "timestamp", "original_path", "photo_id"
        ]

        # We ALWAYS fetch display_path (index 0) and p.name (index 1) for structure keys
        base_cols = ["ph.display_path", "p.name"]

        requested_sql_cols = []
        final_field_names = []
        for f in fields:
            if f in valid_map:
                requested_sql_cols.append(valid_map[f])
                final_field_names.append(f)
            else:
                logger.warning(f"Invalid metadata field requested: {f}")

        if not requested_sql_cols:
            logger.warning("No valid metadata fields requested, returning empty dict.")
            conn.close()
            return {}

        all_selected_cols = base_cols + requested_sql_cols
        cols_str = ", ".join(all_selected_cols)
        placeholders = ",".join(["?"] * len(paths))

        # Note: We use LEFT JOIN to ensure we get data even if the person is 'Unknown' (NULL)
        query = f"""
                SELECT {cols_str}
                FROM photos ph
                LEFT JOIN photo_faces pf ON ph.photo_id = pf.photo_id
                LEFT JOIN people p ON pf.person_id = p.person_id
                WHERE ph.display_path IN ({placeholders})
            """

        try:
            cursor.execute(query, paths)
            results = cursor.fetchall()
        except Exception as e:
            logger.error(f"Metadata Fetch Error: {e}")
            conn.close()
            return {}

        # Transform to Dict -> Name -> Data
        metadata_map = {}
        for row in results:
            path = row[0]
            raw_name = row[1] if row[1] else "Unknown"

            # The rest of the columns correspond exactly to final_field_names
            # row[0] is path, row[1] is name, row[2] is first requested field...
            data_values = row[2:]
            item_data = {}
            photo_data = {}
            # Create dict: {"yaw"; 12.3, "blur": 50.....
            for i in range(len(data_values)):
                if final_field_names[i] in photo_metadata:
                    photo_data[final_field_names[i]] = data_values[i]
                else:
                    item_data[final_field_names[i]] = data_values[i]

            # item_data = {final_field_names[i]: data_values[i] for i in range(len(data_values)) if final_field_names[i] not in photo_metadata}

            if path not in metadata_map:
                metadata_map[path] = {}
                # Insert photo metadata only once when initialized
                metadata_map[path]['photo'] = photo_data

            # Collision Handling:
            # If "Unknown" already exists for this photo, rename to "Unknown_2", etc.
            unique_name = raw_name
            counter = 2
            while unique_name in metadata_map[path]:
                unique_name = f"{raw_name}_{counter}"
                counter += 1

            metadata_map[path][unique_name] = item_data

        conn.close()
        return metadata_map