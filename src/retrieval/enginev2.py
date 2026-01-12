from __future__ import annotations
import clip
import torch
from typing import List, Dict, Any, Tuple

from src.common.types import ImageAnalysisResult
from src.db.storage import DatabaseManager
from src.util.search_config import SearchFilters, Pose
from src.util.logger import logger
from src.rank.heuristic_ranker import HeuristicStrategy
from src.rank.ranker import SearchResultRanker
from src.rank.xgboost_ranker import XGBoostRanker

class SearchEngine:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"✅ Search Engine Ready on {self.device}")
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.ranker = SearchResultRanker(strategy=XGBoostRanker())

    def _encode_text(self, text: str) -> List[float]:
        text_input = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            query_vec = self.model.encode_text(text_input)
            query_vec /= query_vec.norm(dim=-1, keepdim=True)
            return query_vec.cpu().numpy().flatten().tolist()

    def set_ranking_strategy(self, strategy_type: str):
        if strategy_type.lower() == "xgboost":
            self.ranker.set_strategy(XGBoostRanker())
        elif strategy_type.lower() == "heuristic":
            self.ranker.set_strategy(HeuristicStrategy())
        else:
            logger.error(f"Invalid strategy type: {strategy_type}, keeping current settings")

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

        # rank() returns (List[ImageAnalysisResult], self.metrics)
        # It also internalizes the MMR logic we discussed
        final_ranked_results, metrics = self.ranker.process(
            results=hydrated_results,
            semantic_scores=semantic_scores_only,
            target_name=target_person,
            lambda_param=0.7,
            top_k=limit,
            pose=filters.pose,
        )

        logger.info(f"✅ Search complete. Returning top {len(final_ranked_results)} results.")
        return final_ranked_results, metrics