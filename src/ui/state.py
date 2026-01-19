import streamlit as st
import os
import uuid
from dotenv import load_dotenv
from main import DB_CHROMA_PATH, DB_SQL_PATH
from src.db.storage import DatabaseManager
from src.rank.base import RankingResult
from src.retrieval.enginev2 import SearchEngine
from src.util.image_util import load_face_crop_from_str
from src.rag.gpt_client import GPTClient
from src.util.logger import logger

def init_session_state():
    """Initialize essential session variables."""
    if "search_session_id" not in st.session_state:
        st.session_state.search_session_id = str(uuid.uuid4())
    if "selected_pairs" not in st.session_state:
        st.session_state.selected_pairs = set()

@st.cache_resource
def get_db():
    return DatabaseManager(sql_path=DB_SQL_PATH, chroma_path=DB_CHROMA_PATH)

@st.cache_resource
def get_search_engine():
    db = get_db()
    return SearchEngine(db)

@st.cache_resource
def get_gpt_client():
    load_dotenv()
    return GPTClient(os.getenv("OPENAI_API_KEY"))

@st.cache_resource
def load_face_crop_from_str_streamlit(image_path: str, bbox_str: str):
    return load_face_crop_from_str(image_path, bbox_str)

def save_feedback_batch(db_manager: DatabaseManager):
    """
    Commits selections using 'Max Rank Cutoff' logic.
    Passes full ImageAnalysisResult objects to the DB manager.
    """
    # 1. Get Selections
    selected_pairs = st.session_state.get("selected_pairs", set())

    if not selected_pairs:
        st.warning("No selections to save")
        return

    # 2. Get Rich History (Full Objects)
    history_rank_results = st.session_state.get("search_ranking_results", {})

    if not selected_pairs and not history_rank_results:
        st.sidebar.warning("No data to save.")
        return

    # Group clicks by session for processing
    selections_by_session = {}
    for pid, sid in selected_pairs:
        if sid not in selections_by_session:
            selections_by_session[sid] = set()
        selections_by_session[sid].add(pid)

    count_pos = 0
    count_neg = 0

    with st.spinner("Saving training data..."):
        for session_id, positive_set in selections_by_session.items():

            # This returns List[ImageAnalysisResult]
            ranking_results: RankingResult = history_rank_results.get(session_id, [])
            # logger.info(str(photo_rank_metrics_map))

            if not ranking_results:
                continue

            # 1. FIND THE CUTOFF (Last Clicked Rank)
            max_index = -1
            for idx, res in enumerate(ranking_results.ranked_results):
                if res.photo_id in positive_set:
                    max_index = max(max_index, idx)

            # If no clicks in this session (shouldn't happen due to loop logic), skip
            if max_index == -1:
                continue

            # 2. ADD BUFFER (e.g., +2 rows assumed seen)
            cutoff_index = min(max_index + 3, len(ranking_results.ranked_results))
            logger.info(f"Cutoff index for storing interaction is {cutoff_index}")

            # 3. SAVE VALID ZONE
            for i in range(cutoff_index):
                res_obj = ranking_results.ranked_results[i]

                label = 1 if res_obj.photo_id in positive_set else 0
                features = ranking_results.training_features.get(res_obj.display_path, {})
                db_manager.log_interaction_from_features(res_obj, session_id, features, label)

                if label == 1: count_pos += 1
                else: count_neg += 1

    # Cleanup
    st.session_state.selected_pairs = set()
    st.toast(f"✅ Saved {count_pos} Positives & {count_neg} Negatives", icon="💾")
    st.sidebar.success(f"Saved {count_pos} Pos / {count_neg} Neg")