import streamlit as st
import os
import uuid
from dotenv import load_dotenv
from main import DB_CHROMA_PATH, DB_SQL_PATH
from src.db.storage import DatabaseManager
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

    # 2. Get Rich History (Full Objects)
    history_map = st.session_state.get("search_results_objects", {})
    history_metrics = st.session_state.get("search_metrics_map", {})

    if not selected_pairs and not history_map:
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
            results_list = history_map.get(session_id, [])
            metrics_map = history_metrics.get(session_id, {})

            if not results_list:
                continue

            # 1. FIND THE CUTOFF (Last Clicked Rank)
            max_index = -1
            for idx, res in enumerate(results_list):
                if res.photo_id in positive_set:
                    max_index = max(max_index, idx)

            # If no clicks in this session (shouldn't happen due to loop logic), skip
            if max_index == -1:
                continue

            # 2. ADD BUFFER (e.g., +2 rows assumed seen)
            cutoff_index = min(max_index + 3, len(results_list))
            logger.info(f"Cutoff index for storing interaction is {cutoff_index}")

            # 3. SAVE VALID ZONE
            for i in range(cutoff_index):
                res_obj = results_list[i]

                label = 1 if res_obj.photo_id in positive_set else 0
                specific_scores = metrics_map.get(res_obj.display_path, {})

                db_manager.log_interaction_from_object(
                    result=res_obj,
                    session_id=session_id,
                    label=label,
                    dynamic_scores=specific_scores
                )

                if label == 1: count_pos += 1
                else: count_neg += 1

    # Cleanup
    st.session_state.selected_pairs = set()
    st.toast(f"✅ Saved {count_pos} Positives & {count_neg} Negatives", icon="💾")
    st.sidebar.success(f"Saved {count_pos} Pos / {count_neg} Neg")