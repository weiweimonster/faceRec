import streamlit as st
import os
import uuid
from main import DB_CHROMA_PATH
from src.retrieval.enginev2 import SearchEngine
from src.util.search_config import SearchFilters
from src.rag.gpt_client import GPTClient
from src.util.image_util import load_face_crop_from_str
from src.ui.components import render_photo_card
from dotenv import load_dotenv
from src.db.storage import DatabaseManager
from src.util.logger import logger

# --- Configuration ---
DB_SQL_PATH: str = ".db/sqlite/photos.db"


st.set_page_config(layout="wide", page_title="AI Photo Manager")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# --- Cache & Database ---
@st.cache_resource
def get_db():
    """Single point of entry for all DB operations"""
    return DatabaseManager(sql_path=DB_SQL_PATH, chroma_path=DB_CHROMA_PATH)

@st.cache_resource
def get_search_engine():
    db = get_db()
    return SearchEngine(db)

@st.cache_resource
def get_gpt_client():
    return GPTClient(api_key)

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

                # --- CALL THE NEW DB FUNCTION ---
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

# --- Main App ---

st.title("🤖 My AI Photo Album")

# Removed "Chat" tab, focused on Core Tools
app_mode = st.sidebar.radio(
    "Navigate",
    ["🔎 Search", "👥 Labeling"]
)

show_raw_db = st.sidebar.toggle("Show Raw Database Metrics", value=False)
engine = get_search_engine()
# Generate a default session id when started
if "search_session_id" not in st.session_state:
    st.session_state.search_session_id = str(uuid.uuid4())

if app_mode in ["🔎 Search"]:
    st.sidebar.divider()

    st.sidebar.markdown("### ⚙️ Ranking Engine")
    strategy_choice = st.sidebar.radio(
        "Ranking Model:",
        options=["Smart AI (XGBoost)", "Classic (Heuristic)"],
        index=0, # Default to Smart
        help="Switch between the Machine Learning ranker and the manual logic."
    )

    if "XGBoost" in strategy_choice:
        engine.set_ranking_strategy("xgboost")
        st.sidebar.caption("✅ Using ML Model (v0.72)")
    else:
        engine.set_ranking_strategy("heuristic")
        st.sidebar.caption("⚠️ Using Manual Rules")

    st.sidebar.markdown("### 🧠 LTR Training")

    # Show count of currently selected items
    current_selection = st.session_state.get("selected_pairs", set())
    st.sidebar.caption(f"Selected: {len(current_selection)} photos")

    if st.sidebar.button("💾 Save Feedback", type="primary"):
        db = get_db()
        save_feedback_batch(db)

# ==========================================
# TAB 1: SEARCH ENGINE
# ==========================================
if app_mode == "🔎 Search":
    st.header("Search your Memories")
    engine = get_search_engine()
    gpt = get_gpt_client()
    db = get_db()

    with st.form("gen_form"):
        query = st.text_input("Ask for anything...", placeholder="e.g. 'Generate a photo of Jacob playing'")
        submitted = st.form_submit_button("Generate", type="primary")

    if submitted and query:
        # Generate a new session id
        st.session_state.search_session_id = str(uuid.uuid4())
        current_sess_id = st.session_state.search_session_id

        search_filter, message = gpt._get_agent_response(query)
        if search_filter:

            db.log_search_query(current_sess_id, query, search_filter)
            st.success(f"✅ Agent Parsed: {search_filter}")
            with st.expander("🔍 See Raw Arguments", expanded=True):
                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("**Parsed Intent**")
                    if search_filter.fn_name == "search_memory":
                        st.info("📂 Search Database")
                    elif search_filter.fn_name == "generate_image":
                        st.warning("🎨 Generate Image")

                with c2:
                    st.markdown("**Extracted Filters**")
                    st.json(search_filter.to_dict())

        results, metrics = engine.searchv2(search_filter, limit=100)
        st.markdown(f"**Found {len(results)} results for:** `{query}`")

        if "search_results_objects" not in st.session_state:
            st.session_state.search_results_objects = {}
        st.session_state.search_results_objects[current_sess_id] = results

        if "search_metrics_map" not in st.session_state:
            st.session_state.search_metrics_map = {}
        st.session_state.search_metrics_map[current_sess_id] = metrics

        if results:
            cols = st.columns(3)
            current_sess_id = st.session_state.search_session_id
            for i, image_analysis_result in enumerate(results):
                metric = metrics[image_analysis_result.display_path]
                with cols[i % 3]:
                    render_photo_card(
                        image_analysis_result,
                        metric,
                        context_label=f"#{i + 1}",
                        show_raw=show_raw_db,
                        session_id=current_sess_id # --- PASS SESSION ID ---
                    )
        else:
            st.warning("No photos found.")



# ==========================================
# TAB 2: LABELING & INSPECTION (UPDATED)
# ==========================================
elif app_mode == "👥 Labeling":
    # 1. Fetch Clusters
    # 1. Initialize DB Manager (Cached)
    db = get_db()

    # 2. Fetch Clusters (Efficient single-query via Manager)
    # Assumes db.get_people_clusters() returns a list of dictionaries as discussed
    people_rows = db.get_people_clusters()

    if not people_rows:
        st.warning("No clusters found. Please run ingestion first.")
    else:
        # --- Sidebar ---
        st.sidebar.header(f"Found {len(people_rows)} People")

        # Default selection
        if 'selected_pid' not in st.session_state:
            st.session_state.selected_pid = people_rows[0]['id']

        # Render Sidebar List
        for person in people_rows:
            pid = person['id']
            name = person['name']
            path = person['face_path']
            bbox_str = person['bbox']

            with st.sidebar.container():
                cols = st.columns([1, 2])

                # Load Face Crop
                crop = load_face_crop_from_str_streamlit(path, bbox_str)
                if crop is not None:
                    cols[0].image(crop, width="stretch")
                else:
                    cols[0].error("?")

                # Name Input
                new_name = cols[1].text_input("Name", value=name, key=f"input_{pid}", label_visibility="collapsed")

                # Handle Rename / Merge
                if new_name != name:
                    # db.merge_identity handles the SQL transaction safely
                    status_msg = db.merge_identity(pid, new_name)
                    st.toast(status_msg)
                    st.rerun()

                # Selection Button
                if cols[1].button("Select", key=f"btn_{pid}"):
                    st.session_state.selected_pid = pid

        # --- Main Gallery ---
        selected_pid = st.session_state.selected_pid

        # Get the current name safely from our pre-fetched list
        # This avoids a separate SQL query just to get the name
        current_person = next((p for p in people_rows if p['id'] == selected_pid), None)

        if current_person:
            current_name = current_person['name']
            st.subheader(f"📸 Photos of {current_name}")

            # REUSE THE SEARCH PIPELINE
            # 1. Create a filter for this person
            filters = SearchFilters(
                is_person_search=True,
                people=[current_name],
                fn_name="you shouldn't need this",
            )

            # 2. Get Candidates (Fast SQL filtering)
            candidate_paths = db.get_candidate_path(filters)

            if candidate_paths:
                # 3. Hydrate Metadata (Get everything: blur, box, yaw, etc.)
                photos = db.fetch_metadata_batch(candidate_paths, fields="all")

                # 4. Render Grid
                cols = st.columns(3)
                for idx, photo_obj in enumerate(photos):
                    with cols[idx % 3]:
                        # Pass the clean object to your renderer
                        render_photo_card(photo_obj)
            else:
                st.info(f"No photos found for {current_name} (Cluster might be empty).")
        else:
            st.warning("Selected person ID not found. They may have been merged.")
