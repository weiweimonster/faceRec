import streamlit as st
import os
import cv2
import numpy as np

from typing import Optional, List, Tuple

from main import DB_CHROMA_PATH
from src.retrieval.enginev2 import SearchEngine
from src.util.search_config import SearchFilters
from src.rag.gpt_client import GPTClient
from src.util.image_util import get_exif_timestamp, get_timestamp_from_heic, load_face_crop_from_str
from src.ui.components import render_photo_card
from PIL import Image
from dotenv import load_dotenv
from src.db.storage import DatabaseManager

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

# --- Main App ---

st.title("🤖 My AI Photo Album")

# Removed "Chat" tab, focused on Core Tools
app_mode = st.sidebar.radio(
    "Navigate",
    ["🔎 Search", "🎨 Generate", "👥 Labeling"]
)

# ==========================================
# TAB 1: SEARCH ENGINE
# ==========================================
if app_mode == "🔎 Search":
    st.header("Search your Memories")
    engine = get_search_engine()
    gpt = get_gpt_client()
    query = st.text_input("Ask for anything...", placeholder="e.g. 'Ethan at the beach'", key="search_box")

    if query:
        search_filter, message = gpt._get_agent_response(query)
        if search_filter:
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
        if results:
            cols = st.columns(3)
            for i, image_analysis_result in enumerate(results):
                metric = metrics[image_analysis_result.display_path]
                with cols[i % 3]:
                    render_photo_card(image_analysis_result, metric, context_label=f"#{i + 1}")
        else:
            st.warning("No photos found.")

elif app_mode == "🎨 Generate":
    st.header("Search your Memories")
    engine = get_search_engine()
    gpt = get_gpt_client()
    query = st.text_input("Ask for anything...",
                          placeholder="e.g. 'Generate a photo of Jacob playing in New York'", key="generate_box")

    if query:
        search_filter, message = gpt._get_agent_response(query)
        if search_filter:
            st.success(f"✅ Agent Parsed: {search_filter}")
            with st.expander("🔍 See Raw Arguments", expanded=True):
                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("**Parsed Intent**")
                    if search_filter.fn_name == "search_memory":
                        st.warning("📂 Search Database")
                    elif search_filter.fn_name == "generate_image":
                        st.info("🎨 Generate Image")

                with c2:
                    st.markdown("**Extracted Filters**")
                    st.json(search_filter.to_dict())
        results, metrics = engine.searchv2(search_filter, limit=50)
        st.markdown(f"**Found {len(results)} results for:** `{query}`")
        with st.expander("Show Ranked Results", expanded=True):
            if results:
                cols = st.columns(3)
                for i, image_analysis_results in enumerate(results):
                    image_metric = metrics[image_analysis_results.display_path]
                    with cols[i % 3]:
                        render_photo_card(image_analysis_results, image_metric, context_label=f"#{i + 1}")
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
