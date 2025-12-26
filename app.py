import streamlit as st
import sqlite3
import os
import cv2
import numpy as np
import json
from typing import Optional, List, Tuple, Any
from src.retrieval.engine import SearchEngine
from src.rag.bot import PhotoAgent

# --- Configuration ---
DB_SQL_PATH: str = ".db/sqlite/photos.db"  # Updated to match your path structure

st.set_page_config(layout="wide", page_title="AI Photo Manager")


# This caches the model so it doesn't reload on every interaction
@st.cache_resource
def get_search_engine():
    # Only loads CLIP once!
    return SearchEngine(".db/sqlite/photos.db", "db/chroma")


# --- Helper Functions ---

def get_db_connection() -> sqlite3.Connection:
    """
    Establishes a connection to the SQLite database.
    check_same_thread=False is required for Streamlit's threading model.
    """
    return sqlite3.connect(DB_SQL_PATH, check_same_thread=False)

@st.cache_data
def load_face_crop(image_path: str, bbox_json: str) -> Optional[np.ndarray]:
    try:
        if not os.path.exists(image_path):
            return None

        bbox: List[float] = json.loads(bbox_json)

        # Convert [x1, y1, x2, y2] to ints
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = img_rgb.shape

        # Add 20% padding
        pad_x = int((x2 - x1) * 0.2)
        pad_y = int((y2 - y1) * 0.2)

        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w_img, x2 + pad_x)
        crop_y2 = min(h_img, y2 + pad_y)

        # Perform the crop
        crop = img_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
        return crop

    except Exception as e:
        # print(f"Error loading crop: {e}")
        return None


def merge_identities(cursor: sqlite3.Cursor, old_pid: int, new_name: str) -> None:
    """
    Renames a cluster. If the name already exists, merges the clusters.
    """
    # 1. Check if the target name already exists
    cursor.execute("SELECT person_id FROM people WHERE name = ? AND person_id != ?", (new_name, old_pid))
    row = cursor.fetchone()

    if row:
        target_pid = row[0]
        st.toast(f"Merging 'Person {old_pid}' into existing '{new_name}' (ID {target_pid})...")

        # A. Move all faces to the target ID
        cursor.execute("UPDATE photo_faces SET person_id = ? WHERE person_id = ?", (target_pid, old_pid))

        # B. Delete the old identity record
        cursor.execute("DELETE FROM people WHERE person_id = ?", (old_pid,))

    else:
        # Simple rename (no merge)
        cursor.execute("UPDATE people SET name = ? WHERE person_id = ?", (new_name, old_pid))
        st.toast(f"Renamed ID {old_pid} to {new_name}")


# --- Main App Logic ---

st.title("🤖 My AI Photo Album")

# Create Tabs to separate Search from Labeling
tab_search, tab_chat, tab_people = st.tabs(["🔎 Search", "💬 Chat", "👥 Labeling"])

conn: sqlite3.Connection = get_db_connection()
cursor: sqlite3.Cursor = conn.cursor()

# TODO: use dotenv
open_api_key = "REMOVED"
google_api_key = "REMOVED"
# ==========================================
# TAB 1: SEARCH ENGINE (New Feature)
# ==========================================
with tab_search:
    st.header("Search your Memories")

    # 1. Load Engine (Cached)
    # This might take 2-3 seconds the first time you run the app
    engine = get_search_engine()

    # 2. Input
    query = st.text_input("Ask for anything...", placeholder="e.g. 'Ethan at the beach' or 'Birthday cake'",
                          key="search_box")

    # 3. Results
    if query:
        # Run search
        results = engine.search(query, limit=20)

        st.markdown(f"**Found {len(results)} results for:** `{query}`")

        if results:
            # Display Grid
            search_cols = st.columns(4)
            for i, path in enumerate(results):
                if os.path.exists(path):
                    # Use width="stretch" to fix warnings
                    search_cols[i % 4].image(path, width="stretch")
                    # Optional: Add file name caption
                    # search_cols[i % 4].caption(os.path.basename(path))
        else:
            st.warning("No photos found. Try a simpler query!")

# ==========================================
# TAB 2: CHAT (The New Feature)
# ==========================================
with tab_chat:
    st.header("Chat with your Photos")

    # 1. Initialize History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 2. Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # If the bot replied with photos, show them
            if "images" in message and message["images"]:
                cols = st.columns(len(message["images"]))
                for i, img_path in enumerate(message["images"]):
                    cols[i].image(img_path, width="stretch")

    # 3. Handle User Input
    if prompt := st.chat_input("Ask about your photos..."):
        if not open_api_key or not google_api_key:
            st.error("Please enter an API Key in the sidebar first!")
            st.stop()

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Run the Bot
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Initialize engines
                search_engine = get_search_engine()
                bot = PhotoAgent(open_api_key, google_api_key, search_engine)

                # Get response
                result = bot.run(prompt)

                # Show text
                st.markdown(result["answer"])

                # Show sources (the photos it looked at)
                if result.get("sources"):
                    st.caption("I looked at these photos:")
                    cols = st.columns(len(result["sources"]))
                    for i, img_path in enumerate(result["sources"]):
                        cols[i].image(img_path, width="stretch")

                if result.get("generated_image"):
                    st.image(result["generated_image"], caption="Generated by Google Imagen")

        # Save bot response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "images": result.get("sources"),
            "generated_image": result.get("generated_image")
        })

# ==========================================
# TAB 3: LABELING TOOL (Existing Feature)
# ==========================================
with tab_people:
    # 1. Fetch Clusters
    cursor.execute("""
                   SELECT p.person_id,
                          p.name,
                          ph.display_path,
                          pf.bounding_box
                   FROM people p
                            JOIN photo_faces pf ON p.representative_face_id = pf.id
                            JOIN photos ph ON pf.photo_id = ph.photo_id
                   ORDER BY p.person_id ASC
                   """)
    people_rows: List[Tuple[int, str, str, str]] = cursor.fetchall()

    if not people_rows:
        st.warning("No clusters found. Please run ingestion first.")
    else:
        # --- Sidebar Logic (Only relevant for Labeling) ---
        st.sidebar.header(f"Found {len(people_rows)} People")
        st.sidebar.caption("Rename people here 👇")

        if 'selected_pid' not in st.session_state:
            st.session_state.selected_pid = people_rows[0][0]

        # Render Sidebar List
        for pid, name, path, bbox_str in people_rows:
            with st.sidebar.container():
                cols = st.columns([1, 2])

                # Thumbnail
                crop = load_face_crop(path, bbox_str)
                if crop is not None:
                    cols[0].image(crop, width="stretch")
                else:
                    cols[0].error("?")

                # Rename Input
                new_name = cols[1].text_input(
                    label="Name",
                    value=name,
                    key=f"input_{pid}",
                    label_visibility="collapsed"
                )

                # Rename Handler
                if new_name != name:
                    merge_identities(cursor, pid, new_name)
                    conn.commit()
                    st.rerun()

                # Selection Button
                if cols[1].button("Show Photos", key=f"btn_{pid}"):
                    st.session_state.selected_pid = pid

        # --- Main Gallery Area (Filtered by Sidebar) ---

        selected_pid = st.session_state.selected_pid

        # Get fresh name
        cursor.execute("SELECT name FROM people WHERE person_id = ?", (selected_pid,))
        res = cursor.fetchone()
        current_name = res[0] if res else "Unknown"

        st.subheader(f"📸 {current_name}")

        # Fetch photos for selected person
        cursor.execute("""
                       SELECT DISTINCT ph.display_path
                       FROM photo_faces pf
                                JOIN photos ph ON pf.photo_id = ph.photo_id
                       WHERE pf.person_id = ? LIMIT 50
                       """, (selected_pid,))

        person_photos = [r[0] for r in cursor.fetchall()]

        if person_photos:
            p_cols = st.columns(5)
            for idx, path in enumerate(person_photos):
                if os.path.exists(path):
                    p_cols[idx % 5].image(path, width="stretch")
        else:
            st.info("No photos left in this cluster.")

conn.close()