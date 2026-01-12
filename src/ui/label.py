import streamlit as st
from src.ui.state import get_db, load_face_crop_from_str_streamlit
from src.ui.components import render_photo_card
from src.util.search_config import SearchFilters

def render_labeling_page(show_raw=False):
    db = get_db()
    people_rows = db.get_people_clusters()

    if not people_rows:
        st.warning("No clusters found. Please run ingestion first.")
        return

    # Sidebar List
    st.sidebar.header(f"Found {len(people_rows)} People")
    if 'selected_pid' not in st.session_state:
        st.session_state.selected_pid = people_rows[0]['id']

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

    # Main Gallery
    selected_pid = st.session_state.selected_pid
    current_person = next((p for p in people_rows if p['id'] == selected_pid), None)

    if current_person:
        st.subheader(f"📸 Photos of {current_person['name']}")

        # Reuse DB search logic manually
        filters = SearchFilters(is_person_search=True, people=[current_person['name']], fn_name="you shouldn't need this")
        candidate_paths = db.get_candidate_path(filters)

        if candidate_paths:
            photos = db.fetch_metadata_batch(candidate_paths, fields="all")
            cols = st.columns(3)
            for idx, photo in enumerate(photos):
                with cols[idx % 3]:
                    render_photo_card(photo)
        else:
            st.info("No photos found.")
    else:
        st.warning("Selected person ID not found. They may have been merged.")