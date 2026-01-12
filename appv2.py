import streamlit as st
from src.ui.state import init_session_state
from src.ui.sidebar import render_sidebar
from src.ui.search import render_search_page
from src.ui.label import render_labeling_page

st.set_page_config(layout="wide", page_title="AI Photo Manager")
st.title("🤖 My AI Photo Album")
init_session_state()
app_mode, show_raw_db = render_sidebar()

# ==========================================
# TAB 1: SEARCH ENGINE
# ==========================================
if app_mode == "🔎 Search":
    render_search_page(mode="search", show_raw=show_raw_db)

# ==========================================
# TAB 2: LABELING & INSPECTION
# ==========================================
elif app_mode == "👥 Labeling":
    render_labeling_page(show_raw=show_raw_db)
