import streamlit as st
from src.ui.state import get_search_engine, save_feedback_batch, get_db

def render_sidebar():
    """Renders the sidebar and returns the selected App Mode."""

    # 1. Navigation
    app_mode = st.sidebar.radio(
        "Navigate",
        ["🔎 Search", "👥 Labeling", "📊 Analytics"]
    )

    # 2. Search/Generate Controls
    if app_mode in ["🔎 Search"]:
        st.sidebar.divider()
        engine = get_search_engine()

        st.sidebar.markdown("### ⚙️ Ranking Engine")
        strategy_choice = st.sidebar.radio(
            "Ranking Model:",
            options=["Smart AI (XGBoost)", "Classic (Heuristic)"],
            index=0,
            help="Switch between ranking strategies."
        )

        if "XGBoost" in strategy_choice:
            engine.set_ranking_strategy("xgboost")
            st.sidebar.caption("Using ML Model")
        else:
            engine.set_ranking_strategy("heuristic")
            st.sidebar.caption("Using Manual Rules")

        st.sidebar.markdown("### 🧠 LTR Training")
        current_selection = st.session_state.get("selected_pairs", set())
        st.sidebar.caption(f"Selected: {len(current_selection)} photos")

        if st.sidebar.button("💾 Save Feedback", type="primary"):
            db = get_db()
            save_feedback_batch(db)

    # 3. Global Toggles
    show_raw_db = st.sidebar.toggle("Show Raw Database Metrics", value=False)

    return app_mode, show_raw_db