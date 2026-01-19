import streamlit as st
import uuid
from src.ui.state import get_search_engine, get_gpt_client, get_db
from src.ui.components import render_photo_card

def render_search_page(mode="search", show_raw=False):
    st.header("Search your Memories" if mode == "search" else "Generate Memories")

    engine = get_search_engine()
    gpt = get_gpt_client()
    db = get_db()

    with st.form("gen_form"):
        query = st.text_input("Ask for anything...", placeholder="e.g. 'Generate a photo of Jacob playing'")
        submitted = st.form_submit_button("Generate", type="primary")

    if submitted and query:
        # Refresh Session ID
        st.session_state.search_session_id = str(uuid.uuid4())
        current_sess_id = st.session_state.search_session_id

        # 1. GPT Parsing
        search_filter, message = gpt._get_agent_response(query)
        if search_filter:
            # Log the search query with the active ranking model
            db.log_search_query(current_sess_id, query, search_filter, ranking_model=engine.current_strategy)
            st.success(f"✅ Agent Parsed: {search_filter}")
            with st.expander("🔍 See Raw Arguments", expanded=True):
                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("**Parsed Intent**")
                    if search_filter.fn_name == "search_memory":
                        st.info("📂 Search Database")
                    elif search_filter.fn_name == "generate_image":
                        st.warning("🎨 Generate Image (!!! not supported!!!")

                with c2:
                    st.markdown("**Extracted Filters**")
                    st.json(search_filter.to_dict())

        # 2. Execution
        ranking_result = engine.searchv2(search_filter, limit=30)
        st.markdown(f"**Found {len(ranking_result.ranked_results)} results for:** `{query}`")

        # 3. Store History (for LTR)
        # Store in session state for click tracking
        if "search_ranking_results" not in st.session_state:
            st.session_state.search_ranking_results = {}
        st.session_state.search_ranking_results[current_sess_id] = ranking_result

        # 3.1 Log impressions for all shown results
        impressions = [(r.photo_id, i) for i, r in enumerate(ranking_result.ranked_results)]
        db.log_impressions(current_sess_id, impressions)

        # 4. Render Grid
        if ranking_result.ranked_results:
            cols = st.columns(3)
            for i, result in enumerate(ranking_result.ranked_results):
                display_metrics = ranking_result.display_metrics.get(result.display_path, {})
                with cols[i % 3]:
                    render_photo_card(
                        result,
                        display_metrics,
                        context_label=f"#{i+1}",
                        show_raw=show_raw,
                        session_id=current_sess_id,
                        position=i
                    )
        else:
            st.warning("No photos found.")