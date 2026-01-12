import streamlit as st
import os
from typing import Dict, List, Any
from src.common.types import ImageAnalysisResult, FaceData
from src.util.image_util import load_face_crop
from src.util.logger import logger

def toggle_selection(photo_id: str, session_id: str):
    """
    Stores a unique combination of (Photo + Session).
    Allows the same photo to be 'relevant' for multiple different searches.
    """
    if "selected_pairs" not in st.session_state:
        st.session_state.selected_pairs = set() # Set of tuples

    # The unique key is now the PAIR
    interaction_key = (photo_id, session_id)

    if interaction_key in st.session_state.selected_pairs:
        st.session_state.selected_pairs.remove(interaction_key)
        # st.toast(f"↩️ Undo selection", icon="🗑️")
    else:
        st.session_state.selected_pairs.add(interaction_key)
        # st.toast(f"✅ Saved match for this search", icon="⭐")

@st.fragment
def render_photo_card(result: ImageAnalysisResult, metric: Dict[Any, Any] = None, context_label: str = "", show_raw: bool = False, session_id: str = "default"):
    """
    Renders a single photo card with a collapsible Inspector.

    Args:
    """
    if not result:
        st.error("Results not found")
        return

    display_path = result.display_path
    # 1. Render Image
    st.image(display_path, width="stretch")

    is_selected = (result.photo_id, session_id) in st.session_state.get("selected_pairs", set())

    btn_type = "primary" if not is_selected else "secondary"
    btn_label = "🌟 Best Match" if not is_selected else "✅ Selected (Undo)"

    st.button(
        btn_label,
        key=f"btn_{result.photo_id}_{session_id}", # Unique key per session
        type=btn_type,
        on_click=toggle_selection,
        args=(result.photo_id, session_id),
        use_container_width=True
    )

    # 2. The Inspector
    with st.expander(f"🔍 Technical Analysis {context_label}"):

        # --- SECTION A: RANKING METRICS ---
        if metric:
            st.subheader("Ranking Breakdown")
            # Separate keys into groups to make them readable
            semantic_keys = {"semantic", "mmr_rank", "final_relevance"}
            # Everything else starting with 'g_' is global technical
            global_tech = {k: v for k, v in metric.items() if k.startswith("g_")}
            # Everything else starting with 'f_' is face technical
            face_tech = {k: v for k, v in metric.items() if k.startswith("f_")}

            col1, col2 = st.columns(2)
            with col1:
                st.caption("Relevance & Diversity")
                for k in semantic_keys:
                    if k in metric: st.write(f"**{k.replace('_', ' ').title()}:** {metric[k]}")

            with col2:
                st.caption("Global Quality Scores")
                for k, v in global_tech.items():
                    st.write(f"**{k[2:].title()}:** {v}") # Strips 'g_'
            if face_tech:
                st.caption("Normalized Face Scores (For Search Target)")
                scols = st.columns(len(face_tech))
                for idx, (sk, sv) in enumerate(face_tech.items()):
                    label = sk[2:].title()
                    with scols[idx]:
                        st.write(f"**{label}:** {sv:.2f}")


        st.divider()

        # --- SECTION B: FACES ---
        faces: List[FaceData] = result.faces or []
        if show_raw and faces:
            st.subheader(f"👥 Detected {len(faces)} Faces")
            for i, face in enumerate(faces):
                fcol1, fcol2 = st.columns([1, 2])

                with fcol1:
                    cropped_face = load_face_crop(result.display_path, face.bbox)
                    if cropped_face:
                        st.image(cropped_face, width="stretch")

                with fcol2:
                    st.markdown(f"**{face.name or f'Unknown {i+1}'}**")
                    # Use the .metrics property we added to FaceData for dynamic rendering!
                    face_metrics = face.metrics

                    # Group them into a readable string
                    summary = []
                    for k, v in face_metrics.items():
                        if v is not None:
                            val = f"{v:.2f}" if isinstance(v, float) else str(v)
                            summary.append(f"• {k.replace('_', ' ').title()}: `{val}`")

                    st.markdown("<br>".join(summary), unsafe_allow_html=True)
                st.write("") # Spacer
        else:
            st.caption("No faces detected.")

        st.divider()

        # --- SECTION C: FILE INFO ---
        st.caption("📂 File System")
        st.text(f"Timestamp: {result.timestamp}")
        st.code(f"Original: {os.path.basename(result.original_path)}", language="bash")