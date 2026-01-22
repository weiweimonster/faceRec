import streamlit as st
import os
from typing import Dict, List, Any
from src.common.types import ImageAnalysisResult, FaceData
from src.features.registry import extract_face_display_values

def toggle_selection(photo_id: str, session_id: str, position: int = -1):
    """
    Stores a unique combination of (Photo + Session + Position).
    Allows the same photo to be 'relevant' for multiple different searches.
    """
    if "selected_pairs" not in st.session_state:
        st.session_state.selected_pairs = {}  # Dict of {(photo_id, session_id): position}

    # The unique key is the PAIR
    interaction_key = (photo_id, session_id)

    if interaction_key in st.session_state.selected_pairs:
        del st.session_state.selected_pairs[interaction_key]
    else:
        st.session_state.selected_pairs[interaction_key] = position

@st.fragment
def render_photo_card(
        result: ImageAnalysisResult,
        display_metrics: Dict[str, Any],
        context_label: str = "",
        show_raw: bool = False,
        session_id: str = "default",
        position: int = -1
):
    """
    Render photo card with interactive selection and metrics display.

    Args:
        result: ImageAnalysisResult to display
        display_metrics: Normalized/weighted metrics for UI
        context_label: Label like "#1" for rank display
        show_raw: Whether to show detailed face analysis
        session_id: Search session ID for click tracking
        position: Position in search results (0-indexed) for CTR tracking
    """
    if not result:
        st.error("Results not found")
        return

    display_path = result.display_path
    # 1. Render Image
    st.image(display_path, width="stretch")

    is_selected = (result.photo_id, session_id) in st.session_state.get("selected_pairs", {})

    btn_type = "primary" if not is_selected else "secondary"
    btn_label = "🌟 Best Match" if not is_selected else "✅ Selected (Undo)"

    st.button(
        btn_label,
        key=f"btn_{result.photo_id}_{session_id}", # Unique key per session
        type=btn_type,
        on_click=toggle_selection,
        args=(result.photo_id, session_id, position),
        use_container_width=True
    )

    # 2. The Inspector
    with st.expander(f"🔍 Technical Analysis {context_label}"):

        # --- SECTION A: RANKING METRICS ---
        if display_metrics:
            st.subheader("Ranking Breakdown")
            # Separate keys into groups to make them readable
            semantic_keys = {"semantic", "caption", "mmr_rank", "final_relevance", "xgboost_score"}
            global_metrics = {k: v for k, v in display_metrics.items() if k.startswith("g_")}
            face_metrics = {k: v for k, v in display_metrics.items() if k.startswith("f_")}

            col1, col2 = st.columns(2)
            with col1:
                st.caption("Relevance & Diversity")
                for k in semantic_keys:
                    if k in display_metrics: st.write(f"**{k.replace('_', ' ').title()}:** {display_metrics[k]}")

            with col2:
                st.caption("Global Quality Scores")
                for k, v in global_metrics.items():
                    st.write(f"**{k[2:].title()}:** {v}") # Strips 'g_'
            if face_metrics:
                st.caption("Normalized Face Scores (For Search Target)")
                scols = st.columns(len(face_metrics))
                for idx, (sk, sv) in enumerate(face_metrics.items()):
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
                    from src.util.image_util import load_face_crop
                    cropped_face = load_face_crop(result.display_path, face.bbox)
                    if cropped_face:
                        st.image(cropped_face, width="stretch")

                with fcol2:
                    st.markdown(f"**{face.name or f'Unknown {i+1}'}**")
                    # Use registry-based extraction for dynamic rendering
                    face_metrics = extract_face_display_values(face)

                    # Group them into a readable string
                    summary = []
                    for label, v in face_metrics.items():
                        if v is not None:
                            val = f"{v:.2f}" if isinstance(v, float) else str(v)
                            summary.append(f"• {label}: `{val}`")

                    st.markdown("<br>".join(summary), unsafe_allow_html=True)
                st.write("") # Spacer
        else:
            st.caption("No faces detected.")

        st.divider()

        # --- SECTION C: FILE INFO ---
        st.caption("📂 File System")
        st.text(f"Timestamp: {result.timestamp}")
        st.code(f"Original: {os.path.basename(result.original_path)}", language="bash")