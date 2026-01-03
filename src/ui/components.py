import streamlit as st
import os
from PIL import Image
from src.util.image_util import get_exif_timestamp, get_timestamp_from_heic
from typing import Dict, List, Any
from src.common.types import ImageAnalysisResult, FaceData
from src.util.image_util import load_face_crop, calculate_face_dim

def render_photo_card(result: ImageAnalysisResult, metric: Dict[Any, Any] = None, context_label: str = ""):
    """
    Renders a single photo card with a collapsible Inspector.

    Args:
    """
    skip_metric = False
    if not result:
        st.error("Results not found")
        return

    if not metric:
        # If this is not a search then we will skip printing
        skip_metric = True


    display_path = result.display_path
    # 1. Render Image
    st.image(display_path, width="stretch")

    # 2. The Inspector
    with st.expander(f"🔍 Technical Analysis {context_label}"):

        # --- SECTION A: RANKING METRICS ---
        if metric:
            st.subheader("Ranking Breakdown")
            # Separate keys into groups to make them readable
            semantic_keys = {"semantic", "mmr_rank"}
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

        st.divider()

        # --- SECTION B: FACES ---
        faces: List[FaceData] = result.faces or []
        if faces:
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