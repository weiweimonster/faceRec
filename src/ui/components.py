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
    with st.expander(f"🔍 Inspect {context_label}"):
        # Fetching metadata from photo_metadata dict

        db_ts = result.timestamp
        # Section 1: Timestamps
        st.caption(f"**DB Date:** {db_ts}")
        if not skip_metric:
            semantic_similarity = metric.get("semantic_sim")
            if semantic_similarity: st.caption(f"**Semantic Similarity:** {semantic_similarity:.5f}")

            final_relevance = metric.get("final_relevance")
            if final_relevance: st.caption(f"**Final Relevance Score:** {final_relevance:.5f}")
            st.markdown(f"""
                <div style="font-size: 0.8em; margin-bottom: 5px; border-left: 2px solid #555; padding-left: 5px;">
                    • Norm Quality Final (with confidence): {metric["quality_boost"]:.3f}<br>
                    • Norm Quality Raw (w/o confidence): {metric["total_quality_raw"]:.3f}<br>
                    • Norm Brightness: {metric["norm_brightness"]:.3f}<br>
                    • Norm Blur: {metric["norm_blur"]:.2f}<br>
                    • Norm Size (H/W): {metric["norm_size"]:.3f}<br>
                    • Norm Orientation: {metric["norm_orientation"]:.3f}<br>
                    • MMR Rank: {metric["mmr_rank"]}<br>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Print out Metrics


        # Section 2: Faces & Poses (Requested Feature)
        faces: List[FaceData] = result.faces
        if len(faces) > 0:
            for face in faces:
                # st.caption(f"Debug {data}")
                st.caption(f"👥 **Detected {len(faces)} Faces**")
                # Load the cropped face
                cropped_face = load_face_crop(display_path, face.bbox)
                face_width, face_height = calculate_face_dim(face.bbox)
                if cropped_face:
                    st.image(cropped_face, width="stretch")
                st.markdown(f"""
                <div style="font-size: 0.8em; margin-bottom: 5px; border-left: 2px solid #555; padding-left: 5px;">
                    <b>{face.name}</b><br>
                    • Pose: <code>{face.pose}</code> ({face.shot_type})<br>
                    • Angles: Y:{int(face.yaw)}° / P:{int(face.pitch)}°<br>
                    • Confidence: {face.confidence:.2f}<br>
                    • Blur: {face.blur_score:.2f}<br>
                    • Brightness: {face.brightness:.2f}<br>
                    • Face Width: {face_width:.2f}<br>
                    • Face Height: {face_height:.2f}<br>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No faces detected.")

        st.markdown("---")

        # Section 3: Paths
        st.caption("📂 Paths")
        orig_path = result.original_path
        st.code(f"Orig: {os.path.basename(orig_path)}", language="bash")