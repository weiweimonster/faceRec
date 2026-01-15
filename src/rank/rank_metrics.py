from dataclasses import dataclass
from typing import Optional, List

@dataclass
class FaceRankMetrics:
    # Face specific
    f_orient_score: float
    f_blur: float
    f_conf: float
    f_height: float
    f_width: float
    def __str__(self):
        return (
            f"👤 [Face] Conf: {self.f_conf:.2f} | "
            f"Blur: {self.f_blur:.1f} | "
            f"Orient: {self.f_orient_score:.2f} | "
            f"Size: {int(self.f_width)}x{int(self.f_height)}"
        )


@dataclass
class PictureRankMetrics:
    # Global data
    semantic_score: float
    aesthetic_score: float
    year: int
    month: int
    date: int
    g_blur: float
    g_brightness: float
    g_contrast: float
    g_iso: int

    has_face: bool = False
    face_metrics: Optional[List[FaceRankMetrics]] = None

    def __str__(self):
        # 1. Header with Date and Scores
        header = (
            f"🖼️ [Photo] {self.year}-{self.month:02d}-{self.date:02d} | "
            f"Semantic: {self.semantic_score:.3f} | "
            f"Aesthetic: {self.aesthetic_score:.2f}"
        )

        # 2. Technical Stats
        stats = (
            f"   Stats: Blur={self.g_blur:.1f}, Bright={self.g_brightness:.1f}, "
            f"Contrast={self.g_contrast:.1f}, ISO={self.g_iso}"
        )

        # 3. Face Details (Iterate if they exist)
        faces_str = ""
        if self.has_face and self.face_metrics:
            faces_str = "\n   Faces Found:\n"
            for i, face in enumerate(self.face_metrics):
                faces_str += f"     {i+1}. {str(face)}\n"
        elif self.has_face and not self.face_metrics:
            faces_str = "\n   ⚠️ Face flag True but no metrics list.\n"
        else:
            faces_str = "\n   (No faces target/detected)\n"

        return f"{header}\n{stats}{faces_str}"

