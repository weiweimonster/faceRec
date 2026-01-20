"""
PoseExtractionStage: Extract 3D face pose using 3DDFA_V2.
"""
from __future__ import annotations

from typing import List, Optional

from src.ingestion.stages.base import ProcessingStage, PipelineItem
from src.pose.pose_extractor import PoseExtractor
from src.pose.pose import Pose
from src.util.logger import logger


class PoseExtractionStage(ProcessingStage):
    """
    Extract 3D pose (yaw, pitch, roll) for detected faces using 3DDFA_V2.

    Dependencies:
    - face_detection: Requires raw_faces from InsightFace

    Outputs:
    - Updates item.result.faces[i].yaw, pitch, roll, pose
    """

    def __init__(self, use_gpu: bool = True):
        """
        Args:
            use_gpu: Whether to use GPU for pose extraction
        """
        self.use_gpu = use_gpu
        self.pose_extractor: Optional[PoseExtractor] = None

    @property
    def name(self) -> str:
        return "pose_extraction"

    @property
    def dependencies(self) -> List[str]:
        return ["face_detection"]

    @property
    def supports_gpu_batching(self) -> bool:
        return False

    def initialize(self) -> None:
        """Load 3DDFA model."""
        if self.pose_extractor is not None:
            return

        logger.info("Loading 3DDFA_V2 pose extractor...")
        self.pose_extractor = PoseExtractor(gpu=self.use_gpu)
        logger.info("3DDFA_V2 loaded")

    def cleanup(self) -> None:
        """Release model resources."""
        self.pose_extractor = None

    def process(self, items: List[PipelineItem]) -> List[PipelineItem]:
        """Extract pose for all faces in processable items."""
        if self.pose_extractor is None:
            self.initialize()

        for item in items:
            if not item.is_processable():
                continue

            if item.cv_image is None:
                continue  # Skip silently, face detection already failed

            if not item.raw_faces:
                continue  # No faces to process

            try:
                self._extract_poses(item)
            except Exception as e:
                # Pose extraction failure is not fatal - continue without pose data
                logger.warning(f"Pose extraction failed for {item.result.original_path}: {e}")

        return items

    def _extract_poses(self, item: PipelineItem) -> None:
        """Extract pose for all faces in a single image."""
        poses = self.pose_extractor.extract_pose_from_faces(item.cv_image, item.raw_faces)

        # Validate index alignment
        if len(poses) != len(item.result.faces):
            logger.warning(
                f"Pose count mismatch: {len(poses)} poses for {len(item.result.faces)} faces"
            )
            return

        # Update face data with pose information
        for pose_angles, face_data in zip(poses, item.result.faces):
            yaw, pitch, roll = pose_angles
            face_data.yaw = yaw
            face_data.pitch = pitch
            face_data.roll = roll
            face_data.pose = Pose.from_angles(yaw, pitch)
