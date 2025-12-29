from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, List, Any
from contextlib import contextmanager

import numpy as np
import cv2
import yaml
import sys
import os

@contextmanager
def pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

class PoseExtractor:
    def __init__(self, repo_root: Optional[Path] = None, gpu: bool = True):
        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[2] / "third_party" / "3DDFA_V2"

        self.repo_root = repo_root
        sys.path.insert(0, str(repo_root))

        from TDDFA import TDDFA  # type: ignore
        from FaceBoxes import FaceBoxes  # type: ignore
        from utils.pose import calc_pose  # type: ignore

        self._calc_pose = calc_pose

        cfg = yaml.safe_load(open(self.repo_root / "configs" / "mb1_120x120.yml", "r"))

        with pushd(self.repo_root):
            self.tddfa = TDDFA(gpu_mode=gpu, **cfg)

        self.detector = FaceBoxes()

        # keep handle to pose util
        from utils.pose import calc_pose  # type: ignore
        self._calc_pose = calc_pose

    @staticmethod
    def face_to_boxes(faces: List[Any]) -> np.ndarray:
        """
        Convert insightface faces -> 3DDFA boxes (N,5): [x1,y1,x2,y2,score]
        :param faces:
        :return:
        """
        boxes: List[List[float]] = []
        for f in faces:
            score = float(getattr(f, "det_score", 1.0))
            x1, y1, x2, y2 = map(float, f.bbox)
            boxes.append([x1, y1, x2, y2, score])
        return np.asarray(boxes, dtype=np.float32)

    def extract_pose_from_faces(self, img_bgr: np.ndarray, faces: List[Any]) -> List[List[float]]:
        h, w, _ = img_bgr.shape
        boxes = self.face_to_boxes(faces)

        param_lst, roi_box_lst = self.tddfa(img_bgr, boxes)

        results: List[List[float]] = []
        for i, param in enumerate(param_lst):
            _, pose = self._calc_pose(param)
            yaw, pitch, roll = pose
            results.append([float(yaw), float(pitch), float(roll)])

        return results


