import sqlite3
import cv2
import json
import os
import os
# --- FORCE CPU MODE ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      # Hide GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # Fix crash on some Intel/AMD CPUs

import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from deepface import DeepFace
from tqdm import tqdm
from PIL import Image, ExifTags
from datetime import datetime


class FaceTagger:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

        # --- NEW: Modern MediaPipe Tasks API Setup ---
        # This requires 'face_landmarker.task' in your project root
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1)
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        # ---------------------------------------------

        self._migrate_db()

    def _migrate_db(self):
        """Adds new metadata columns to the table if they don't exist."""
        cursor = self.conn.cursor()

        # 1. Update PHOTOS table (Timestamp)
        cursor.execute("PRAGMA table_info(photos)")
        photo_cols = {row[1] for row in cursor.fetchall()}
        if "timestamp" not in photo_cols:
            print("   + Adding column: timestamp to PHOTOS table")
            cursor.execute("ALTER TABLE photos ADD COLUMN timestamp DATETIME")

        # 2. Update PHOTO_FACES table (Attributes)
        face_cols_to_add = [
            ("pose", "TEXT"),
            ("shot_type", "TEXT"),
            ("emotion", "TEXT"),
            ("age_est", "INTEGER"),
            ("lighting", "TEXT")
        ]
        cursor.execute("PRAGMA table_info(photo_faces)")
        existing_face_cols = {row[1] for row in cursor.fetchall()}

        for col_name, col_type in face_cols_to_add:
            if col_name not in existing_face_cols:
                print(f"   + Adding column: {col_name} to PHOTO_FACES table")
                cursor.execute(f"ALTER TABLE photo_faces ADD COLUMN {col_name} {col_type}")

        self.conn.commit()

    def _get_exif_timestamp(self, pil_image):
        try:
            exif = pil_image._getexif()
            if not exif: return None
            date_str = exif.get(36867)  # DateTimeOriginal
            if date_str:
                try:
                    dt = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    return None
            return None
        except Exception:
            return None

    def _get_shot_type(self, img_w, img_h, bbox):
        x1, y1, x2, y2 = bbox
        face_area = (x2 - x1) * (y2 - y1)
        total_area = img_w * img_h
        ratio = face_area / (total_area + 1e-6)
        if ratio > 0.25: return "Close-up"
        if ratio > 0.08: return "Medium-Shot"
        return "Full-Body"

    def _get_head_pose(self, image_rgb):
        """
        Uses Modern MediaPipe Tasks API to estimate yaw.
        """
        try:
            # Convert numpy array to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # Detect
            detection_result = self.landmarker.detect(mp_image)

            if not detection_result.face_landmarks:
                return "Unknown"

            # Get Landmarks
            landmarks = detection_result.face_landmarks[0]

            # Extract key points (normalized 0-1)
            nose_tip = landmarks[1]
            left_ear = landmarks[234]
            right_ear = landmarks[454]

            # Calculate Yaw
            dist_left = abs(nose_tip.x - left_ear.x)
            dist_right = abs(nose_tip.x - right_ear.x)
            ratio = dist_left / (dist_right + 1e-6)

            if ratio > 3.5: return "Side-Left"
            if ratio < 0.28: return "Side-Right"
            if ratio > 1.8: return "Angled-Left"
            if ratio < 0.55: return "Angled-Right"
            return "Front"

        except Exception as e:
            # print(f"MediaPipe Error: {e}")
            return "Unknown"

    def _analyze_deepface(self, crop_bgr):
        try:
            objs = DeepFace.analyze(
                img_path=crop_bgr,
                actions=['age', 'emotion'],
                enforce_detection=False,
                detector_backend='skip',
                silent=True
            )
            return objs[0]['age'], objs[0]['dominant_emotion']
        except Exception:
            return None, None

    def run(self):
        print("🚀 Starting Smart Tagging Process (Modern API)...")
        cursor = self.conn.cursor()

        # PHASE 1: Timestamps
        print("\n📅 Phase 1: Timestamps...")
        cursor.execute("SELECT photo_id, display_path FROM photos WHERE timestamp IS NULL")
        photos = cursor.fetchall()
        for photo_id, path in tqdm(photos, desc="Photos"):
            if os.path.exists(path):
                try:
                    img = Image.open(path)
                    ts = self._get_exif_timestamp(img)
                    if ts:
                        cursor.execute("UPDATE photos SET timestamp = ? WHERE photo_id = ?", (ts, photo_id))
                except:
                    pass
        self.conn.commit()

        # PHASE 2: Attributes
        print("\n👤 Phase 2: Face Attributes...")
        cursor.execute("""
                       SELECT pf.id, pf.photo_id, pf.bounding_box, ph.display_path
                       FROM photo_faces pf
                                JOIN photos ph ON pf.photo_id = ph.photo_id
                       WHERE pf.pose IS NULL
                          OR pf.shot_type IS NULL
                       """)
        rows = cursor.fetchall()

        for row in tqdm(rows, desc="Faces"):
            face_id, photo_id, bbox_json, path = row
            if not os.path.exists(path): continue

            try:
                img_bgr = cv2.imread(path)
                if img_bgr is None: continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h, w, _ = img_bgr.shape

                bbox = json.loads(bbox_json)
                x1, y1, x2, y2 = [int(b) for b in bbox]

                shot_type = self._get_shot_type(w, h, bbox)

                pad_x, pad_y = int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)
                cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

                face_crop_rgb = img_rgb[cy1:cy2, cx1:cx2]
                face_crop_bgr = img_bgr[cy1:cy2, cx1:cx2]

                if face_crop_rgb.size == 0: continue

                # New Pose Call
                pose = self._get_head_pose(face_crop_rgb)

                age, emotion = None, None
                if shot_type in ["Close-up", "Medium-Shot"]:
                    age, emotion = self._analyze_deepface(face_crop_bgr)

                cursor.execute("""
                               UPDATE photo_faces
                               SET pose=?,
                                   shot_type=?,
                                   age_est=?,
                                   emotion=?
                               WHERE id = ?
                               """, (pose, shot_type, age, emotion, face_id))
            except Exception:
                pass

        self.conn.commit()
        print("✅ Tagging Complete!")
        self.conn.close()


if __name__ == "__main__":
    tagger = FaceTagger(".db/sqlite/photos.db")
    tagger.run()