from typing import List, Tuple, Any
import cv2
import numpy as np
from datetime import datetime
import json
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
import os

def is_face_too_small(
    bbox: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    min_side_px: int = 32,
    min_area_ratio: float = 0.001,
):
    x1, y1, x2, y2 = bbox
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)

    if w < min_side_px or h < min_side_px:
        return True

    face_area = w * h
    image_area = image_width * image_height

    if image_area <= 0:
        return True

    if (face_area / image_area) < min_area_ratio:
        return True

    return False

def calculate_face_quality(img_bgr: np.ndarray, bbox: List[int]) -> Tuple[float, float]:
    try:
        x1, y1, x2, y2 = bbox
        h, w, _ = img_bgr.shape

        pad_x, pad_y = int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)
        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        face_crop_bgr = img_bgr[cy1:cy2, cx1:cx2]

        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)

        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        bright = float(np.mean(gray))

        return blur, bright
    except:
        return 0.0, 0.0

def calculate_shot_type(img_bgr: np.ndarray, bbox: List[int]) -> str:
    x1, y1, x2, y2 = bbox
    face_area = (x2 - x1) * (y2 - y1)
    h, w, _ = img_bgr.shape
    total_area = w * h
    ratio = face_area / (total_area + 1e-6)
    shot_type = "Full-Body"
    if ratio > 0.25:
        shot_type = "Close-up"
    elif ratio > 0.08:
        shot_type = "Medium-Shot"
    return shot_type


def get_exif_timestamp(image_path: str) -> str | None:
    """
    Extracts the creation timestamp from EXIF.
    Supports standard 'YYYY:MM:DD' tags and Apple's hidden 'UserComment' JSON.
    Returns format: "YYYY-MM-DD HH:MM:SS"
    """
    try:
        img = Image.open(image_path)
        if not img: return None
        exif = img._getexif()
        if not exif: return None

        # --- STRATEGY 1: Standard EXIF Tags ---
        # 36867: DateTimeOriginal (Best)
        # 36868: DateTimeDigitized (Good for scans)
        # 306:   DateTime (Last Modified - Least reliable)
        standard_tags = [36867, 36868, 306]

        for tag_id in standard_tags:
            date_str = exif.get(tag_id)
            if date_str:
                try:
                    # Standard EXIF format is "YYYY:MM:DD HH:MM:SS"
                    # We parse it to validate, then re-format to "YYYY-MM-DD..."
                    dt = datetime.strptime(str(date_str), "%Y:%m:%d %H:%M:%S")
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue  # Try the next tag if this one is malformed

        # --- STRATEGY 2: Apple 'UserComment' (Hidden JSON) ---
        # Common in photos from 3rd party iPhone apps (Instagram, Snapchat, etc.)
        user_comment = exif.get(37510)

        if user_comment:
            try:
                # 1. Clean up binary string (remove 'ASCII\x00...')
                decoded = user_comment.decode('utf-8', errors='ignore')
                json_start = decoded.find('{')

                if json_start != -1:
                    data = json.loads(decoded[json_start:])

                    if 'date' in data:
                        apple_ts = float(data['date'])

                        # 2. Convert Apple Epoch (2001-01-01) to Unix Epoch
                        # Offset = 978307200 seconds
                        unix_ts = apple_ts + 978307200

                        dt = datetime.fromtimestamp(unix_ts)
                        return dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass  # Fail silently if JSON is bad

        return None

    except Exception:
        return None


def get_timestamp_from_heic(heic_path: str) -> str | None:
    from pillow_heif import register_heif_opener
    from PIL import Image

    register_heif_opener()
    img = Image.open(heic_path)
    exif = img.getexif()
    if 36867 in exif:
         return exif[36867]
    elif 306 in exif:
        return exif[306]

    return None

def plot_image_from_path(path: str, title: str = None, figsize: tuple = (15, 15)):
    import matplotlib.pyplot as plt

    try:
        register_heif_opener()
    except ImportError:
        print("⚠️ pillow-heif not installed. HEIC files will fail.")
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    try:
        # Open the image (HEIC support is handled by register_heif_opener)
        img = Image.open(path)

        # 1. Fix Rotation (Phone cameras often save images sideways with an EXIF tag)
        img = ImageOps.exif_transpose(img)

        # 2. Convert to RGB (Matplotlib expects RGB, removes Alpha channels if present)
        img = img.convert("RGB")

        # 3. Plot
        plt.figure(figsize=figsize)
        plt.imshow(img)

        if title:
            plt.title(title)

        plt.axis('off')  # Hide the x/y axis numbers
        plt.show()

    except Exception as e:
        print(f"❌ Error plotting image: {e}")

