from __future__ import annotations
import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener

register_heif_opener()
def load_image_for_processing(file_path: str) -> Image.Image | None:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        img = Image.open(file_path)

        # Force conversion to RGB
        # Fix issues with RGBA PNGs or Grayscale images crashing some model
        img = img.convert("RGB")
        return img
    except (OSError, UnidentifiedImageError, ValueError) as e:
        print(f"⚠️ Warning: Could not load image {file_path}. Error: {e}")
        return None

def ensure_display_version(file_path: str, cache_dir: str, quality: int = 80) -> str:
    """
    Checks if a file is browser-compatible (JPG/PNG).
    If it is HEIC (or other non-web formats), converts it to JPEG and saves it
    to the cache directory.

    Args:
        file_path (str): Source file path.
        cache_dir (str): Directory where converted copies are stored.
        quality (int): JPEG quality (1-100) for the cached version.

    Returns:
        str: The absolute path to the displayable image (either original or cached).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg", ".png"}:
        return str(path.resolve())

    if ext in ['.heic', '.heif']:
        os.makedirs(cache_dir, exist_ok=True)

        # Define the new file name .heic -> .jpg
        # Note: We might want to hash the filepath to avoid collisions
        new_filename = path.stem + ".jpg"
        cached_path = os.path.join(cache_dir, new_filename)

        if os.path.exists(cached_path):
            return cached_path

        try:
            # Perform conversion
            img = Image.open(file_path)
            img.convert("RGB").save(cached_path, "JPEG", quality=quality)
            return cached_path
        except Exception as e:
            print(f"❌ Failed to convert HEIC {file_path}: {e}")
            # Fallback: return original and hope for the best (or handle error upstream)
            return file_path
    print("file extension not supported")
    return file_path