import os
import shutil
import pytest
from PIL import Image
from src.ingestion.format_handler import ensure_display_version, load_image_for_processing
from unittest.mock import patch

@pytest.fixture
def temp_dirs(tmp_path):
    # Create a temp source and cache directory
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return source_dir, cache_dir

def test_load_image_handles_standard_jpg(temp_dirs):
    source, _ = temp_dirs
    img_path = source / "test.jpg"

    # Create a real small jpeg
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)

    loaded = load_image_for_processing(str(img_path))
    assert loaded is not None
    assert loaded.size == (100, 100)
    assert loaded.mode == "RGB"

def test_ensure_display_version_skips_jpg(temp_dirs):
    """
    If file is already jpg, it should just return the original path.
    """
    source, cache = temp_dirs
    img_path = source / "photo.jpg"
    img_path.touch()

    result = ensure_display_version(str(img_path), str(cache))
    assert result == str(img_path)
    # Ensure nothing was put in the cache
    assert len(list(cache.iterdir())) == 0

def test_ensure_display_version_converts_heic(temp_dirs):
    """
    We simulate a HEIC conversion
    Note: Creating a real HEIC programmatically is hard, so we mock Image.open
    and the save method to verify the logic flow.
    """
    source, cache = temp_dirs
    heic_path = source / "photo.heic"
    heic_path.touch()

    # Mock PIL so we don't need a real HEIC file
    # with pytest.raises(Exception):
    #     pass

    with patch("src.ingestion.format_handler.Image.open") as mock_open:
        mock_img = patch('unittest.mock.Mock').start()
        mock_img_instance = mock_open.return_value

        # Run function
        result_path = ensure_display_version(str(heic_path), str(cache))

        assert result_path.endswith(".jpg")
        assert "converted_cache" in str(result_path) or "cache" in str(result_path)
        mock_img_instance.convert.assert_called_with("RGB")
        mock_img_instance.convert().save.assert_called()

