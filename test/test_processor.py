from datetime import datetime

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.ingestion.processor import FeatureExtractor
from src.common.types import ImageAnalysisResult
import sys
from src.util.image_util import calculate_shot_type, calculate_face_quality, compute_global_visual_stats

# We mock these 'src' imports before importing the actual module
# so the test doesn't crash if these files aren't in the python path during testing.
sys.modules['src.pose.pose_extractor'] = MagicMock()
sys.modules['src.util.image_util'] = MagicMock()
sys.modules['src.model.florence'] = MagicMock()
sys.modules['src.model.text_embedder'] = MagicMock()

@pytest.fixture
def mock_dependencies():
    """
    Sets up the complex web of dependencies including:
    - FaceAnalysis (InsightFace)
    - CLIP
    - Pytorch
    - AestheticPredictor
    - Image Utils (ISO, Timestamp, etc)
    """
    with patch('src.ingestion.processor.FaceAnalysis') as mock_fa, \
            patch('src.ingestion.processor.clip.load') as mock_clip, \
            patch('src.ingestion.processor.PoseExtractor') as mock_pose_extractor, \
            patch('src.ingestion.processor.cv2') as mock_cv2, \
            patch('src.ingestion.processor.torch') as mock_torch, \
            patch('src.ingestion.processor.AestheticPredictor') as mock_aesthetic_cls, \
            patch('src.ingestion.processor.VisionScanner') as mock_vision_cls, \
            patch('src.ingestion.processor.TextEmbedder') as mock_embedder_cls, \
            patch('src.ingestion.processor.get_exif_timestamp') as mock_exif_ts, \
            patch('src.ingestion.processor.get_disk_timestamp') as mock_disk_ts, \
            patch('src.ingestion.processor.get_exif_iso') as mock_iso, \
            patch('src.ingestion.processor.is_face_too_small') as mock_too_small:

        # 1. Setup FaceAnalysis
        instance_fa = mock_fa.return_value
        instance_fa.prepare.return_value = None

        # 2. Setup CLIP
        mock_clip_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_clip.return_value = (mock_clip_model, mock_preprocess)

        # Mock the encode_image flow
        mock_tensor = MagicMock()
        mock_tensor.norm.return_value = 1.0
        mock_tensor.__itruediv__.return_value = mock_tensor
        mock_tensor.cpu().numpy.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_clip_model.encode_image.return_value = mock_tensor

        # 3. Setup Torch
        mock_torch.cuda.is_available.return_value = False
        mock_torch.load.return_value = {}

        # 4. Setup Aesthetic Predictor
        mock_aesthetic_instance = mock_aesthetic_cls.return_value
        mock_aesthetic_instance.return_value.item.return_value = 7.5

        # 5. Setup Vision Scanner (NEW)
        mock_vision_instance = mock_vision_cls.return_value
        mock_vision_instance.extract_caption.return_value = "A detailed caption of a test image"

        # 6. Setup Text Embedder (NEW)
        mock_embedder_instance = mock_embedder_cls.return_value
        # Mocking a 5-dimension vector for testing
        mock_embedder_instance.embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]

        # 7. Defaults for Utils
        mock_too_small.return_value = False
        mock_exif_ts.return_value = datetime(2023, 1, 1, 12, 0, 0) # Default EXIF TS
        mock_disk_ts.return_value = datetime(2024, 1, 1, 12, 0, 0) # Default Disk TS
        mock_iso.return_value = 100

        yield {
            "cv2": mock_cv2,
            "face_app": instance_fa,
            "clip_model": mock_clip_model,
            "pose": mock_pose_extractor,
            "vision": mock_vision_instance,
            "embedder": mock_embedder_instance,
            "exif_ts": mock_exif_ts,
            "disk_ts": mock_disk_ts,
            "iso": mock_iso,
            "is_face_too_small": mock_too_small
        }

@pytest.fixture
def extractor(mock_dependencies):
    """Returns an instance of FeatureExtractor with mocks injected."""
    return FeatureExtractor(use_gpu=False)

@pytest.fixture
def sample_image():
    """Creates a dummy black image (100x100)."""
    return np.zeros((100, 100, 3), dtype=np.uint8)

@pytest.fixture
def mock_face_obj():
    """Creates a mock object resembling what InsightFace returns."""
    # InsightFace returns objects with attributes, not dicts
    face = MagicMock()
    face.bbox = np.array([10, 10, 50, 50], dtype=np.float32) # x1, y1, x2, y2
    face.embedding = np.random.rand(512)
    face.det_score = 0.99
    face.yaw = 0.0
    face.pitch = 0.0
    return face

def test_initialization(mock_dependencies):
    """Test that models are loaded and prepared correctly."""
    FeatureExtractor(use_gpu=False)

    # Assert FaceAnalysis was initialized and prepared
    mock_dependencies["face_app"].prepare.assert_called_once()
    # Assert CLIP was loaded
    assert mock_dependencies["clip_model"] is not None

def test_process_image_file_not_found(extractor, mock_dependencies):
    """Test handling of invalid image paths."""
    mock_dependencies["cv2"].imread.return_value = None

    result = extractor.process_image("non_existent.jpg", "non_existent_raw.heic")
    assert result is None


def test_process_image_happy_path(extractor, mock_dependencies, sample_image, mock_face_obj):
    """Test a standard successful image processing flow."""
    # 1. Mock reading the image
    mock_dependencies["cv2"].imread.return_value = sample_image
    mock_dependencies["cv2"].cvtColor.return_value = sample_image

    # 2. Mock face detection finding 1 face
    mock_dependencies["face_app"].get.return_value = [mock_face_obj]

    # 3. Mock pose extraction (Must return list of tuples matching number of faces)
    # (yaw, pitch, roll)
    mock_dependencies["pose"].return_value.extract_pose_from_faces.return_value = [(10.0, 20.0, 5.0)]

    # 4. Mock CLIP encoding
    # CLIP encode_image returns a tensor
    mock_tensor = MagicMock()
    mock_tensor.norm.return_value = 1.0  # Simplify normalization
    mock_tensor.__itruediv__.return_value = mock_tensor
    mock_tensor.cpu().numpy.return_value = np.array([[0.1, 0.2, 0.3]])  # The vector
    mock_dependencies["clip_model"].encode_image.return_value = mock_tensor

    # Run
    result = extractor.process_image("test.jpg", "test_raw.jpg")

    # Assertions
    assert isinstance(result, ImageAnalysisResult)
    assert len(result.faces) == 1
    assert result.original_width == 100
    assert result.original_height == 100
    assert result.aesthetic_score == 7.5
    assert result.original_path == "test_raw.jpg"

    # Check Face Data
    face = result.faces[0]
    assert face.confidence == 0.99
    assert face.yaw == 10.0
    assert face.pitch == 20.0
    assert face.shot_type != ""  # Should be calculated

    # Check Semantic Vector
    assert result.semantic_vector.shape == (3,)

    # Check the caption
    assert result.caption == "A detailed caption of a test image"
    assert result.caption_vector == [0.1, 0.2, 0.3, 0.4, 0.5]

    mock_dependencies["vision"].extract_caption.assert_called_with("test.jpg")
    mock_dependencies["embedder"].embed.assert_called_with("A detailed caption of a test image")

def test_full_flow_happy_path(extractor, mock_dependencies, sample_image, mock_face_obj):
    """
    Verifies the complete flow including:
    - Face Detection
    - Pose Extraction
    - CLIP Encoding
    - Metadata (ISO/Timestamp)
    """
    # Setup
    mock_dependencies["cv2"].imread.return_value = sample_image
    mock_dependencies["cv2"].cvtColor.return_value = sample_image
    mock_dependencies["face_app"].get.return_value = [mock_face_obj]
    mock_dependencies["pose"].return_value.extract_pose_from_faces.return_value = [(10.0, 20.0, 5.0)]

    # Run
    result = extractor.process_image("test.jpg", "raw.jpg")

    # Assertions
    assert result is not None
    assert len(result.faces) == 1
    assert result.iso == 100
    assert result.timestamp == datetime(2023, 1, 1, 12, 0, 0)

    # Check if Global Stats were calculated (defaults for zero-image are 0.0)
    assert result.global_blur == 0.0
    assert result.global_brightness == 0.0

    # Check Face Data injection
    assert result.faces[0].yaw == 10.0  # From Pose Extractor
    assert result.faces[0].pitch == 20.0

    # Check the caption
    assert result.caption == "A detailed caption of a test image"
    assert result.caption_vector == [0.1, 0.2, 0.3, 0.4, 0.5]

    mock_dependencies["vision"].extract_caption.assert_called_with("test.jpg")
    mock_dependencies["embedder"].embed.assert_called_with("A detailed caption of a test image")

def test_timestamp_fallback_logic(extractor, mock_dependencies, sample_image):
    """
    CRITICAL: Tests that if EXIF timestamp is missing, it falls back to Disk timestamp.
    """
    mock_dependencies["cv2"].imread.return_value = sample_image
    mock_dependencies["cv2"].cvtColor.return_value = sample_image
    mock_dependencies["face_app"].get.return_value = [] # No faces needed

    # 1. Simulate Missing EXIF
    mock_dependencies["exif_ts"].return_value = None

    # 2. Simulate Existing Disk Time
    expected_disk_time = datetime(2025, 5, 20)
    mock_dependencies["disk_ts"].return_value = expected_disk_time

    result = extractor.process_image("test.jpg", "raw.jpg")

    assert result.timestamp == expected_disk_time
    mock_dependencies["exif_ts"].assert_called_once()
    mock_dependencies["disk_ts"].assert_called_once()

def test_iso_missing_handling(extractor, mock_dependencies, sample_image):
    """
    Verifies that if ISO is None, the result object contains None (not 0 or -1).
    """
    mock_dependencies["cv2"].imread.return_value = sample_image
    mock_dependencies["cv2"].cvtColor.return_value = sample_image
    mock_dependencies["face_app"].get.return_value = []

    # Simulate ISO missing (e.g. Screenshot)
    mock_dependencies["iso"].return_value = None

    result = extractor.process_image("test.jpg", "raw.jpg")

    assert result.iso is None

def test_global_visual_stats_computation():
    """
    Unit test for the helper function `compute_global_visual_stats`.
    We create a real numpy image to verify the math works.
    """
    # Create a 100x100 grayscale image with a gradient (so contrast > 0)
    img = np.linspace(0, 255, 10000).reshape(100, 100).astype(np.uint8)
    # Convert to BGR for the function
    img_bgr = np.stack((img,)*3, axis=-1)

    stats = compute_global_visual_stats(img_bgr)

    # Gradient image should have significant contrast
    assert stats["global_contrast"] > 0.0
    # Mean brightness should be roughly 127.5
    assert 120 < stats["global_brightness"] < 135
    # Blur score (Laplacian) might be low for a smooth gradient, but should be float
    assert isinstance(stats["global_blur"], float)

def test_corrupt_image_handling(extractor, mock_dependencies):
    """Test that cv2.imread failing returns None safely."""
    mock_dependencies["cv2"].imread.return_value = None

    result = extractor.process_image("corrupt.jpg", "raw.jpg")

    assert result is None

def test_face_filtering_integration(extractor, mock_dependencies, sample_image, mock_face_obj):
    """
    Test that the pipeline correctly removes small faces before sending to Pose/CLIP.
    """
    mock_dependencies["cv2"].imread.return_value = sample_image
    mock_dependencies["cv2"].cvtColor.return_value = sample_image
    mock_dependencies["face_app"].get.return_value = [mock_face_obj, mock_face_obj]

    # First face kept, Second face dropped
    mock_dependencies["is_face_too_small"].side_effect = [False, True]

    # Mock Pose to expect only 1 face
    mock_dependencies["pose"].return_value.extract_pose_from_faces.return_value = [(10,10,10)]

    result = extractor.process_image("test.jpg", "raw.jpg")

    assert len(result.faces) == 1

def test_filter_small_faces(extractor, mock_dependencies, sample_image, mock_face_obj):
    """Test that faces marked as 'too small' are excluded."""
    mock_dependencies["cv2"].imread.return_value = sample_image
    mock_dependencies["cv2"].cvtColor.return_value = sample_image

    # Create two faces
    face_large = mock_face_obj
    face_small = MagicMock()
    face_small.bbox = np.array([0, 0, 5, 5])  # Tiny face
    face_small.embedding = np.zeros(512)
    face_small.det_score = 0.5

    mock_dependencies["face_app"].get.return_value = [face_large, face_small]

    # Mock is_face_too_small to return True for the second call
    # side_effect allows us to define return values for sequential calls
    mock_dependencies["is_face_too_small"].side_effect = [False, True]

    # Pose extractor should only receive the ONE kept face
    mock_dependencies["pose"].return_value.extract_pose_from_faces.return_value = [(0, 0, 0)]

    # Mock CLIP
    mock_tensor = MagicMock()
    mock_tensor.norm.return_value = 1.0
    mock_tensor.cpu().numpy.return_value = np.zeros((1, 512))
    mock_dependencies["clip_model"].encode_image.return_value = mock_tensor

    result = extractor.process_image("test.jpg", "test_raw.jpg")

    # We should only have 1 face remaining
    assert len(result.faces) == 1
    # Verify the pose extractor was called with the filtered list
    kept_faces_arg = mock_dependencies["pose"].return_value.extract_pose_from_faces.call_args[0][1]
    assert len(kept_faces_arg) == 1
    assert kept_faces_arg[0] == face_large


def test_shot_type_calculation(extractor, sample_image):
    """
    Unit test for shot type logic based on bbox area vs image area.
    Image is 100x100 (Area = 10,000)
    """
    # 1. Close-up (> 25% area) -> 60x60 = 3600 (36%)
    bbox_close = [0, 0, 60, 60]
    assert calculate_shot_type(sample_image, bbox_close) == "Close-up"

    # 2. Medium-Shot (> 8% area) -> 30x30 = 900 (9%)
    bbox_medium = [0, 0, 30, 30]
    assert calculate_shot_type(sample_image, bbox_medium) == "Medium-Shot"

    # 3. Full-Body (< 8% area) -> 10x10 = 100 (1%)
    bbox_full = [0, 0, 10, 10]
    assert calculate_shot_type(sample_image, bbox_full) == "Full-Body"


def test_face_quality_metrics(extractor, sample_image, mock_dependencies):
    """Test that blur and brightness don't crash and return floats."""
    # 1. Mock Color Conversion
    # The code calls cv2.cvtColor to get 'gray'.
    # We must make it return a real numpy array so np.mean(gray) works later.
    mock_dependencies["cv2"].cvtColor.return_value = np.zeros((60, 60), dtype=np.uint8)

    # 2. Mock Laplacian Variance
    # The code calls: cv2.Laplacian(...).var()
    # We tell the mock: "When .var() is called on the result of Laplacian, return 10.5"
    mock_dependencies["cv2"].Laplacian.return_value.var.return_value = 10.5

    # Create a simple bbox in the middle
    bbox = [20, 20, 80, 80]

    blur, bright = calculate_face_quality(sample_image, bbox)

    assert isinstance(blur, float)
    assert isinstance(bright, float)
    # Since image is all black (zeros), brightness should be 0
    assert bright == 0.0


def test_error_propagation(extractor, mock_dependencies):
    """Test that exceptions inside process_image are raised correctly."""
    mock_dependencies["cv2"].imread.side_effect = Exception("Disk Error")

    with pytest.raises(Exception) as excinfo:
        extractor.process_image("broken.jpg", "broken_raw.jpg")

    assert "Disk Error" in str(excinfo.value)

def test_process_image_returns_valid_dataclass(extractor, mock_dependencies, sample_image, mock_face_obj):
    # Setup InsightFace return data
    mock_dependencies["face_app"].get.return_value = [mock_face_obj]
    mock_dependencies["cv2"].imread.return_value = sample_image
    mock_dependencies["cv2"].cvtColor.return_value = sample_image

    # Setup CLIP return value
    # Clip returns a Tensor.encode_image()
    # encode_image() -> result -> result.norm() -> .... -> result.cpu().numpy()
    mock_tensor = MagicMock()
    # Mock the normalization division behavior
    mock_tensor.norm.return_value = 1.0  # Simplify normalization
    mock_tensor.cpu().numpy.return_value = np.array([[0.1, 0.2, 0.3]])  # The vector
    mock_dependencies["clip_model"].encode_image.return_value = mock_tensor

    # Ensure the in-place division (/=) returns the SAME mock object
    # otherwise configuration below is lost on the result of the division
    mock_tensor.__itruediv__.return_value = mock_tensor
    mock_dependencies["clip_model"].mock_clip_model.encode_image.return_value = mock_tensor
    mock_dependencies["pose"].return_value.extract_pose_from_faces.return_value = [(10.0, 20.0, 5.0)]

    # Mock cv2.imread s we don't need a real image
    result = extractor.process_image("fake_path.jpg", "fake_raw_path.jpg")

    assert isinstance(result, ImageAnalysisResult)
    assert len(result.faces) == 1
    assert result.original_width == 100
    assert result.original_height == 100
    assert result.original_path == "fake_raw_path.jpg"
    assert result.display_path == "fake_path.jpg"
    assert result.aesthetic_score == 7.5

    # Check Faces
    face = result.faces[0]
    assert face.confidence == 0.99
    assert face.yaw == 10.0
    assert face.pitch == 20.0
    assert face.shot_type != ""  # Should be calculated

    assert result.semantic_vector.shape == (3,)

def test_captioning_failure_handling(extractor, mock_dependencies, sample_image):
    """
    Test that if the vision model returns an empty string (failure case),
    the embedder handles it safely and we still get a valid result object.
    """
    # Setup standard mocks
    mock_dependencies["cv2"].imread.return_value = sample_image
    mock_dependencies["cv2"].cvtColor.return_value = sample_image
    mock_dependencies["face_app"].get.return_value = []

    # Mock CLIP
    mock_tensor = MagicMock()
    mock_tensor.norm.return_value = 1.0
    mock_tensor.cpu().numpy.return_value = np.zeros((1, 512))
    mock_dependencies["clip_model"].encode_image.return_value = mock_tensor

    # --- 1. SIMULATE FAILURE ---
    # Vision model returns empty string
    mock_dependencies["vision"].extract_caption.return_value = ""

    # Embedder returns None when given empty string (assuming your Embedder logic does this)
    mock_dependencies["embedder"].embed.return_value = None

    # Run
    result = extractor.process_image("test.jpg", "raw.jpg")

    # Assertions
    assert result is not None
    assert result.caption == ""
    assert result.caption_vector is None  # Should be None, not crash
