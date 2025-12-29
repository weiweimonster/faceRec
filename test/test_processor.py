import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.ingestion.processor import FeatureExtractor
from src.common.types import ImageAnalysisResult
import sys
from src.util.image_util import calculate_shot_type, calculate_face_quality

# We mock these 'src' imports before importing the actual module
# so the test doesn't crash if these files aren't in the python path during testing.
sys.modules['src.pose.pose_extractor'] = MagicMock()
sys.modules['src.util.image_util'] = MagicMock()

@pytest.fixture
def mock_dependencies():
    with patch('src.ingestion.processor.FaceAnalysis') as mock_fa, \
        patch('src.ingestion.processor.clip.load') as mock_clip, \
        patch('src.ingestion.processor.PoseExtractor') as mock_pose_extractor, \
        patch('src.ingestion.processor.cv2') as mock_cv2, \
        patch('src.ingestion.processor.torch') as mock_torch, \
        patch('src.util.image_util.is_face_too_small') as mock_too_small:
        # Setup FaceAnalysis mock
        instance_fa = mock_fa.return_value
        instance_fa.prepare.return_value = None

        # Setup CLIP mock (returns model, preprocess)
        mock_clip_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_clip.return_value = (mock_clip_model, mock_preprocess)

        # Setup Torch mock for device check
        mock_torch.cuda.is_available.return_value = False

        # Setup generic response for is_face_too_small (default to False/Keep face)
        mock_too_small.return_value = False

        yield {
            "cv2": mock_cv2,
            "face_app": instance_fa,
            "clip_model": mock_clip_model,
            "clip_preprocess": mock_preprocess,
            "pose": mock_pose_extractor,
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

    result = extractor.process_image("non_existent.jpg")
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
    result = extractor.process_image("test.jpg")

    # Assertions
    assert isinstance(result, ImageAnalysisResult)
    assert len(result.faces) == 1
    assert result.original_width == 100
    assert result.original_height == 100

    # Check Face Data
    face = result.faces[0]
    assert face.confidence == 0.99
    assert face.yaw == 10.0
    assert face.pitch == 20.0
    assert face.shot_type != ""  # Should be calculated

    # Check Semantic Vector
    assert result.semantic_vector.shape == (3,)


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

    result = extractor.process_image("test.jpg")

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
        extractor.process_image("broken.jpg")

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
    result = extractor.process_image("fake_path.jpg")

    assert isinstance(result, ImageAnalysisResult)
    assert len(result.faces) == 1
    assert result.original_width == 100
    assert result.original_height == 100

    # Check Faces
    face = result.faces[0]
    assert face.confidence == 0.99
    assert face.yaw == 10.0
    assert face.pitch == 20.0
    assert face.shot_type != ""  # Should be calculated

    assert result.semantic_vector.shape == (3,)
