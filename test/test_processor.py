import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.ingestion.processor import FeatureExtractor, ImageAnalysisResult

@pytest.fixture
def mock_engine():
    """
    Initialize Feature Extractor but replaces the havy AI models with Mocks
    """
    with patch("src.ingestion.processor.FaceAnalysis") as MockFaceApp, \
        patch("src.ingestion.processor.clip.load") as MockClip:

        # Setup Mock InsightFace
        mock_face_instance = MockFaceApp.return_value
        mock_face_instance.prepare.return_value = None

        # Setup Mock CLIP
        mock_clip_model = MagicMock()
        mock_preprocess = MagicMock()
        # clip.load returns (model, preprocess)
        MockClip.return_value = (mock_clip_model, mock_preprocess)

        engine = FeatureExtractor(use_gpu=False)
        # Attach mocks to instance for assertion
        engine.mock_face_app = mock_face_instance
        engine.mock_clip_model = mock_clip_model

        return engine

def test_process_image_returns_valid_dataclass(mock_engine):
    # Setup InsightFace return data
    mock_face = MagicMock()
    mock_face.bbox = np.array([10, 15, 45, 50])
    mock_face.embedding = np.random.rand(512).astype(np.float32)
    mock_face.det_score = 0.98
    mock_engine.mock_face_app.get.return_value = [mock_face]

    # Setup CLIP return value
    # Clip returns a Tensor.encode_image()
    # encode_image() -> result -> result.norm() -> .... -> result.cpu().numpy()
    mock_tensor = MagicMock()
    # Mock the normalization division behavior
    mock_tensor.norm.return_value = 1.0

    # Ensure the in-place division (/=) returns the SAME mock object
    # otherwise configuration below is lost on the result of the division
    mock_tensor.__itruediv__.return_value = mock_tensor

    mock_tensor.cpu.return_value.numpy.return_value = np.random.rand(1, 512).astype(np.float32)

    mock_engine.mock_clip_model.encode_image.return_value = mock_tensor

    # Mock cv2.imread s we don't need a real image
    with patch("src.ingestion.processor.cv2.imread") as mock_imread:
        mock_imread.return_value = np.zeros((640, 480, 3), dtype=np.uint8)

        result = mock_engine.process_image("fake_path.jpg")

    assert isinstance(result, ImageAnalysisResult)
    assert result.original_width == 480
    assert result.original_height == 640

    # Check Faces
    assert len(result.faces) == 1
    assert result.faces[0].confidence == 0.98
    assert isinstance(result.faces[0].bbox, list)
    assert result.faces[0].bbox == [10, 15, 45, 50]

    assert result.semantic_vector.shape == (512,)

def test_process_image_returns_none_one_bad_file(mock_engine):
    with patch("src.ingestion.processor.cv2.imread") as mock_imread:
        mock_imread.return_value = None
        result = mock_engine.process_image("missing.jpg")
    assert result is None
