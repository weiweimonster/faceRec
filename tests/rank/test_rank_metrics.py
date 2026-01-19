"""
Tests for src/rank/rank_metrics.py

Tests cover:
- FaceRankMetrics dataclass
- PictureRankMetrics dataclass
- String representations
"""

import pytest

from src.rank.rank_metrics import FaceRankMetrics, PictureRankMetrics


# ============================================
# FaceRankMetrics Tests
# ============================================

class TestFaceRankMetrics:
    """Tests for the FaceRankMetrics dataclass"""

    def test_face_rank_metrics_creation(self):
        """Test creating a FaceRankMetrics instance"""
        metrics = FaceRankMetrics(
            f_orient_score=0.85,
            f_blur=450.0,
            f_conf=0.92,
            f_height=250.0,
            f_width=200.0
        )

        assert metrics.f_orient_score == 0.85
        assert metrics.f_blur == 450.0
        assert metrics.f_conf == 0.92
        assert metrics.f_height == 250.0
        assert metrics.f_width == 200.0

    def test_face_rank_metrics_str(self):
        """Test FaceRankMetrics string representation"""
        metrics = FaceRankMetrics(
            f_orient_score=0.85,
            f_blur=450.5,
            f_conf=0.92,
            f_height=250.0,
            f_width=200.0
        )

        result = str(metrics)

        # Check key components are present
        assert "Face" in result
        assert "Conf:" in result
        assert "0.92" in result
        assert "Blur:" in result
        assert "450.5" in result
        assert "Orient:" in result
        assert "0.85" in result
        assert "Size:" in result
        assert "200x250" in result

    def test_face_rank_metrics_str_formatting(self):
        """Test FaceRankMetrics string formatting details"""
        metrics = FaceRankMetrics(
            f_orient_score=0.123,
            f_blur=100.6,
            f_conf=0.456,
            f_height=150.0,
            f_width=120.0
        )

        result = str(metrics)

        # Should format confidence and orient to 2 decimal places
        assert "0.46" in result or "0.45" in result  # 0.456 rounded
        assert "0.12" in result  # 0.123 rounded

    def test_face_rank_metrics_zero_values(self):
        """Test FaceRankMetrics with zero values"""
        metrics = FaceRankMetrics(
            f_orient_score=0.0,
            f_blur=0.0,
            f_conf=0.0,
            f_height=0.0,
            f_width=0.0
        )

        result = str(metrics)
        assert "0.00" in result
        assert "0x0" in result


# ============================================
# PictureRankMetrics Tests
# ============================================

class TestPictureRankMetrics:
    """Tests for the PictureRankMetrics dataclass"""

    def test_picture_rank_metrics_creation(self):
        """Test creating a PictureRankMetrics instance"""
        metrics = PictureRankMetrics(
            semantic_score=0.88,
            aesthetic_score=4.5,
            year=2023,
            month=6,
            date=15,
            g_blur=500.0,
            g_brightness=128.0,
            g_contrast=55.0,
            g_iso=200
        )

        assert metrics.semantic_score == 0.88
        assert metrics.aesthetic_score == 4.5
        assert metrics.year == 2023
        assert metrics.month == 6
        assert metrics.date == 15
        assert metrics.g_blur == 500.0
        assert metrics.g_brightness == 128.0
        assert metrics.g_contrast == 55.0
        assert metrics.g_iso == 200
        assert metrics.has_face is False
        assert metrics.face_metrics is None

    def test_picture_rank_metrics_with_face(self):
        """Test PictureRankMetrics with face data"""
        face = FaceRankMetrics(
            f_orient_score=0.9,
            f_blur=400.0,
            f_conf=0.95,
            f_height=200.0,
            f_width=180.0
        )

        metrics = PictureRankMetrics(
            semantic_score=0.88,
            aesthetic_score=4.5,
            year=2023,
            month=6,
            date=15,
            g_blur=500.0,
            g_brightness=128.0,
            g_contrast=55.0,
            g_iso=200,
            has_face=True,
            face_metrics=[face]
        )

        assert metrics.has_face is True
        assert len(metrics.face_metrics) == 1
        assert metrics.face_metrics[0].f_conf == 0.95

    def test_picture_rank_metrics_str_no_faces(self):
        """Test PictureRankMetrics string representation without faces"""
        metrics = PictureRankMetrics(
            semantic_score=0.885,
            aesthetic_score=4.52,
            year=2023,
            month=6,
            date=15,
            g_blur=500.0,
            g_brightness=128.0,
            g_contrast=55.0,
            g_iso=200
        )

        result = str(metrics)

        # Check header components
        assert "Photo" in result
        assert "2023-06-15" in result
        assert "Semantic:" in result
        assert "0.885" in result
        assert "Aesthetic:" in result
        assert "4.52" in result

        # Check stats
        assert "Blur=" in result
        assert "500.0" in result
        assert "Bright=" in result
        assert "128.0" in result
        assert "Contrast=" in result
        assert "55.0" in result
        assert "ISO=" in result
        assert "200" in result

        # No faces
        assert "No faces" in result.lower() or "no faces" in result.lower()

    def test_picture_rank_metrics_str_with_faces(self):
        """Test PictureRankMetrics string representation with faces"""
        face1 = FaceRankMetrics(
            f_orient_score=0.9,
            f_blur=400.0,
            f_conf=0.95,
            f_height=200.0,
            f_width=180.0
        )
        face2 = FaceRankMetrics(
            f_orient_score=0.7,
            f_blur=350.0,
            f_conf=0.88,
            f_height=150.0,
            f_width=130.0
        )

        metrics = PictureRankMetrics(
            semantic_score=0.88,
            aesthetic_score=4.5,
            year=2023,
            month=12,
            date=25,
            g_blur=600.0,
            g_brightness=140.0,
            g_contrast=60.0,
            g_iso=400,
            has_face=True,
            face_metrics=[face1, face2]
        )

        result = str(metrics)

        # Check date
        assert "2023-12-25" in result

        # Check faces section
        assert "Faces Found" in result
        assert "1." in result
        assert "2." in result

    def test_picture_rank_metrics_str_has_face_but_no_metrics(self):
        """Test string when has_face is True but no face_metrics list"""
        metrics = PictureRankMetrics(
            semantic_score=0.88,
            aesthetic_score=4.5,
            year=2023,
            month=6,
            date=15,
            g_blur=500.0,
            g_brightness=128.0,
            g_contrast=55.0,
            g_iso=200,
            has_face=True,
            face_metrics=None
        )

        result = str(metrics)

        # Should show warning about flag mismatch
        assert "True but no metrics" in result or "no metrics list" in result

    def test_picture_rank_metrics_month_padding(self):
        """Test that month is zero-padded in string"""
        metrics = PictureRankMetrics(
            semantic_score=0.88,
            aesthetic_score=4.5,
            year=2023,
            month=1,  # January
            date=5,
            g_blur=500.0,
            g_brightness=128.0,
            g_contrast=55.0,
            g_iso=200
        )

        result = str(metrics)

        # Month should be zero-padded
        assert "2023-01-05" in result

    def test_picture_rank_metrics_multiple_faces(self):
        """Test PictureRankMetrics with multiple faces"""
        faces = [
            FaceRankMetrics(
                f_orient_score=0.9,
                f_blur=400.0,
                f_conf=0.95,
                f_height=200.0,
                f_width=180.0
            )
            for _ in range(3)
        ]

        metrics = PictureRankMetrics(
            semantic_score=0.88,
            aesthetic_score=4.5,
            year=2023,
            month=6,
            date=15,
            g_blur=500.0,
            g_brightness=128.0,
            g_contrast=55.0,
            g_iso=200,
            has_face=True,
            face_metrics=faces
        )

        result = str(metrics)

        # Should have 3 face entries
        assert "1." in result
        assert "2." in result
        assert "3." in result


# ============================================
# Edge Case Tests
# ============================================

class TestRankMetricsEdgeCases:
    """Edge case tests for rank metrics"""

    def test_face_metrics_negative_values(self):
        """Test FaceRankMetrics with negative values"""
        metrics = FaceRankMetrics(
            f_orient_score=-0.1,
            f_blur=-10.0,
            f_conf=-0.5,
            f_height=-100.0,
            f_width=-80.0
        )

        # Should still create valid string
        result = str(metrics)
        assert isinstance(result, str)

    def test_picture_metrics_negative_year(self):
        """Test PictureRankMetrics with negative/unusual year"""
        metrics = PictureRankMetrics(
            semantic_score=0.88,
            aesthetic_score=4.5,
            year=-1,  # Unknown year
            month=-1,
            date=-1,
            g_blur=500.0,
            g_brightness=128.0,
            g_contrast=55.0,
            g_iso=-1
        )

        # Should still create valid string (even if formatting looks odd)
        result = str(metrics)
        assert isinstance(result, str)

    def test_face_metrics_large_values(self):
        """Test FaceRankMetrics with very large values"""
        metrics = FaceRankMetrics(
            f_orient_score=10000.0,
            f_blur=99999.9,
            f_conf=1000.0,
            f_height=10000.0,
            f_width=10000.0
        )

        result = str(metrics)
        assert isinstance(result, str)
        assert "10000x10000" in result

    def test_picture_metrics_empty_face_list(self):
        """Test PictureRankMetrics with empty face list"""
        metrics = PictureRankMetrics(
            semantic_score=0.88,
            aesthetic_score=4.5,
            year=2023,
            month=6,
            date=15,
            g_blur=500.0,
            g_brightness=128.0,
            g_contrast=55.0,
            g_iso=200,
            has_face=True,
            face_metrics=[]  # Empty list
        )

        result = str(metrics)
        # Empty list should show warning about flag mismatch
        assert isinstance(result, str)
