"""
Tests for ingestion metrics collection.

Tests cover:
- IngestionMetricsTracker class
- MetricsCollectionStage
"""

import pytest
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

from src.ingestion.metrics_tracker import IngestionMetricsTracker
from src.ingestion.stages.metrics_collection import MetricsCollectionStage
from src.ingestion.stages.base import PipelineItem
from src.common.types import ImageAnalysisResult, FaceData


# ============================================
# IngestionMetricsTracker Tests
# ============================================

class TestIngestionMetricsTrackerInit:
    """Tests for IngestionMetricsTracker initialization"""

    def test_init_creates_empty_tracker(self):
        """Test that initialization creates empty data structures"""
        tracker = IngestionMetricsTracker()
        assert tracker.total_photos == 0
        assert tracker.total_faces == 0
        assert len(tracker.numeric_data) == 0
        assert len(tracker.missing_counts) == 0

    def test_init_loads_trackable_features(self):
        """Test that trackable features are loaded from registry"""
        tracker = IngestionMetricsTracker()
        assert len(tracker.trackable_features) > 0
        # Should include features with track_in_metrics=True
        assert "aesthetic_score" in tracker.trackable_features
        # Should exclude features with track_in_metrics=False
        assert "semantic_score" not in tracker.trackable_features


class TestIngestionMetricsTrackerUpdate:
    """Tests for IngestionMetricsTracker.update method"""

    def test_update_increments_photo_count(self):
        """Test that update increments total_photos"""
        tracker = IngestionMetricsTracker()
        result = ImageAnalysisResult(
            original_path="/test.jpg",
            photo_id="1",
            display_path="/test.jpg",
        )
        tracker.update(result)
        assert tracker.total_photos == 1

    def test_update_counts_faces(self):
        """Test that update counts faces correctly"""
        tracker = IngestionMetricsTracker()
        result = ImageAnalysisResult(
            original_path="/test.jpg",
            photo_id="1",
            display_path="/test.jpg",
            faces=[FaceData(), FaceData(), FaceData()],
        )
        tracker.update(result)
        assert tracker.total_faces == 3

    def test_update_tracks_numeric_values(self):
        """Test that update tracks numeric feature values"""
        tracker = IngestionMetricsTracker()
        result = ImageAnalysisResult(
            original_path="/test.jpg",
            photo_id="1",
            display_path="/test.jpg",
            aesthetic_score=5.2,
            global_blur=150.0,
            global_brightness=128.0,
        )
        tracker.update(result)
        assert "aesthetic_score" in tracker.numeric_data
        assert 5.2 in tracker.numeric_data["aesthetic_score"]

    def test_update_tracks_missing_values(self):
        """Test that update counts missing (None) values"""
        tracker = IngestionMetricsTracker()
        result = ImageAnalysisResult(
            original_path="/test.jpg",
            photo_id="1",
            display_path="/test.jpg",
            aesthetic_score=None,  # Missing
        )
        tracker.update(result)
        assert "aesthetic_score" in tracker.missing_counts

    def test_update_handles_none_result(self):
        """Test that update handles None result gracefully"""
        tracker = IngestionMetricsTracker()
        tracker.update(None)
        assert tracker.total_photos == 0

    def test_update_multiple_results(self):
        """Test updating with multiple results"""
        tracker = IngestionMetricsTracker()
        for i in range(5):
            result = ImageAnalysisResult(
                original_path=f"/test{i}.jpg",
                photo_id=str(i),
                display_path=f"/test{i}.jpg",
                aesthetic_score=4.0 + i * 0.5,
            )
            tracker.update(result)
        assert tracker.total_photos == 5
        assert len(tracker.numeric_data["aesthetic_score"]) == 5


class TestIngestionMetricsTrackerFinalizeReport:
    """Tests for IngestionMetricsTracker.finalize_report method"""

    def test_finalize_report_returns_dict(self):
        """Test that finalize_report returns a dictionary"""
        tracker = IngestionMetricsTracker()
        result = ImageAnalysisResult(
            original_path="/test.jpg",
            photo_id="1",
            display_path="/test.jpg",
            aesthetic_score=5.0,
        )
        tracker.update(result)
        report = tracker.finalize_report()
        assert isinstance(report, dict)

    def test_finalize_report_contains_summary(self):
        """Test that report contains summary section"""
        tracker = IngestionMetricsTracker()
        result = ImageAnalysisResult(
            original_path="/test.jpg",
            photo_id="1",
            display_path="/test.jpg",
            faces=[FaceData(), FaceData()],
        )
        tracker.update(result)
        report = tracker.finalize_report()
        assert "summary" in report
        assert report["summary"]["total_photos"] == 1
        assert report["summary"]["total_faces"] == 2

    def test_finalize_report_contains_numeric_stats(self):
        """Test that report contains numeric statistics"""
        tracker = IngestionMetricsTracker()
        for i in range(10):
            result = ImageAnalysisResult(
                original_path=f"/test{i}.jpg",
                photo_id=str(i),
                display_path=f"/test{i}.jpg",
                aesthetic_score=4.0 + i * 0.2,
            )
            tracker.update(result)
        report = tracker.finalize_report()
        assert "numeric_stats" in report
        if "aesthetic_score" in report["numeric_stats"]:
            stats = report["numeric_stats"]["aesthetic_score"]
            assert "count" in stats
            assert "min" in stats
            assert "max" in stats
            assert "mean" in stats
            assert "p05" in stats
            assert "p95" in stats

    def test_finalize_report_saves_to_file(self):
        """Test that finalize_report saves to JSON file"""
        tracker = IngestionMetricsTracker()
        result = ImageAnalysisResult(
            original_path="/test.jpg",
            photo_id="1",
            display_path="/test.jpg",
            aesthetic_score=5.0,
        )
        tracker.update(result)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name

        try:
            tracker.finalize_report(output_path)
            assert os.path.exists(output_path)
            with open(output_path, 'r') as f:
                saved_report = json.load(f)
            assert "summary" in saved_report
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


# ============================================
# MetricsCollectionStage Tests
# ============================================

class TestMetricsCollectionStageInit:
    """Tests for MetricsCollectionStage initialization"""

    def test_init_default_output_path(self):
        """Test default output path"""
        stage = MetricsCollectionStage()
        assert stage.output_path == "ingestion_metrics.json"

    def test_init_custom_output_path(self):
        """Test custom output path"""
        stage = MetricsCollectionStage(output_path="custom_metrics.json")
        assert stage.output_path == "custom_metrics.json"

    def test_init_creates_tracker(self):
        """Test that initialization creates a tracker"""
        stage = MetricsCollectionStage()
        assert stage.tracker is not None
        assert isinstance(stage.tracker, IngestionMetricsTracker)


class TestMetricsCollectionStageProperties:
    """Tests for MetricsCollectionStage properties"""

    def test_name_property(self):
        """Test stage name property"""
        stage = MetricsCollectionStage()
        assert stage.name == "metrics_collection"

    def test_supports_gpu_batching_property(self):
        """Test that stage doesn't support GPU batching"""
        stage = MetricsCollectionStage()
        assert stage.supports_gpu_batching is False


class TestMetricsCollectionStageProcess:
    """Tests for MetricsCollectionStage.process method"""

    def test_process_collects_metrics(self):
        """Test that process collects metrics from items"""
        stage = MetricsCollectionStage()

        # Create pipeline items
        items = []
        for i in range(3):
            result = ImageAnalysisResult(
                original_path=f"/test{i}.jpg",
                photo_id=str(i),
                display_path=f"/test{i}.jpg",
                aesthetic_score=5.0,
            )
            item = PipelineItem(result=result)
            items.append(item)

        processed = stage.process(items)

        assert stage.tracker.total_photos == 3
        assert processed == items  # Pass-through

    def test_process_skips_errored_items(self):
        """Test that process skips items with errors"""
        stage = MetricsCollectionStage()

        result = ImageAnalysisResult(
            original_path="/test.jpg",
            photo_id="1",
            display_path="/test.jpg",
        )
        item = PipelineItem(result=result, error="Some error")
        items = [item]

        stage.process(items)
        assert stage.tracker.total_photos == 0

    def test_process_skips_skipped_items(self):
        """Test that process skips items marked as skipped"""
        stage = MetricsCollectionStage()

        result = ImageAnalysisResult(
            original_path="/test.jpg",
            photo_id="1",
            display_path="/test.jpg",
        )
        item = PipelineItem(result=result, skipped=True)
        items = [item]

        stage.process(items)
        assert stage.tracker.total_photos == 0

    def test_process_returns_items_unchanged(self):
        """Test that process returns items unchanged (pass-through)"""
        stage = MetricsCollectionStage()

        result = ImageAnalysisResult(
            original_path="/test.jpg",
            photo_id="1",
            display_path="/test.jpg",
            aesthetic_score=5.0,
        )
        item = PipelineItem(result=result)
        items = [item]

        processed = stage.process(items)
        assert processed is items
        assert processed[0].result.aesthetic_score == 5.0


class TestMetricsCollectionStageCleanup:
    """Tests for MetricsCollectionStage.cleanup method"""

    def test_cleanup_generates_report(self):
        """Test that cleanup generates report when photos processed"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name

        try:
            stage = MetricsCollectionStage(output_path=output_path)

            result = ImageAnalysisResult(
                original_path="/test.jpg",
                photo_id="1",
                display_path="/test.jpg",
                aesthetic_score=5.0,
            )
            item = PipelineItem(result=result)
            stage.process([item])
            stage.cleanup()

            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_cleanup_skips_report_when_no_photos(self):
        """Test that cleanup skips report when no photos processed"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        os.remove(output_path)  # Remove so we can check it's not created

        try:
            stage = MetricsCollectionStage(output_path=output_path)
            stage.cleanup()  # No photos processed

            assert not os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)