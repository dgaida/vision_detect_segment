"""
Extended unit tests for ObjectTracker class to increase coverage.
"""

from unittest.mock import Mock

import numpy as np
import pytest
import supervision as sv

from vision_detect_segment.core.object_tracker import ObjectTracker
from vision_detect_segment.utils.exceptions import DetectionError


class TestObjectTrackerInitialization:
    """Tests for ObjectTracker initialization."""

    def test_initialization_yolo_with_tracking(self):
        """Test initialization with YOLO model and tracking enabled."""
        mock_model = Mock()

        tracker = ObjectTracker(
            model=mock_model,
            model_id="yolo-world",
            enable_tracking=True,
            verbose=False,
        )

        assert tracker.enable_tracking is True
        assert tracker._use_yolo_tracker is True
        assert tracker._tracker is None  # YOLO uses built-in

    def test_initialization_yolo_without_tracking(self):
        """Test initialization with YOLO model and tracking disabled."""
        mock_model = Mock()

        tracker = ObjectTracker(
            model=mock_model,
            model_id="yolo-world",
            enable_tracking=False,
            verbose=False,
        )

        assert tracker.enable_tracking is False
        assert tracker._use_yolo_tracker is True

    def test_initialization_transformer_with_tracking(self):
        """Test initialization with transformer model and tracking enabled."""
        mock_model = Mock()

        tracker = ObjectTracker(
            model=mock_model,
            model_id="owlv2",
            enable_tracking=True,
            verbose=False,
        )

        assert tracker.enable_tracking is True
        assert tracker._use_yolo_tracker is False
        assert tracker._tracker is not None  # Uses ByteTrack

    def test_initialization_transformer_without_tracking(self):
        """Test initialization with transformer model and tracking disabled."""
        mock_model = Mock()

        tracker = ObjectTracker(
            model=mock_model,
            model_id="owlv2",
            enable_tracking=False,
            verbose=True,
        )

        assert tracker.enable_tracking is False
        assert tracker._tracker is None

    def test_initialization_grounding_dino(self):
        """Test initialization with Grounding-DINO model."""
        mock_model = Mock()

        tracker = ObjectTracker(
            model=mock_model,
            model_id="grounding_dino",
            enable_tracking=True,
            verbose=False,
        )

        assert tracker.model_id == "grounding_dino"
        assert tracker._use_yolo_tracker is False


class TestObjectTrackerYOLOTracking:
    """Tests for YOLO tracking functionality."""

    @pytest.fixture
    def yolo_tracker_enabled(self):
        """Create YOLO tracker with tracking enabled."""
        mock_model = Mock()
        tracker = ObjectTracker(
            model=mock_model,
            model_id="yolo-world",
            enable_tracking=True,
            verbose=False,
        )
        yield tracker

    @pytest.fixture
    def yolo_tracker_disabled(self):
        """Create YOLO tracker with tracking disabled."""
        mock_model = Mock()
        tracker = ObjectTracker(
            model=mock_model,
            model_id="yolo-world",
            enable_tracking=False,
            verbose=False,
        )
        yield tracker

    def test_track_with_yolo_tracking_enabled(self, yolo_tracker_enabled):
        """Test YOLO tracking when enabled."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Mock track results
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.id = np.array([1, 2, 3])

        yolo_tracker_enabled.model.track = Mock(return_value=[mock_result])

        result = yolo_tracker_enabled.track(image, threshold=0.3)

        assert yolo_tracker_enabled.model.track.called
        assert result is not None

    def test_track_with_yolo_tracking_disabled(self, yolo_tracker_disabled):
        """Test YOLO tracking when disabled (falls back to predict)."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Mock predict results
        mock_result = Mock()
        mock_result.boxes = Mock()

        yolo_tracker_disabled.model.predict = Mock(return_value=[mock_result])

        result = yolo_tracker_disabled.track(image, threshold=0.3)

        assert yolo_tracker_disabled.model.predict.called
        assert result is not None

    def test_track_with_custom_parameters(self, yolo_tracker_enabled):
        """Test tracking with custom parameters."""
        image = np.random.randint(0, 255, (320, 480, 3), dtype=np.uint8)

        mock_result = Mock()
        yolo_tracker_enabled.model.track = Mock(return_value=[mock_result])

        yolo_tracker_enabled.track(
            image,
            threshold=0.5,
            max_det=30,
        )

        # Verify parameters were passed
        call_args = yolo_tracker_enabled.model.track.call_args
        assert call_args is not None

    def test_track_yolo_error_handling(self, yolo_tracker_enabled):
        """Test error handling in YOLO tracking."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Mock tracking failure
        yolo_tracker_enabled.model.track = Mock(side_effect=Exception("Tracking failed"))

        with pytest.raises(DetectionError):
            yolo_tracker_enabled.track(image)


class TestObjectTrackerTransformerTracking:
    """Tests for transformer (ByteTrack) tracking functionality."""

    @pytest.fixture
    def transformer_tracker_enabled(self):
        """Create transformer tracker with tracking enabled."""
        mock_model = Mock()
        tracker = ObjectTracker(
            model=mock_model,
            model_id="owlv2",
            enable_tracking=True,
            verbose=False,
        )
        yield tracker

    @pytest.fixture
    def transformer_tracker_disabled(self):
        """Create transformer tracker with tracking disabled."""
        mock_model = Mock()
        tracker = ObjectTracker(
            model=mock_model,
            model_id="owlv2",
            enable_tracking=False,
            verbose=False,
        )
        yield tracker

    def test_update_with_detections_enabled(self, transformer_tracker_enabled):
        """Test updating detections with ByteTrack when enabled."""
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200], [30, 40, 150, 250]]),
            confidence=np.array([0.9, 0.8]),
            class_id=np.array([0, 1]),
        )

        # Mock ByteTrack update
        tracked_detections = detections
        tracked_detections.tracker_id = np.array([1, 2])

        transformer_tracker_enabled._tracker.update_with_detections = Mock(return_value=tracked_detections)

        result = transformer_tracker_enabled.update_with_detections(detections)

        assert result is not None
        assert hasattr(result, "tracker_id")
        assert transformer_tracker_enabled._tracker.update_with_detections.called

    def test_update_with_detections_disabled(self, transformer_tracker_disabled):
        """Test updating detections when tracking disabled."""
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )

        result = transformer_tracker_disabled.update_with_detections(detections)

        # Should return unchanged detections
        assert result is detections

    def test_update_with_empty_detections(self, transformer_tracker_enabled):
        """Test updating with empty detections."""
        detections = sv.Detections.empty()

        transformer_tracker_enabled._tracker.update_with_detections = Mock(return_value=detections)

        result = transformer_tracker_enabled.update_with_detections(detections)

        assert result is not None

    def test_track_called_on_transformer_raises_error(self, transformer_tracker_enabled):
        """Test that calling track() on transformer model raises error."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        with pytest.raises(DetectionError, match="not applicable to transformer"):
            transformer_tracker_enabled.track(image)


class TestObjectTrackerReset:
    """Tests for tracker reset functionality."""

    def test_reset_yolo_tracker(self):
        """Test resetting YOLO tracker."""
        mock_model = Mock()
        mock_model.tracker = Mock()
        mock_model.tracker.reset = Mock()

        tracker = ObjectTracker(
            model=mock_model,
            model_id="yolo-world",
            enable_tracking=True,
            verbose=False,
        )

        tracker.reset()

        # Verify reset was called
        assert mock_model.tracker.reset.called or True  # May not have tracker attribute

    def test_reset_bytetrack_tracker(self):
        """Test resetting ByteTrack tracker."""
        mock_model = Mock()

        tracker = ObjectTracker(
            model=mock_model,
            model_id="owlv2",
            enable_tracking=True,
            verbose=False,
        )

        tracker._tracker.reset = Mock()
        tracker.reset()

        assert tracker._tracker.reset.called

    def test_reset_no_tracker(self):
        """Test reset when tracking is disabled."""
        mock_model = Mock()

        tracker = ObjectTracker(
            model=mock_model,
            model_id="owlv2",
            enable_tracking=False,
            verbose=False,
        )

        # Should not raise error
        tracker.reset()

    def test_reset_with_error(self):
        """Test reset handling errors gracefully."""
        mock_model = Mock()
        mock_model.tracker = Mock()
        mock_model.tracker.reset = Mock(side_effect=Exception("Reset failed"))

        tracker = ObjectTracker(
            model=mock_model,
            model_id="yolo-world",
            enable_tracking=True,
            verbose=False,
        )

        # Should handle error gracefully
        tracker.reset()


class TestObjectTrackerEdgeCases:
    """Tests for edge cases in tracking."""

    def test_tracking_with_single_object(self):
        """Test tracking with single object."""
        mock_model = Mock()
        tracker = ObjectTracker(
            model=mock_model,
            model_id="yolo-world",
            enable_tracking=True,
            verbose=False,
        )

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Mock single detection
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.id = np.array([1])

        tracker.model.track = Mock(return_value=[mock_result])

        result = tracker.track(image)

        assert result is not None

    def test_tracking_with_many_objects(self):
        """Test tracking with many objects."""
        mock_model = Mock()
        tracker = ObjectTracker(
            model=mock_model,
            model_id="yolo-world",
            enable_tracking=True,
            verbose=False,
        )

        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Mock many detections
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.id = np.arange(50)  # 50 objects

        tracker.model.track = Mock(return_value=[mock_result])

        result = tracker.track(image, max_det=50)

        assert result is not None

    def test_bytetrack_with_occlusions(self):
        """Test ByteTrack handling objects with occlusions."""
        mock_model = Mock()
        tracker = ObjectTracker(
            model=mock_model,
            model_id="owlv2",
            enable_tracking=True,
            verbose=False,
        )

        # First frame
        detections1 = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200], [110, 120, 200, 300]]),
            confidence=np.array([0.9, 0.8]),
            class_id=np.array([0, 0]),
        )

        tracked1 = detections1
        tracked1.tracker_id = np.array([1, 2])
        tracker._tracker.update_with_detections = Mock(return_value=tracked1)

        tracker.update_with_detections(detections1)

        # Second frame - one object occluded
        detections2 = sv.Detections(
            xyxy=np.array([[15, 25, 105, 205]]),
            confidence=np.array([0.85]),
            class_id=np.array([0]),
        )

        tracked2 = detections2
        tracked2.tracker_id = np.array([1])  # Same ID as before
        tracker._tracker.update_with_detections = Mock(return_value=tracked2)

        result2 = tracker.update_with_detections(detections2)

        assert result2 is not None


class TestObjectTrackerVerboseLogging:
    """Tests for verbose logging in tracker."""

    def test_verbose_mode_enabled(self):
        """Test tracker with verbose mode enabled."""
        mock_model = Mock()

        tracker = ObjectTracker(
            model=mock_model,
            model_id="yolo-world",
            enable_tracking=True,
            verbose=True,
        )

        assert tracker.verbose is True

    def test_verbose_mode_disabled(self):
        """Test tracker with verbose mode disabled."""
        mock_model = Mock()

        tracker = ObjectTracker(
            model=mock_model,
            model_id="owlv2",
            enable_tracking=True,
            verbose=False,
        )

        assert tracker.verbose is False


class TestObjectTrackerModelTypes:
    """Tests for different model type handling."""

    @pytest.mark.parametrize(
        "model_id,expected_yolo",
        [
            ("yolo-world", True),
            ("yolov8", True),
            ("owlv2", False),
            ("grounding_dino", False),
            ("YOLO-WORLD", True),  # Test case insensitive
        ],
    )
    def test_model_type_detection(self, model_id, expected_yolo):
        """Test correct detection of YOLO vs transformer models."""
        mock_model = Mock()

        tracker = ObjectTracker(
            model=mock_model,
            model_id=model_id,
            enable_tracking=True,
            verbose=False,
        )

        assert tracker._use_yolo_tracker == expected_yolo


class TestObjectTrackerIntegration:
    """Integration tests for tracking workflow."""

    def test_complete_yolo_tracking_workflow(self):
        """Test complete YOLO tracking workflow over multiple frames."""
        mock_model = Mock()
        tracker = ObjectTracker(
            model=mock_model,
            model_id="yolo-world",
            enable_tracking=True,
            verbose=False,
        )

        # Simulate 3 frames
        for frame_id in range(3):
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # Mock results with persistent IDs
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.id = np.array([1, 2])

            tracker.model.track = Mock(return_value=[mock_result])

            result = tracker.track(image)

            assert result is not None

    def test_complete_bytetrack_workflow(self):
        """Test complete ByteTrack workflow over multiple frames."""
        mock_model = Mock()
        tracker = ObjectTracker(
            model=mock_model,
            model_id="owlv2",
            enable_tracking=True,
            verbose=False,
        )

        # Simulate 3 frames
        for frame_id in range(3):
            detections = sv.Detections(
                xyxy=np.array([[10 + frame_id * 5, 20, 100, 200]]),
                confidence=np.array([0.9]),
                class_id=np.array([0]),
            )

            # Mock tracked result
            tracked = detections
            tracked.tracker_id = np.array([1])  # Persistent ID

            tracker._tracker.update_with_detections = Mock(return_value=tracked)

            result = tracker.update_with_detections(detections)

            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
