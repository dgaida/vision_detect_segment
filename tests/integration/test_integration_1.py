"""
Integration tests for vision_detect_segment package.
Tests component interactions and end-to-end workflows.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import supervision as sv
import torch

from vision_detect_segment.core.object_detector import ObjectDetector
from vision_detect_segment.core.object_segmenter import ObjectSegmenter
from vision_detect_segment.core.visualcortex import VisualCortex
from vision_detect_segment.utils.config import (
    create_test_config,
)
from vision_detect_segment.utils.utils import create_test_image


class TestDetectorSegmenterIntegration:
    """Test integration between ObjectDetector and ObjectSegmenter."""

    @pytest.fixture
    def mock_detector_and_segmenter(self):
        """Create integrated detector and segmenter."""
        config = create_test_config()

        with patch("vision_detect_segment.core.object_detector.ObjectDetector._load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test_object", "another_object"]],
                config=config,
            )

            # Mock the model to return detections
            detector._model = Mock()
            detector._processor = Mock()

            # Mock segmenter
            segmenter = Mock(spec=ObjectSegmenter)
            segmenter.segment_box_in_image = Mock()
            detector._segmenter = segmenter

            yield detector, segmenter

    def test_detection_with_segmentation(self, mock_detector_and_segmenter):
        """Test detection pipeline with segmentation enabled."""
        detector, segmenter = mock_detector_and_segmenter

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Mock detection results
        mock_results = {
            "boxes": torch.tensor([[10.0, 20.0, 100.0, 200.0]]),
            "scores": torch.tensor([0.95]),
            "labels": torch.tensor([0]),
        }
        print(mock_results)

        with patch.object(detector, "_detect_transformer_based") as mock_detect:
            mock_detect.return_value = [
                {
                    "label": "test_object",
                    "confidence": 0.95,
                    "bbox": {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 200},
                    "has_mask": False,
                }
            ]

            # Mock segmentation
            mock_mask = np.random.randint(0, 255, (90, 90), dtype=np.uint8)
            segmenter.segment_box_in_image.return_value = (mock_mask, mock_mask > 0)

            results = detector.detect_objects(image)

            assert len(results) > 0
            assert results[0]["label"] == "test_object"

    def test_detection_without_segmentation(self, mock_detector_and_segmenter):
        """Test detection pipeline with segmentation disabled."""
        detector, segmenter = mock_detector_and_segmenter
        detector._segmenter._segmenter = None  # Disable segmentation

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        with patch.object(detector, "_detect_transformer_based") as mock_detect:
            mock_detect.return_value = [
                {
                    "label": "test_object",
                    "confidence": 0.95,
                    "bbox": {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 200},
                    "has_mask": False,
                }
            ]

            results = detector.detect_objects(image)

            assert len(results) > 0
            assert results[0]["has_mask"] is False


class TestDetectorTrackerIntegration:
    """Test integration between ObjectDetector and ObjectTracker."""

    @pytest.fixture
    def detector_with_tracking(self):
        """Create detector with tracking enabled."""
        config = create_test_config()

        with patch("vision_detect_segment.core.object_detector.ObjectDetector._load_model") as mock_load:
            mock_model = Mock()
            mock_load.return_value = (mock_model, Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["object1", "object2"]],
                config=config,
                enable_tracking=True,
            )

            yield detector

    def test_tracking_across_frames(self, detector_with_tracking):
        """Test object tracking across multiple frames."""
        detector = detector_with_tracking

        # Simulate 3 frames with consistent object
        for frame_num in range(3):
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # Mock detection with tracking
            with patch.object(detector, "_detect_transformer_based") as mock_detect:
                mock_detect.return_value = [
                    {
                        "label": "object1",
                        "confidence": 0.9,
                        "bbox": {"x_min": 10 + frame_num * 5, "y_min": 20, "x_max": 100, "y_max": 200},
                        "has_mask": False,
                    }
                ]

                # Mock tracker update
                if detector._tracker and detector._tracker._tracker:
                    mock_detection = sv.Detections(
                        xyxy=np.array([[10.0, 20.0, 100.0, 200.0]]),
                        confidence=np.array([0.9]),
                        class_id=np.array([0]),
                    )
                    mock_detection.tracker_id = np.array([1])
                    detector._tracker._tracker.update_with_detections = Mock(return_value=mock_detection)

                results = detector.detect_objects(image)

                assert len(results) > 0

    def test_track_id_persistence(self, detector_with_tracking):
        """Test that track IDs persist across frames."""
        detector = detector_with_tracking

        track_ids_per_frame = []

        for frame_num in range(5):
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            with patch.object(detector, "_detect_transformer_based") as mock_detect:
                # Create detection with track ID
                detection = {
                    "label": "object1",
                    "confidence": 0.9,
                    "bbox": {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 200},
                    "has_mask": False,
                    "track_id": 1,  # Consistent ID
                }
                mock_detect.return_value = [detection]

                results = detector.detect_objects(image)

                if results and "track_id" in results[0]:
                    track_ids_per_frame.append(results[0]["track_id"])

        # Verify track ID consistency
        if track_ids_per_frame:
            assert all(tid == track_ids_per_frame[0] for tid in track_ids_per_frame)


class TestVisualCortexIntegration:
    """Test VisualCortex integration with all components."""

    @pytest.fixture
    def mock_cortex(self):
        """Create VisualCortex with mocked components."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    verbose=False,
                    config=config,
                )

                # Setup mock detector
                mock_detector = Mock()
                mock_detector.detect_objects = Mock(return_value=[])
                mock_detector.get_detections = Mock(return_value=None)
                mock_detector.get_label_texts = Mock(return_value=None)
                mock_detector.get_object_labels = Mock(return_value=[["test"]])
                cortex._object_detector = mock_detector

                yield cortex

    def test_end_to_end_detection_workflow(self, mock_cortex):
        """Test complete detection workflow from image to annotations."""
        cortex = mock_cortex

        # Create test image
        image = create_test_image(shapes=["square", "circle"])
        metadata = {"robot": "test_robot", "frame_id": 1}

        # Mock detection results
        cortex._object_detector.detect_objects.return_value = [
            {
                "label": "blue square",
                "confidence": 0.95,
                "bbox": {"x_min": 50, "y_min": 50, "x_max": 130, "y_max": 130},
            }
        ]

        # Mock supervision detections
        mock_detections = sv.Detections(
            xyxy=np.array([[50, 50, 130, 130]]),
            confidence=np.array([0.95]),
            class_id=np.array([0]),
        )
        cortex._object_detector.get_detections.return_value = mock_detections
        cortex._object_detector.get_label_texts.return_value = np.array(["blue square (0.95)"])

        # Process image
        cortex.process_image_callback(image, metadata, None)

        # Verify processing
        assert cortex._img_work is not None
        assert cortex._processed_frames == 1
        assert len(cortex._detected_objects) > 0
        assert cortex._annotated_frame is not None

    def test_multiple_frame_processing(self, mock_cortex):
        """Test processing multiple frames sequentially."""
        cortex = mock_cortex

        num_frames = 5

        for frame_id in range(num_frames):
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            metadata = {"frame_id": frame_id}

            cortex._object_detector.detect_objects.return_value = [
                {
                    "label": f"object_{frame_id}",
                    "confidence": 0.9,
                    "bbox": {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 200},
                }
            ]

            cortex.process_image_callback(image, metadata, None)

        assert cortex._processed_frames == num_frames

    def test_detection_and_annotation_consistency(self, mock_cortex):
        """Test that annotations match detections."""
        cortex = mock_cortex

        image = create_test_image()
        metadata = {}

        # Setup detections
        detections_list = [
            {
                "label": "object1",
                "confidence": 0.9,
                "bbox": {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 200},
            },
            {
                "label": "object2",
                "confidence": 0.85,
                "bbox": {"x_min": 150, "y_min": 50, "x_max": 250, "y_max": 150},
            },
        ]

        cortex._object_detector.detect_objects.return_value = detections_list

        # Mock supervision detections
        mock_detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200], [150, 50, 250, 150]]),
            confidence=np.array([0.9, 0.85]),
            class_id=np.array([0, 1]),
        )
        cortex._object_detector.get_detections.return_value = mock_detections

        cortex.process_image_callback(image, metadata, None)

        # Verify detection count matches
        assert len(cortex._detected_objects) == len(detections_list)


class TestConfigurationIntegration:
    """Test configuration propagation through components."""

    def test_config_propagation_to_detector(self):
        """Test that configuration properly propagates to ObjectDetector."""
        config = create_test_config()
        config.model.confidence_threshold = 0.4

        with patch("vision_detect_segment.core.object_detector.ObjectDetector._load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test"]],
                config=config,
            )

            assert detector._config.model.confidence_threshold == 0.4

    def test_config_propagation_to_cortex(self):
        """Test that configuration properly propagates to VisualCortex."""
        config = create_test_config()
        config.annotation.resize_scale_factor = 3.0

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    config=config,
                )

                assert cortex._config.annotation.resize_scale_factor == 3.0

    def test_custom_labels_through_config(self):
        """Test custom object labels through configuration."""
        config = create_test_config()
        custom_labels = ["cat", "dog", "bird"]
        config.set_object_labels(custom_labels)

        with patch("vision_detect_segment.core.object_detector.ObjectDetector._load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=config.get_object_labels(),
                config=config,
            )

            labels = detector.get_object_labels()[0]
            assert all(label in labels for label in custom_labels)


class TestErrorHandlingIntegration:
    """Test error handling across component boundaries."""

    def test_detection_error_recovery(self):
        """Test system recovery from detection errors."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    config=config,
                )

                # Setup detector that fails once then succeeds
                mock_detector = Mock()
                call_count = 0

                def detect_side_effect(image):
                    nonlocal call_count
                    call_count += 1
                    if call_count == 1:
                        raise Exception("Detection failed")
                    return [{"label": "object", "confidence": 0.9, "bbox": {}}]

                mock_detector.detect_objects = Mock(side_effect=detect_side_effect)
                mock_detector.get_detections = Mock(return_value=None)
                cortex._object_detector = mock_detector

                # First call fails
                image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                cortex.process_image_callback(image, {}, None)

                # Second call succeeds
                cortex.process_image_callback(image, {}, None)

                assert call_count == 2

    def test_segmentation_error_fallback(self):
        """Test fallback when segmentation fails."""
        config = create_test_config()

        with patch("vision_detect_segment.core.object_detector.ObjectDetector._load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test"]],
                config=config,
            )

            # Make segmentation fail
            detector._segmenter.segment_box_in_image = Mock(side_effect=Exception("Segmentation failed"))

            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

            with patch.object(detector, "_detect_transformer_based") as mock_detect:
                mock_detect.return_value = [
                    {
                        "label": "test",
                        "confidence": 0.9,
                        "bbox": {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 200},
                        "has_mask": False,
                    }
                ]

                # Should not raise, should continue without segmentation
                results = detector.detect_objects(image)
                assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
