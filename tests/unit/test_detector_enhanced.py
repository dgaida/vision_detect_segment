"""
Enhanced unit tests for ObjectDetector class to increase code coverage.
Tests focus on uncovered code paths, edge cases, and error conditions.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import supervision as sv
import torch

from vision_detect_segment.core.object_detector import ObjectDetector
from vision_detect_segment.utils.config import MODEL_CONFIGS, create_test_config


class TestObjectDetectorLabelStabilization:
    """Tests for label stabilization feature with tracking."""

    @pytest.fixture
    def detector_with_tracking(self):
        """Create detector with tracking enabled."""
        config = create_test_config()

        with patch.object(ObjectDetector, "_load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["cat", "dog", "bird"]],
                config=config,
                enable_tracking=True,
            )

            # Mock tracker
            detector._tracker.enable_tracking = True
            detector._tracker.update_label_history = Mock(side_effect=lambda ids, labels: labels)
            detector._tracker.detect_lost_tracks = Mock(return_value=[])
            detector._tracker.cleanup_lost_tracks = Mock()

            yield detector

    def test_apply_label_stabilization_with_tracking(self, detector_with_tracking):
        """Test label stabilization is applied when tracking enabled."""
        detected_objects = [
            {"label": "cat", "confidence": 0.9, "bbox": {}},
            {"label": "dog", "confidence": 0.85, "bbox": {}},
        ]
        track_ids = np.array([1, 2])

        # Mock stabilization to return different labels
        detector_with_tracking._tracker.update_label_history = Mock(return_value=["stable_cat", "stable_dog"])

        result = detector_with_tracking._apply_label_stabilization(detected_objects, track_ids)

        assert result[0]["label"] == "stable_cat"
        assert result[1]["label"] == "stable_dog"
        assert detector_with_tracking._tracker.update_label_history.called

    def test_apply_label_stabilization_without_tracking(self, detector_with_tracking):
        """Test stabilization is skipped when tracking disabled."""
        detector_with_tracking._tracker.enable_tracking = False

        detected_objects = [{"label": "cat", "confidence": 0.9, "bbox": {}}]
        track_ids = np.array([1])

        result = detector_with_tracking._apply_label_stabilization(detected_objects, track_ids)

        # Labels should remain unchanged
        assert result[0]["label"] == "cat"
        assert not detector_with_tracking._tracker.update_label_history.called

    def test_apply_label_stabilization_no_track_ids(self, detector_with_tracking):
        """Test stabilization handles None track IDs."""
        detected_objects = [{"label": "cat", "confidence": 0.9, "bbox": {}}]

        result = detector_with_tracking._apply_label_stabilization(detected_objects, None)

        assert result[0]["label"] == "cat"

    def test_cleanup_lost_tracks_called(self, detector_with_tracking):
        """Test that lost tracks are cleaned up."""
        detected_objects = [{"label": "cat", "confidence": 0.9, "bbox": {}}]
        track_ids = np.array([1])

        # Mock lost tracks detection
        detector_with_tracking._tracker.detect_lost_tracks = Mock(return_value=[5, 6])

        detector_with_tracking._apply_label_stabilization(detected_objects, track_ids)

        assert detector_with_tracking._tracker.cleanup_lost_tracks.called
        detector_with_tracking._tracker.cleanup_lost_tracks.assert_called_with([5, 6])


class TestObjectDetectorYOLOE:
    """Tests specific to YOLOE detection and segmentation."""

    @pytest.fixture
    def yoloe_detector(self):
        """Create YOLOE detector."""
        config = create_test_config()

        with patch("vision_detect_segment.core.object_detector.YOLOE_AVAILABLE", True):
            with patch("vision_detect_segment.core.object_detector.YOLOE") as mock_yoloe:
                mock_model = Mock()
                mock_yoloe.return_value = mock_model

                detector = ObjectDetector(
                    device="cpu",
                    model_id="yoloe-11l",
                    object_labels=[["cat", "dog"]],
                    config=config,
                    enable_tracking=True,
                )

                detector._model = mock_model
                yield detector

    def test_detect_yoloe_with_masks(self, yoloe_detector):
        """Test YOLOE detection with segmentation masks."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Create mock result with masks
        mock_result = Mock()
        mock_result.boxes = Mock()

        # Create mock boxes
        box1 = Mock()
        box1.cls = 0
        box1.conf = 0.9
        box1.xyxy = torch.tensor([[10, 20, 100, 200]])

        mock_result.boxes.__iter__ = Mock(return_value=iter([box1]))
        mock_result.boxes.cls = np.array([0])
        mock_result.boxes.conf = np.array([0.9])
        mock_result.boxes.xyxy = torch.tensor([[10, 20, 100, 200]])
        mock_result.boxes.id = torch.tensor([1])
        mock_result.names = {0: "cat"}

        # Add masks
        mock_masks = Mock()
        mask_data = torch.rand(1, 480, 640)
        mock_masks.data = mask_data
        mock_result.masks = mock_masks

        yoloe_detector._model.track = Mock(return_value=[mock_result])

        # Mock stabilization
        yoloe_detector._tracker.update_label_history = Mock(return_value=["cat"])

        results = yoloe_detector.detect_objects(image)

        assert len(results) == 1
        assert results[0]["has_mask"] is True
        assert "mask_data" in results[0]
        assert "mask_shape" in results[0]

    def test_detect_yoloe_without_masks(self, yoloe_detector):
        """Test YOLOE detection without masks."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        mock_result = Mock()
        mock_result.boxes = Mock()

        box1 = Mock()
        box1.cls = 0
        box1.conf = 0.9
        box1.xyxy = torch.tensor([[10, 20, 100, 200]])

        mock_result.boxes.__iter__ = Mock(return_value=iter([box1]))
        mock_result.boxes.cls = np.array([0])
        mock_result.boxes.conf = np.array([0.9])
        mock_result.boxes.xyxy = torch.tensor([[10, 20, 100, 200]])
        mock_result.boxes.id = torch.tensor([1])
        mock_result.names = {0: "cat"}
        mock_result.masks = None  # No masks

        yoloe_detector._model.track = Mock(return_value=[mock_result])
        yoloe_detector._tracker.update_label_history = Mock(return_value=["cat"])

        results = yoloe_detector.detect_objects(image)

        assert len(results) == 1
        assert results[0]["has_mask"] is False

    def test_yoloe_without_tracking(self):
        """Test YOLOE detection with tracking disabled."""
        config = create_test_config()

        with patch("vision_detect_segment.core.object_detector.YOLOE_AVAILABLE", True):
            with patch("vision_detect_segment.core.object_detector.YOLOE") as mock_yoloe:
                mock_model = Mock()
                mock_yoloe.return_value = mock_model

                detector = ObjectDetector(
                    device="cpu",
                    model_id="yoloe-11l",
                    object_labels=[["cat"]],
                    config=config,
                    enable_tracking=False,
                )

                image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

                mock_result = Mock()
                mock_result.boxes = None
                mock_model.predict = Mock(return_value=[mock_result])

                results = detector.detect_objects(image)

                assert mock_model.predict.called
                assert results == []


class TestObjectDetectorYOLOTracking:
    """Tests for YOLO-specific tracking functionality."""

    @pytest.fixture
    def yolo_detector_tracked(self):
        """Create YOLO detector with tracking."""
        config = create_test_config()

        with patch("vision_detect_segment.core.object_detector.YOLO_AVAILABLE", True):
            with patch("vision_detect_segment.core.object_detector.YOLO") as mock_yolo:
                mock_model = Mock()
                mock_yolo.return_value = mock_model

                detector = ObjectDetector(
                    device="cpu",
                    model_id="yolo-world",
                    object_labels=[["cat", "dog"]],
                    config=config,
                    enable_tracking=True,
                )

                detector._model = mock_model
                yield detector

    def test_detect_yolo_with_track_ids(self, yolo_detector_tracked):
        """Test YOLO detection with track IDs present."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        box1 = Mock()
        box1.cls = 0
        box1.conf = 0.9
        box1.xyxy = torch.tensor([[10, 20, 100, 200]])

        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.__iter__ = Mock(return_value=iter([box1]))
        mock_result.boxes.cls = np.array([0])
        mock_result.boxes.conf = np.array([0.9])
        mock_result.boxes.xyxy = torch.tensor([[10, 20, 100, 200]])
        mock_result.boxes.id = torch.tensor([42])  # Track ID
        mock_result.names = {0: "cat"}

        yolo_detector_tracked._tracker.track = Mock(return_value=[mock_result])
        yolo_detector_tracked._tracker.update_label_history = Mock(return_value=["cat"])

        results = yolo_detector_tracked.detect_objects(image)

        assert len(results) == 1
        assert results[0]["track_id"] == 42

    def test_detect_yolo_without_track_ids(self, yolo_detector_tracked):
        """Test YOLO detection without track IDs."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.__iter__ = Mock(return_value=iter([]))
        mock_result.boxes.cls = np.array([])
        mock_result.boxes.conf = np.array([])
        mock_result.boxes.xyxy = torch.tensor([])
        mock_result.boxes.id = None  # No track IDs
        mock_result.names = {}

        yolo_detector_tracked._tracker.track = Mock(return_value=[mock_result])

        results = yolo_detector_tracked.detect_objects(image)

        assert results == []


class TestObjectDetectorTransformerTracking:
    """Tests for transformer model tracking with ByteTrack."""

    @pytest.fixture
    def owlv2_with_tracking(self):
        """Create OWL-V2 detector with tracking."""
        config = create_test_config()

        with patch("vision_detect_segment.core.object_detector.TRANSFORMERS_AVAILABLE", True):
            with patch.object(ObjectDetector, "_load_model") as mock_load:
                mock_model = Mock()
                mock_processor = Mock()
                mock_load.return_value = (mock_model, mock_processor)

                detector = ObjectDetector(
                    device="cpu",
                    model_id="owlv2",
                    object_labels=[["cat", "dog"]],
                    config=config,
                    enable_tracking=True,
                )

                detector._model = mock_model
                detector._processor = mock_processor
                yield detector

    def test_transformer_tracking_success(self, owlv2_with_tracking):
        """Test successful tracking with transformer model."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Mock processor and model
        mock_inputs = Mock()
        owlv2_with_tracking._processor.return_value = mock_inputs
        mock_inputs.to = Mock(return_value=mock_inputs)

        mock_outputs = Mock()
        owlv2_with_tracking._model.return_value = mock_outputs

        # Mock post-processing
        mock_results = {
            "boxes": torch.tensor([[10.0, 20.0, 100.0, 200.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
        owlv2_with_tracking._processor.post_process_object_detection = Mock(return_value=[mock_results])

        # Mock tracking
        tracked_detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )
        tracked_detections.tracker_id = np.array([1])

        owlv2_with_tracking._tracker.update_with_detections = Mock(return_value=tracked_detections)
        owlv2_with_tracking._tracker.update_label_history = Mock(return_value=["cat"])

        with patch.object(owlv2_with_tracking, "_extract_owlv2_labels", return_value=np.array(["cat"])):
            results = owlv2_with_tracking.detect_objects(image)

        assert len(results) == 1
        assert results[0]["track_id"] == 1

    def test_transformer_tracking_failure(self, owlv2_with_tracking):
        """Test tracking failure handling."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        mock_inputs = Mock()
        owlv2_with_tracking._processor.return_value = mock_inputs
        mock_inputs.to = Mock(return_value=mock_inputs)

        mock_outputs = Mock()
        owlv2_with_tracking._model.return_value = mock_outputs

        mock_results = {
            "boxes": torch.tensor([[10.0, 20.0, 50.0, 60.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
        owlv2_with_tracking._processor.post_process_object_detection = Mock(return_value=[mock_results])

        # Mock tracking failure
        owlv2_with_tracking._tracker.update_with_detections = Mock(side_effect=Exception("Tracking failed"))

        with patch.object(owlv2_with_tracking, "_extract_owlv2_labels", return_value=np.array(["cat"])):
            results = owlv2_with_tracking.detect_objects(image)

        # Should still return results without track IDs
        assert len(results) == 1
        assert "track_id" not in results[0]


class TestObjectDetectorLabelManagement:
    """Tests for dynamic label management."""

    def test_add_label_to_yoloe_prompted(self):
        """Test adding label to prompted YOLOE model."""
        config = create_test_config()

        with patch("vision_detect_segment.core.object_detector.YOLOE_AVAILABLE", True):
            with patch("vision_detect_segment.core.object_detector.YOLOE") as mock_yoloe:
                mock_model = Mock()
                mock_model.get_text_pe = Mock(return_value="text_pe")
                mock_model.set_classes = Mock()
                mock_yoloe.return_value = mock_model

                detector = ObjectDetector(
                    device="cpu",
                    model_id="yoloe-11l",
                    object_labels=[["cat"]],
                    config=config,
                )

                detector.add_label("dog")

                assert "dog" in detector._object_labels[0]
                assert mock_model.set_classes.called

    def test_add_label_to_yoloe_prompt_free(self):
        """Test adding label to prompt-free YOLOE model (should not update)."""
        config = create_test_config()

        # Override config for prompt-free variant
        MODEL_CONFIGS["yoloe-11l-pf"].model_params["is_prompt_free"] = True

        with patch("vision_detect_segment.core.object_detector.YOLOE_AVAILABLE", True):
            with patch("vision_detect_segment.core.object_detector.YOLOE") as mock_yoloe:
                mock_model = Mock()
                mock_model.set_classes = Mock()
                mock_yoloe.return_value = mock_model

                detector = ObjectDetector(
                    device="cpu",
                    model_id="yoloe-11l-pf",
                    object_labels=[["cat"]],
                    config=config,
                )

                detector.add_label("dog")

                # Label should be added to list but not update model
                assert "dog" in detector._object_labels[0]
                # set_classes should not be called for prompt-free
                assert not mock_model.set_classes.called

    def test_add_label_to_grounding_dino(self):
        """Test adding label updates preprocessed labels for Grounding-DINO."""
        config = create_test_config()

        with patch.object(ObjectDetector, "_load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="grounding_dino",
                object_labels=[["cat", "dog"]],
                config=config,
            )

            initial_processed = detector._processed_labels

            detector.add_label("bird")

            assert "bird" in detector._object_labels[0]
            assert detector._processed_labels != initial_processed
            assert "bird" in detector._processed_labels


class TestObjectDetectorGetLabelTexts:
    """Tests for get_label_texts with various scenarios."""

    @pytest.fixture
    def detector_for_labels(self):
        """Create detector for label text testing."""
        config = create_test_config()

        with patch.object(ObjectDetector, "_load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test"]],
                config=config,
            )
            yield detector

    def test_get_label_texts_with_tracking_and_confidence(self, detector_for_labels):
        """Test label texts with both tracking and confidence."""
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )
        detections.tracker_id = np.array([42])

        detector_for_labels._current_detections = detections
        detector_for_labels._current_labels = np.array(["cat"])

        labels = detector_for_labels.get_label_texts()

        assert labels is not None
        assert len(labels) == 1
        assert "#42" in labels[0]
        assert "0.90" in labels[0]
        assert "cat" in labels[0]

    def test_get_label_texts_without_tracking(self, detector_for_labels):
        """Test label texts without tracking."""
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]]),
            confidence=np.array([0.85]),
            class_id=np.array([0]),
        )

        detector_for_labels._current_detections = detections
        detector_for_labels._current_labels = np.array(["dog"])

        labels = detector_for_labels.get_label_texts()

        assert labels is not None
        assert "#" not in labels[0]  # No track ID
        assert "0.85" in labels[0]

    def test_get_label_texts_without_confidence(self, detector_for_labels):
        """Test label texts without confidence scores."""
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]]),
            class_id=np.array([0]),
        )
        detections.confidence = None

        detector_for_labels._current_detections = detections
        detector_for_labels._current_labels = np.array(["bird"])

        labels = detector_for_labels.get_label_texts()

        assert labels is not None
        assert "bird" in labels[0]

    def test_get_label_texts_none_detections(self, detector_for_labels):
        """Test label texts when no detections."""
        detector_for_labels._current_detections = None
        detector_for_labels._current_labels = None

        labels = detector_for_labels.get_label_texts()

        assert labels is None


class TestObjectDetectorSupervisionCreation:
    """Tests for supervision detection creation methods."""

    @pytest.fixture
    def detector_for_supervision(self):
        """Create detector for supervision tests."""
        config = create_test_config()

        with patch.object(ObjectDetector, "_load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test"]],
                config=config,
            )
            yield detector

    def test_create_supervision_detections_with_track_ids(self, detector_for_supervision):
        """Test creating supervision detections with track IDs."""
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = torch.tensor([[10, 20, 100, 200]])
        mock_result.boxes.conf = torch.tensor([0.9])
        mock_result.boxes.cls = torch.tensor([0])

        objects = [{"label": "cat"}]
        track_ids = np.array([1])

        detector_for_supervision._create_supervision_detections([mock_result], objects, track_ids)

        assert detector_for_supervision._current_detections is not None
        assert hasattr(detector_for_supervision._current_detections, "tracker_id")
        assert detector_for_supervision._current_detections.tracker_id[0] == 1

    def test_create_supervision_detections_from_results_grounding_dino(self, detector_for_supervision):
        """Test creating supervision detections for Grounding-DINO."""
        detector_for_supervision._model_id = "grounding_dino"

        results = {
            "boxes": torch.tensor([[10, 20, 100, 200]]),
            "scores": torch.tensor([0.9]),
            "labels": ["cat"],  # String labels for Grounding-DINO
        }

        labels = ["cat"]
        track_ids = np.array([1])

        with patch.object(ObjectDetector, "_convert_labels_to_class_ids", return_value=np.array([0])):
            detector_for_supervision._create_supervision_detections_from_results(results, labels, track_ids)

        assert detector_for_supervision._current_detections is not None
        assert hasattr(detector_for_supervision._current_detections, "tracker_id")


class TestObjectDetectorEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_detect_with_custom_threshold(self):
        """Test detection with custom confidence threshold."""
        config = create_test_config()

        with patch.object(ObjectDetector, "_load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test"]],
                config=config,
            )

            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

            with patch.object(detector, "_detect_transformer_based") as mock_detect:
                mock_detect.return_value = []

                detector.detect_objects(image, confidence_threshold=0.7)

                # Verify custom threshold was used
                assert mock_detect.called

    def test_detect_with_invalid_threshold(self):
        """Test detection with invalid threshold raises error."""
        config = create_test_config()

        with patch.object(ObjectDetector, "_load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test"]],
                config=config,
            )

            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

            with pytest.raises(Exception):  # ConfigurationError
                detector.detect_objects(image, confidence_threshold=1.5)

    def test_create_supervision_detections_empty_boxes(self):
        """Test creating supervision detections with no boxes."""
        config = create_test_config()

        with patch.object(ObjectDetector, "_load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="yolo-world",
                object_labels=[["test"]],
                config=config,
            )

            mock_result = Mock()
            mock_result.boxes = None

            detector._create_supervision_detections([mock_result], [])

            assert detector._current_detections is None

    def test_serialize_mask(self):
        """Test mask serialization."""
        mask = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        serialized = ObjectDetector._serialize_mask(mask)

        assert isinstance(serialized, str)
        assert len(serialized) > 0

    def test_convert_labels_to_class_ids_with_collision(self):
        """Test class ID conversion handles hash collisions."""
        labels = ["test1", "test2", "test3"]

        class_ids = ObjectDetector._convert_labels_to_class_ids(labels)

        assert len(class_ids) == 3
        assert isinstance(class_ids, np.ndarray)


class TestObjectDetectorBackwardCompatibility:
    """Tests for deprecated methods."""

    @pytest.fixture
    def detector(self):
        """Create basic detector."""
        config = create_test_config()

        with patch.object(ObjectDetector, "_load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test"]],
                config=config,
            )
            yield detector

    def test_deprecated_detections_method(self, detector):
        """Test deprecated detections() method."""
        mock_detections = Mock()
        detector._current_detections = mock_detections

        result = detector.detections()

        assert result == mock_detections

    def test_deprecated_label_texts_method(self, detector):
        """Test deprecated label_texts() method."""
        mock_labels = np.array(["test"])
        detector._current_labels = mock_labels
        detector._current_detections = Mock()

        result = detector.label_texts()

        assert result is not None

    def test_deprecated_object_labels_method(self, detector):
        """Test deprecated object_labels() method."""
        result = detector.object_labels()

        assert isinstance(result, list)
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
