"""
Extended unit tests for ObjectDetector class to increase coverage.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import supervision as sv
import torch

from vision_detect_segment.core.object_detector import ObjectDetector
from vision_detect_segment.utils.config import create_test_config
from vision_detect_segment.utils.exceptions import (
    ModelLoadError,
)


class TestObjectDetectorInitialization:
    """Tests for ObjectDetector initialization."""

    @pytest.mark.parametrize("model_id", ["owlv2", "yolo-world", "grounding_dino"])
    def test_initialization_different_models(self, model_id):
        """Test initializing detector with different model types."""
        config = create_test_config()
        config.model.name = model_id

        # Mock model loading to avoid actual downloads
        with patch.object(ObjectDetector, "_load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id=model_id,
                object_labels=[["test_object"]],
                verbose=False,
                config=config,
                enable_tracking=False,
            )

            assert detector._model_id == model_id
            assert detector._device in ["cpu", "cuda"]

    def test_initialization_with_gpu(self):
        """Test initialization with GPU preference."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = create_test_config()
        with patch.object(ObjectDetector, "_load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cuda",
                model_id="owlv2",
                object_labels=[["test"]],
                config=config,
            )

            assert detector._device == "cuda"

    def test_initialization_with_tracking_enabled(self):
        """Test initialization with tracking enabled."""
        config = create_test_config()
        with patch.object(ObjectDetector, "_load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test"]],
                config=config,
                enable_tracking=True,
            )

            assert detector._tracker is not None

    def test_invalid_model_raises_error(self):
        """Test that invalid model ID raises error."""
        with pytest.raises((ModelLoadError, ValueError)):
            ObjectDetector(
                device="cpu",
                model_id="invalid_model_xyz",
                object_labels=[["test"]],
                verbose=False,
            )

    def test_redis_initialization_failure_handling(self):
        """Test graceful handling of Redis initialization failure."""
        config = create_test_config()

        with patch.object(ObjectDetector, "_load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            with patch("vision_detect_segment.core.object_detector.RedisMessageBroker") as mock_redis:
                mock_redis.side_effect = Exception("Redis connection failed")

                # Should not raise, but log warning
                detector = ObjectDetector(
                    device="cpu",
                    model_id="owlv2",
                    object_labels=[["test"]],
                    verbose=True,
                    config=config,
                )

                assert detector._redis_broker is None


class TestObjectDetectorLabels:
    """Tests for label management in ObjectDetector."""

    @pytest.fixture
    def mock_detector(self):
        """Create a mock detector with patched model loading."""
        config = create_test_config()
        with patch.object(ObjectDetector, "_load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["object1", "object2"]],
                config=config,
            )
            yield detector

    def test_add_label(self, mock_detector):
        """Test adding new labels."""
        initial_count = len(mock_detector._object_labels[0])
        mock_detector.add_label("new_object")

        assert len(mock_detector._object_labels[0]) == initial_count + 1
        assert "new_object" in mock_detector._object_labels[0]

    def test_get_object_labels(self, mock_detector):
        """Test retrieving object labels."""
        labels = mock_detector.get_object_labels()
        assert isinstance(labels, list)
        assert len(labels) > 0
        assert isinstance(labels[0], list)

    def test_preprocess_labels_grounding_dino(self):
        """Test label preprocessing for Grounding-DINO."""
        labels = [["object1", "object2", "object3"]]
        processed = ObjectDetector._preprocess_labels(labels, "grounding_dino")

        assert isinstance(processed, str)
        assert "object1" in processed
        assert "." in processed  # Should have periods

    def test_preprocess_labels_other_models(self):
        """Test label preprocessing for other models."""
        labels = [["object1", "object2"]]
        processed = ObjectDetector._preprocess_labels(labels, "owlv2")

        assert processed == labels  # Should return unchanged


class TestObjectDetectorDetection:
    """Tests for detection methods."""

    @pytest.fixture
    def mock_detector_with_model(self):
        """Create detector with mocked model."""
        config = create_test_config()

        mock_model = Mock()
        mock_processor = Mock()

        with patch.object(ObjectDetector, "_load_model") as mock_load:
            mock_load.return_value = (mock_model, mock_processor)

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test_object"]],
                config=config,
            )
            detector._model = mock_model
            detector._processor = mock_processor
            yield detector

    def test_detect_objects_basic(self, mock_detector_with_model):
        """Test basic object detection."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Mock detection results
        with patch.object(mock_detector_with_model, "_detect_transformer_based") as mock_detect:
            mock_detect.return_value = [
                {
                    "label": "test_object",
                    "confidence": 0.95,
                    "bbox": {"x_min": 100, "y_min": 100, "x_max": 200, "y_max": 200},
                    "has_mask": False,
                }
            ]

            results = mock_detector_with_model.detect_objects(image)

            assert len(results) > 0
            assert results[0]["label"] == "test_object"
            assert 0.0 <= results[0]["confidence"] <= 1.0

    def test_detect_objects_with_custom_threshold(self, mock_detector_with_model):
        """Test detection with custom confidence threshold."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        with patch.object(mock_detector_with_model, "_detect_transformer_based") as mock_detect:
            mock_detect.return_value = []

            mock_detector_with_model.detect_objects(image, confidence_threshold=0.5)

            # Verify threshold was passed
            assert mock_detect.called

    def test_detect_objects_empty_result(self, mock_detector_with_model):
        """Test detection returning no objects."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        with patch.object(mock_detector_with_model, "_detect_transformer_based") as mock_detect:
            mock_detect.return_value = []

            results = mock_detector_with_model.detect_objects(image)

            assert results == []

    def test_detect_objects_error_handling(self, mock_detector_with_model):
        """Test error handling during detection."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        with patch.object(mock_detector_with_model, "_detect_transformer_based") as mock_detect:
            mock_detect.side_effect = Exception("Detection failed")

            # Should return empty list, not raise
            results = mock_detector_with_model.detect_objects(image)
            assert results == []


class TestObjectDetectorTransformerDetection:
    """Tests for transformer-based detection methods."""

    @pytest.fixture
    def detector_owlv2(self):
        """Create OWL-V2 detector with mocked components."""
        config = create_test_config()

        mock_model = Mock()
        mock_processor = Mock()

        with patch.object(ObjectDetector, "_load_model") as mock_load:
            mock_load.return_value = (mock_model, mock_processor)

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test_object"]],
                config=config,
            )
            yield detector

    def test_create_object_dicts(self):
        """Test creating object dictionaries from results."""
        results = {
            "boxes": torch.tensor([[10, 20, 100, 200], [30, 40, 150, 250]]),
            "scores": torch.tensor([0.9, 0.8]),
        }
        labels = ["object1", "object2"]

        objects = ObjectDetector._create_object_dicts(results, labels)

        assert len(objects) == 2
        assert objects[0]["label"] == "object1"
        assert (objects[0]["confidence"] > 0.89) & (objects[0]["confidence"] < 0.91)
        assert "bbox" in objects[0]

    def test_serialize_mask(self):
        """Test mask serialization."""
        mask = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        serialized = ObjectDetector._serialize_mask(mask)

        assert isinstance(serialized, str)
        assert len(serialized) > 0

    def test_convert_labels_to_class_ids(self):
        """Test converting labels to class IDs."""
        labels = ["label1", "label2", "label3"]

        class_ids = ObjectDetector._convert_labels_to_class_ids(labels)

        assert len(class_ids) == 3
        assert isinstance(class_ids, np.ndarray)


class TestObjectDetectorSegmentation:
    """Tests for segmentation functionality."""

    @pytest.fixture
    def detector_with_segmenter(self):
        """Create detector with mocked segmenter."""
        config = create_test_config()

        with patch.object(ObjectDetector, "_load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test"]],
                config=config,
            )

            # Mock segmenter
            detector._segmenter._segmenter = Mock()
            detector._segmenter._model_id = "fastsam"

            yield detector

    def test_add_segmentation_success(self, detector_with_segmenter):
        """Test adding segmentation masks to detections."""
        objects = [
            {
                "label": "test",
                "confidence": 0.9,
                "bbox": {"x_min": 10, "y_min": 10, "x_max": 100, "y_max": 100},
                "has_mask": False,
            }
        ]

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = torch.tensor([[10, 10, 100, 100]])

        # Mock segmentation
        mock_mask = np.random.randint(0, 255, (90, 90), dtype=np.uint8)
        detector_with_segmenter._segmenter.segment_box_in_image = Mock(return_value=(mock_mask, mock_mask > 0))

        result = detector_with_segmenter._add_segmentation(objects, image, boxes)

        assert result[0]["has_mask"] is True
        assert "mask_data" in result[0]

    def test_add_segmentation_no_segmenter(self):
        """Test adding segmentation when segmenter unavailable."""
        config = create_test_config()

        with patch.object(ObjectDetector, "_load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test"]],
                config=config,
            )

            detector._segmenter._segmenter = None

            objects = [{"label": "test", "confidence": 0.9, "has_mask": False}]
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            boxes = torch.tensor([[10, 10, 50, 50]])

            result = detector._add_segmentation(objects, image, boxes)

            # Should return unchanged
            assert result == objects


class TestObjectDetectorPublishing:
    """Tests for Redis publishing functionality."""

    @pytest.fixture
    def detector_with_redis(self):
        """Create detector with mocked Redis."""
        config = create_test_config()

        with patch.object(ObjectDetector, "_load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test"]],
                config=config,
            )

            # Mock Redis broker
            detector._redis_broker = Mock()

            yield detector

    def test_publish_detections_success(self, detector_with_redis):
        """Test successful detection publishing."""
        objects = [
            {
                "label": "test",
                "confidence": 0.9,
                "bbox": {"x_min": 10, "y_min": 10, "x_max": 100, "y_max": 100},
            }
        ]

        detector_with_redis._publish_detections(objects, "owlv2")

        assert detector_with_redis._redis_broker.publish_objects.called

    def test_publish_detections_empty(self, detector_with_redis):
        """Test publishing empty detections."""
        detector_with_redis._publish_detections([], "owlv2")

        # Should not call publish for empty detections
        assert not detector_with_redis._redis_broker.publish_objects.called

    def test_publish_detections_no_broker(self):
        """Test publishing when Redis broker unavailable."""
        config = create_test_config()

        with patch.object(ObjectDetector, "_load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test"]],
                config=config,
            )

            detector._redis_broker = None

            # Should not raise error
            detector._publish_detections([{"test": "data"}], "owlv2")


class TestObjectDetectorSupervisionIntegration:
    """Tests for supervision library integration."""

    @pytest.fixture
    def detector(self):
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

    def test_create_supervision_detections_from_results(self, detector):
        """Test creating supervision detections from results."""
        results = {
            "boxes": torch.tensor([[10, 20, 100, 200]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
        labels = ["test_object"]

        detector._create_supervision_detections_from_results(results, labels)

        assert detector._current_detections is not None
        assert len(detector._current_detections.xyxy) == 1

    def test_get_detections(self, detector):
        """Test getting supervision detections."""
        # Create mock detection
        detector._current_detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )

        detections = detector.get_detections()

        assert detections is not None
        assert len(detections.xyxy) == 1

    def test_get_label_texts(self, detector):
        """Test getting label texts."""
        detector._current_detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )
        detector._current_labels = np.array(["test_object"])

        labels = detector.get_label_texts()

        assert labels is not None
        assert len(labels) == 1
        assert "test_object" in labels[0]

    def test_get_label_texts_with_tracking(self, detector):
        """Test getting label texts with track IDs."""
        detector._current_detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )
        detector._current_detections.tracker_id = np.array([1])
        detector._current_labels = np.array(["test_object"])

        labels = detector.get_label_texts()

        assert labels is not None
        assert "#1" in labels[0]  # Should include track ID


class TestObjectDetectorModelLoading:
    """Tests for model loading methods."""

    def test_load_yolo_model(self):
        """Test YOLO model loading."""
        create_test_config()

        with patch("vision_detect_segment.core.object_detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model

            with patch.object(ObjectDetector, "__init__", lambda x, **kwargs: None):
                detector = ObjectDetector.__new__(ObjectDetector)
                detector._object_labels = [["test"]]
                detector._model_id = "yolo-world"

                model, processor = detector._load_yolo_model()

                assert model is not None
                assert processor is None  # YOLO doesn't use processor

    def test_validate_model_availability_missing_dependency(self):
        """Test model availability validation with missing dependency."""
        config = create_test_config()

        with patch("vision_detect_segment.core.object_detector.YOLO_AVAILABLE", False):
            with pytest.raises(ModelLoadError):
                ObjectDetector(
                    device="cpu",
                    model_id="yolo-world",
                    object_labels=[["test"]],
                    config=config,
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
