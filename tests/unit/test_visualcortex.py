"""
Extended unit tests for VisualCortex class to increase coverage.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import supervision as sv

from vision_detect_segment.core.visualcortex import VisualCortex
from vision_detect_segment.utils.config import create_test_config

# import copy


class TestVisualCortexInitialization:
    """Tests for VisualCortex initialization."""

    @pytest.mark.parametrize("model_id", ["owlv2", "yolo-world", "grounding_dino"])
    def test_initialization_different_models(self, model_id):
        """Test initialization with different detection models."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(
                    objdetect_model_id=model_id,
                    device="cpu",
                    verbose=False,
                    config=config,
                )

                assert cortex._objdetect_model_id == model_id

    def test_initialization_with_custom_stream_name(self):
        """Test initialization with custom stream name."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    stream_name="custom_stream",
                    verbose=False,
                    config=config,
                )

                assert cortex._stream_name == "custom_stream"

    def test_initialization_redis_failure(self):
        """Test initialization when Redis fails."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer") as mock_redis:
                mock_redis.side_effect = Exception("Redis unavailable")

                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    verbose=False,
                    config=config,
                )

                # Should handle gracefully
                assert cortex._streamer is None

    def test_initialization_with_verbose(self):
        """Test initialization with verbose logging."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    verbose=True,
                    config=config,
                )

                assert cortex.verbose is True


class TestVisualCortexDetection:
    """Tests for detection methods."""

    @pytest.fixture
    def mock_cortex(self):
        """Create VisualCortex with mocked components."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector") as mock_detector:
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer") as mock_streamer:
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    verbose=False,
                    config=config,
                )

                cortex._object_detector = mock_detector()
                cortex._streamer = mock_streamer()

                yield cortex

    def test_detect_objects_from_redis_success(self, mock_cortex):
        """Test successful detection from Redis."""
        # Mock image from Redis
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        metadata = {"robot": "test_robot", "frame_id": 1}

        mock_cortex._streamer.get_latest_image = Mock(return_value=(image, metadata))

        # Mock detection
        mock_cortex._object_detector.detect_objects = Mock(
            return_value=[
                {
                    "label": "test_object",
                    "confidence": 0.9,
                    "bbox": {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 200},
                }
            ]
        )

        result = mock_cortex.detect_objects_from_redis()

        assert result is True
        assert len(mock_cortex._detected_objects) > 0

    def test_detect_objects_from_redis_no_image(self, mock_cortex):
        """Test detection when no image available."""
        mock_cortex._streamer.get_latest_image = Mock(return_value=None)

        result = mock_cortex.detect_objects_from_redis()

        assert result is False

    def test_detect_objects_from_redis_no_streamer(self):
        """Test detection when streamer unavailable."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            cortex = VisualCortex(
                objdetect_model_id="owlv2",
                device="cpu",
                verbose=False,
                config=config,
            )

            cortex._streamer = None

            result = cortex.detect_objects_from_redis()

            assert result is False


class TestVisualCortexImageProcessing:
    """Tests for image processing callback."""

    @pytest.fixture
    def cortex_with_detector(self):
        """Create cortex with working detector."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    verbose=False,
                    config=config,
                )

                # Mock detector
                mock_det_instance = Mock()
                mock_det_instance.detect_objects = Mock(return_value=[])
                mock_det_instance.get_detections = Mock(return_value=None)
                cortex._object_detector = mock_det_instance

                yield cortex

    def test_process_image_callback_success(self, cortex_with_detector):
        """Test successful image processing."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        metadata = {"robot": "test_robot", "workspace_id": "ws1"}

        cortex_with_detector.process_image_callback(image, metadata, None)

        assert cortex_with_detector._img_work is not None
        assert cortex_with_detector._processed_frames == 1

    def test_process_image_callback_invalid_image(self, cortex_with_detector):
        """Test processing with invalid image."""
        metadata = {}

        # Should handle error gracefully
        cortex_with_detector.process_image_callback(None, metadata, None)

        # Image should not be stored
        assert cortex_with_detector._img_work is None

    def test_process_image_callback_with_detections(self, cortex_with_detector):
        """Test processing with successful detections."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        metadata = {"frame_id": 5}

        # Mock detections
        cortex_with_detector._object_detector.detect_objects = Mock(
            return_value=[
                {
                    "label": "object1",
                    "confidence": 0.9,
                    "bbox": {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 200},
                }
            ]
        )

        cortex_with_detector.process_image_callback(image, metadata, None)

        assert len(cortex_with_detector._detected_objects) == 1


class TestVisualCortexAnnotation:
    """Tests for annotation creation."""

    @pytest.fixture
    def cortex_for_annotation(self):
        """Create cortex for annotation tests."""
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
                mock_detector.get_detections = Mock(return_value=None)
                mock_detector.get_label_texts = Mock(return_value=None)
                cortex._object_detector = mock_detector

                yield cortex

    def test_create_annotated_frame_no_image(self, cortex_for_annotation):
        """Test annotation with no image."""
        cortex_for_annotation._img_work = None

        cortex_for_annotation._create_annotated_frame([])

        assert cortex_for_annotation._annotated_frame is None

    def test_create_annotated_frame_no_detections(self, cortex_for_annotation):
        """Test annotation with no detections."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cortex_for_annotation._img_work = image

        cortex_for_annotation._create_annotated_frame([])

        # Should create resized frame even without detections
        assert cortex_for_annotation._annotated_frame is not None

    def test_create_annotated_frame_with_detections(self, cortex_for_annotation):
        """Test annotation with detections."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cortex_for_annotation._img_work = image

        # Mock detection data
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )

        detected_objects = [
            {
                "label": "test",
                "confidence": 0.9,
                "bbox": {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 200},
            }
        ]

        cortex_for_annotation._object_detector.get_detections = Mock(return_value=detections)
        cortex_for_annotation._object_detector.get_label_texts = Mock(return_value=np.array(["test (0.90)"]))

        cortex_for_annotation._create_annotated_frame(detected_objects)

        assert cortex_for_annotation._annotated_frame is not None

    def test_scale_detections(self, cortex_for_annotation):
        """Test scaling detection coordinates."""
        detections = sv.Detections(
            xyxy=np.array([[10.0, 20.0, 100.0, 200.0]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )

        scaled = cortex_for_annotation._scale_detections(detections, 2.0, 1.5)

        # Check scaling - new detection object should have scaled coordinates
        assert scaled.xyxy[0][0] == 20.0  # x_min scaled by 2.0
        assert scaled.xyxy[0][1] == 30.0  # y_min scaled by 1.5
        assert scaled.xyxy[0][2] == 200.0  # x_max scaled by 2.0
        assert scaled.xyxy[0][3] == 300.0  # y_max scaled by 1.5


class TestVisualCortexGetters:
    """Tests for getter methods."""

    @pytest.fixture
    def cortex_with_data(self):
        """Create cortex with data."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    verbose=False,
                    config=config,
                )

                # Setup data
                cortex._img_work = np.zeros((100, 100, 3), dtype=np.uint8)
                cortex._annotated_frame = np.zeros((200, 200, 3), dtype=np.uint8)
                cortex._detected_objects = [
                    {"label": "obj1", "confidence": 0.9},
                    {"label": "obj2", "confidence": 0.8},
                ]
                cortex._processed_frames = 5

                yield cortex

    def test_get_current_image(self, cortex_with_data):
        """Test getting current image."""
        image = cortex_with_data.get_current_image(resize=False)

        assert image is not None
        assert image.shape == (100, 100, 3)

    def test_get_current_image_with_resize(self, cortex_with_data):
        """Test getting current image with resize."""
        # Small image should be resized
        cortex_with_data._img_work = np.zeros((50, 50, 3), dtype=np.uint8)

        image = cortex_with_data.get_current_image(resize=True)

        assert image is not None
        assert image.shape[0] > 50  # Should be resized

    def test_get_annotated_image(self, cortex_with_data):
        """Test getting annotated image."""
        image = cortex_with_data.get_annotated_image()

        assert image is not None
        assert image.shape == (200, 200, 3)

    def test_get_detected_objects(self, cortex_with_data):
        """Test getting detected objects."""
        objects = cortex_with_data.get_detected_objects()

        assert len(objects) == 2
        assert objects[0]["label"] == "obj1"

        # Test that it's a deep copy - modifying should not affect original
        objects[0]["label"] = "modified"
        assert cortex_with_data._detected_objects[0]["label"] == "obj1"

    def test_get_object_labels(self, cortex_with_data):
        """Test getting object labels."""
        mock_detector = Mock()
        mock_detector.get_object_labels = Mock(return_value=[["label1", "label2"]])
        cortex_with_data._object_detector = mock_detector

        labels = cortex_with_data.get_object_labels()

        assert len(labels) == 1
        assert "label1" in labels[0]

    def test_get_processed_frames_count(self, cortex_with_data):
        """Test getting processed frames count."""
        count = cortex_with_data.get_processed_frames_count()

        assert count == 5

    def test_get_device(self, cortex_with_data):
        """Test getting device."""
        device = cortex_with_data.get_device()

        assert device in ["cpu", "cuda"]


class TestVisualCortexLabelManagement:
    """Tests for label management."""

    @pytest.fixture
    def cortex_for_labels(self):
        """Create cortex for label tests."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    verbose=False,
                    config=config,
                )

                # Mock detector
                mock_detector = Mock()
                mock_detector.add_label = Mock()
                cortex._object_detector = mock_detector

                yield cortex

    def test_add_detectable_object(self, cortex_for_labels):
        """Test adding new detectable object."""
        cortex_for_labels.add_detectable_object("new_object")

        assert cortex_for_labels._object_detector.add_label.called

    def test_add_detectable_object_verbose(self, cortex_for_labels):
        """Test adding object with verbose logging."""
        cortex_for_labels.verbose = True

        cortex_for_labels.add_detectable_object("new_object")

        assert cortex_for_labels._object_detector.add_label.called


class TestVisualCortexUtilities:
    """Tests for utility methods."""

    @pytest.fixture
    def cortex(self):
        """Create basic cortex."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    verbose=False,
                    config=config,
                )
                yield cortex

    def test_clear_cache(self, cortex):
        """Test clearing GPU cache."""
        with patch("vision_detect_segment.core.visualcortex.clear_gpu_cache") as mock_clear:
            cortex.clear_cache()

            assert mock_clear.called

    def test_get_memory_usage(self, cortex):
        """Test getting memory usage."""
        with patch("vision_detect_segment.utils.utils.get_memory_usage") as mock_mem:
            mock_mem.return_value = {"rss_mb": 100.0, "vms_mb": 200.0}

            memory = cortex.get_memory_usage()

            assert "rss_mb" in memory or memory == {}

    def test_get_stats(self, cortex):
        """Test getting processing statistics."""
        cortex._processed_frames = 10
        cortex._img_work = np.zeros((100, 100, 3), dtype=np.uint8)
        cortex._detected_objects = [{"label": "obj1"}]

        stats = cortex.get_stats()

        assert stats["processed_frames"] == 10
        assert stats["has_current_image"] is True
        assert stats["current_detections_count"] == 1


class TestVisualCortexProperties:
    """Tests for property compatibility."""

    @pytest.fixture
    def cortex_with_properties(self):
        """Create cortex for property tests."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    verbose=False,
                    config=config,
                )

                cortex._img_work = np.zeros((100, 100, 3), dtype=np.uint8)
                cortex._annotated_frame = np.zeros((200, 200, 3), dtype=np.uint8)
                cortex._detected_objects = [{"label": "obj"}]
                cortex._processed_frames = 3

                yield cortex

    def test_current_image_property(self, cortex_with_properties):
        """Test current_image property."""
        image = cortex_with_properties.current_image

        assert image is not None

    def test_annotated_image_property(self, cortex_with_properties):
        """Test annotated_image property."""
        image = cortex_with_properties.annotated_image

        assert image is not None

    def test_detected_objects_property(self, cortex_with_properties):
        """Test detected_objects property."""
        objects = cortex_with_properties.detected_objects

        assert len(objects) == 1

    def test_object_labels_property(self, cortex_with_properties):
        """Test object_labels property."""
        mock_detector = Mock()
        mock_detector.get_object_labels = Mock(return_value=[["label1"]])
        cortex_with_properties._object_detector = mock_detector

        labels = cortex_with_properties.object_labels

        assert len(labels) > 0

    def test_processed_frames_property(self, cortex_with_properties):
        """Test processed_frames property."""
        frames = cortex_with_properties.processed_frames

        assert frames == 3


class TestVisualCortexErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def cortex_error_test(self):
        """Create cortex for error testing."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    verbose=True,
                    config=config,
                )

                mock_detector = Mock()
                cortex._object_detector = mock_detector

                yield cortex

    def test_detection_error_handling(self, cortex_error_test):
        """Test handling of detection errors."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        metadata = {}

        # Mock detection failure
        cortex_error_test._object_detector.detect_objects = Mock(side_effect=Exception("Detection failed"))

        # Should handle gracefully
        cortex_error_test.process_image_callback(image, metadata, None)

    def test_annotation_error_handling(self, cortex_error_test):
        """Test handling of annotation errors."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cortex_error_test._img_work = image

        # Mock annotation failure
        cortex_error_test._corner_annotator = Mock()
        cortex_error_test._corner_annotator.annotate = Mock(side_effect=Exception("Annotation failed"))

        detections = sv.Detections(
            xyxy=np.array([[10, 20, 50, 60]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )

        cortex_error_test._object_detector.get_detections = Mock(return_value=detections)

        # Should handle gracefully
        cortex_error_test._create_annotated_frame([{"label": "test"}])


class TestVisualCortexDeprecatedMethods:
    """Tests for deprecated method compatibility."""

    @pytest.fixture
    def cortex_deprecated(self):
        """Create cortex for deprecated methods."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    verbose=False,
                    config=config,
                )

                cortex._img_work = np.zeros((100, 100, 3), dtype=np.uint8)
                cortex._annotated_frame = np.zeros((200, 200, 3), dtype=np.uint8)

                yield cortex

    def test_img_work_method(self, cortex_deprecated):
        """Test deprecated img_work() method."""
        image = cortex_deprecated.img_work()

        assert image is not None

    def test_annotated_frame_method(self, cortex_deprecated):
        """Test deprecated annotated_frame() method."""
        image = cortex_deprecated.annotated_frame()

        assert image is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
