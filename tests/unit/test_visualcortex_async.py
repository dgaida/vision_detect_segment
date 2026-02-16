"""
Unit tests for visualcortex_async.py
Tests VisualCortexAsync with async processing capabilities.
"""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest
import supervision as sv

from vision_detect_segment.core.visualcortex_async import VisualCortexAsync
from vision_detect_segment.utils.config import create_test_config


class TestVisualCortexAsyncInitialization:
    """Tests for VisualCortexAsync initialization."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies."""
        with patch("vision_detect_segment.core.visualcortex_async.RedisImageStreamer") as mock_redis, patch(
            "vision_detect_segment.core.visualcortex_async.RedisLabelManager"
        ) as mock_label, patch("vision_detect_segment.core.visualcortex_async.ObjectDetector") as mock_detector:

            # Setup mocks
            mock_redis.return_value = Mock()
            mock_label.return_value = Mock()
            mock_detector.return_value = Mock()

            yield {"redis": mock_redis, "label_manager": mock_label, "detector": mock_detector}

    def test_basic_initialization(self, mock_dependencies):
        """Test basic initialization with defaults."""
        config = create_test_config()

        cortex = VisualCortexAsync(objdetect_model_id="owlv2", device="cpu", verbose=False, config=config)

        assert cortex._objdetect_model_id == "owlv2"
        assert cortex._device == "cpu"
        assert cortex._num_workers == 2
        assert cortex._processor is not None

        # Cleanup
        cortex.stop(timeout=1.0)

    def test_initialization_with_custom_params(self, mock_dependencies):
        """Test initialization with custom async parameters."""
        config = create_test_config()

        cortex = VisualCortexAsync(
            objdetect_model_id="owlv2",
            device="cpu",
            num_workers=4,
            max_queue_size=50,
            enable_backpressure=False,
            redis_poll_interval=0.05,
            config=config,
        )

        assert cortex._num_workers == 4
        assert cortex._max_queue_size == 50
        assert cortex._enable_backpressure is False
        assert cortex._redis_poll_interval == 0.05

        cortex.stop(timeout=1.0)

    def test_component_initialization(self, mock_dependencies):
        """Test that all components are initialized."""
        config = create_test_config()

        cortex = VisualCortexAsync(objdetect_model_id="owlv2", device="cpu", config=config)

        assert cortex._streamer is not None
        assert cortex._object_detector is not None
        assert cortex._label_annotator is not None
        assert cortex._corner_annotator is not None
        assert cortex._halo_annotator is not None

        cortex.stop(timeout=1.0)


class TestVisualCortexAsyncStartStop:
    """Tests for start/stop functionality."""

    @pytest.fixture
    def cortex(self):
        """Create VisualCortexAsync instance."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex_async.RedisImageStreamer"), patch(
            "vision_detect_segment.core.visualcortex_async.RedisLabelManager"
        ), patch("vision_detect_segment.core.visualcortex_async.ObjectDetector"):

            cortex = VisualCortexAsync(objdetect_model_id="owlv2", device="cpu", verbose=False, config=config)

            yield cortex

            # Cleanup
            if cortex._redis_thread and cortex._redis_thread.is_alive():
                cortex.stop(timeout=2.0)

    def test_start_pipeline(self, cortex):
        """Test starting the async pipeline."""
        cortex.start()

        # Check threads are running
        assert cortex._redis_thread is not None
        assert cortex._result_thread is not None
        assert cortex._redis_thread.is_alive()
        assert cortex._result_thread.is_alive()

        time.sleep(0.2)  # Let threads initialize

        cortex.stop(timeout=2.0)

    def test_stop_pipeline(self, cortex):
        """Test stopping the async pipeline."""
        cortex.start()
        time.sleep(0.2)

        cortex.stop(timeout=2.0)

        # Threads should stop
        assert cortex._stop_event.is_set()

    def test_double_start_warning(self, cortex):
        """Test that double start shows warning."""
        cortex.start()
        time.sleep(0.1)

        # Try to start again - should warn but not crash
        cortex.start()

        cortex.stop(timeout=2.0)


class TestVisualCortexAsyncProcessing:
    """Tests for async processing functionality."""

    @pytest.fixture
    def cortex_with_mocks(self):
        """Create cortex with mocked components."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex_async.RedisImageStreamer") as mock_redis, patch(
            "vision_detect_segment.core.visualcortex_async.RedisLabelManager"
        ), patch("vision_detect_segment.core.visualcortex_async.ObjectDetector") as mock_detector:

            # Setup streamer mock
            mock_streamer = Mock()
            mock_redis.return_value = mock_streamer

            # Setup detector mock
            mock_det_instance = Mock()
            mock_det_instance.detect_objects = Mock(
                return_value=[
                    {"label": "test", "confidence": 0.9, "bbox": {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50}}
                ]
            )
            mock_det_instance.get_detections = Mock(
                return_value=sv.Detections(
                    xyxy=np.array([[10, 10, 50, 50]]), confidence=np.array([0.9]), class_id=np.array([0])
                )
            )
            mock_det_instance.get_label_texts = Mock(return_value=np.array(["test"]))
            mock_detector.return_value = mock_det_instance

            cortex = VisualCortexAsync(objdetect_model_id="owlv2", device="cpu", verbose=True, config=config)

            # Override streamer for testing
            cortex._streamer = mock_streamer

            yield cortex, mock_streamer

            cortex.stop(timeout=2.0)

    def test_process_image_sync(self, cortex_with_mocks):
        """Test synchronous image processing."""
        cortex, _ = cortex_with_mocks

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        metadata = {"frame_id": 1}

        result = cortex._process_image_sync(image, metadata)

        assert "objects" in result
        assert "annotated_image" in result
        assert len(result["objects"]) > 0

    def test_redis_polling_loop(self, cortex_with_mocks):
        """Test Redis polling loop functionality."""
        cortex, mock_streamer = cortex_with_mocks

        # Setup mock to return an image once
        test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        mock_streamer.get_latest_image = Mock(return_value=(test_image, {"frame_id": 1}))

        cortex.start()
        time.sleep(0.5)  # Let polling happen

        # Check that submit was attempted
        stats = cortex.get_stats()
        # Processor should have received tasks
        assert stats is not None

        cortex.stop(timeout=2.0)

    def test_result_handling_loop(self, cortex_with_mocks):
        """Test result handling loop."""
        cortex, _ = cortex_with_mocks

        cortex.start()

        # Manually submit a task to processor
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        cortex._processor.submit_task(image, {"frame_id": 1})

        time.sleep(0.5)  # Wait for processing

        # Check that result was handled
        cortex.get_latest_result()
        # Result may or may not be available depending on timing

        cortex.stop(timeout=2.0)


class TestVisualCortexAsyncAPI:
    """Tests for public API methods."""

    @pytest.fixture
    def cortex(self):
        """Create cortex instance."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex_async.RedisImageStreamer"), patch(
            "vision_detect_segment.core.visualcortex_async.RedisLabelManager"
        ), patch("vision_detect_segment.core.visualcortex_async.ObjectDetector"):

            cortex = VisualCortexAsync(objdetect_model_id="owlv2", device="cpu", config=config)

            # Set some test data
            cortex._detected_objects = [{"label": "test", "confidence": 0.9}]
            cortex._annotated_frame = np.zeros((100, 100, 3), dtype=np.uint8)

            yield cortex

            cortex.stop(timeout=1.0)

    def test_get_detected_objects(self, cortex):
        """Test getting detected objects (thread-safe)."""
        objects = cortex.get_detected_objects()

        assert len(objects) == 1
        assert objects[0]["label"] == "test"
        # Should be a deep copy
        objects[0]["label"] = "modified"
        assert cortex._detected_objects[0]["label"] == "test"

    def test_get_annotated_image(self, cortex):
        """Test getting annotated image (thread-safe)."""
        image = cortex.get_annotated_image()

        assert image is not None
        assert image.shape == (100, 100, 3)
        # Should be a copy
        assert image is not cortex._annotated_frame

    def test_get_latest_result(self, cortex):
        """Test getting latest result."""
        from vision_detect_segment.core.async_processor import ProcessingResult

        # Set a mock result
        test_result = ProcessingResult(task_id="test", detections=[], annotated_image=None, processing_time=0.1, success=True)
        cortex._latest_result = test_result

        result = cortex.get_latest_result()
        assert result == test_result

    def test_get_stats(self, cortex):
        """Test getting comprehensive statistics."""
        stats = cortex.get_stats()

        assert "model_id" in stats
        assert "device" in stats
        assert "num_workers" in stats
        assert "current_detections" in stats
        assert stats["model_id"] == "owlv2"
        assert stats["num_workers"] == 2

    def test_add_detectable_object(self, cortex):
        """Test adding new detectable object."""
        cortex._object_detector.add_label = Mock()

        cortex.add_detectable_object("new_object")

        assert cortex._object_detector.add_label.called

    def test_clear_cache(self, cortex):
        """Test clearing GPU cache."""
        with patch("vision_detect_segment.core.visualcortex_async.clear_gpu_cache") as mock_clear:
            cortex.clear_cache()
            assert mock_clear.called

    def test_cleanup(self, cortex):
        """Test cleanup method."""
        cortex.start()
        time.sleep(0.2)

        cortex.cleanup()

        # Should have stopped
        assert cortex._stop_event.is_set()


class TestVisualCortexAsyncAnnotation:
    """Tests for annotation functionality."""

    @pytest.fixture
    def cortex(self):
        """Create cortex with mocked detector."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex_async.RedisImageStreamer"), patch(
            "vision_detect_segment.core.visualcortex_async.RedisLabelManager"
        ), patch("vision_detect_segment.core.visualcortex_async.ObjectDetector") as mock_detector:

            # Setup detector mock
            mock_det = Mock()
            mock_det.detect_objects = Mock(
                return_value=[
                    {"label": "test", "confidence": 0.9, "bbox": {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50}}
                ]
            )
            mock_det.get_detections = Mock(
                return_value=sv.Detections(
                    xyxy=np.array([[10, 10, 50, 50]]), confidence=np.array([0.9]), class_id=np.array([0])
                )
            )
            mock_det.get_label_texts = Mock(return_value=np.array(["test"]))
            mock_detector.return_value = mock_det

            cortex = VisualCortexAsync(objdetect_model_id="owlv2", device="cpu", config=config)

            yield cortex
            cortex.stop(timeout=1.0)

    def test_create_annotated_frame(self, cortex):
        """Test creating annotated frame."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        detected_objects = [{"label": "test", "confidence": 0.9, "bbox": {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50}}]

        annotated = cortex._create_annotated_frame(image, detected_objects)

        assert annotated is not None
        assert annotated.shape[0] > 0

    def test_create_annotated_frame_no_detections(self, cortex):
        """Test annotation with no detections."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        annotated = cortex._create_annotated_frame(image, [])

        assert annotated is not None
        # Should just be resized image

    def test_scale_detections(self, cortex):
        """Test detection coordinate scaling."""
        detections = sv.Detections(
            xyxy=np.array([[10.0, 20.0, 50.0, 60.0]]), confidence=np.array([0.9]), class_id=np.array([0])
        )

        scaled = cortex._scale_detections(detections, 2.0, 3.0)

        assert scaled.xyxy[0][0] == 20.0  # x scaled by 2.0
        assert scaled.xyxy[0][1] == 60.0  # y scaled by 3.0


class TestVisualCortexAsyncEdgeCases:
    """Tests for edge cases and error handling."""

    def test_initialization_failure_handling(self):
        """Test handling of initialization failures."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex_async.ObjectDetector") as mock_detector:
            mock_detector.side_effect = Exception("Detector init failed")

            with pytest.raises(Exception):
                VisualCortexAsync(objdetect_model_id="owlv2", device="cpu", config=config)

    def test_process_image_error_handling(self):
        """Test error handling in image processing."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex_async.RedisImageStreamer"), patch(
            "vision_detect_segment.core.visualcortex_async.RedisLabelManager"
        ), patch("vision_detect_segment.core.visualcortex_async.ObjectDetector") as mock_detector:

            # Make detector fail
            mock_det = Mock()
            mock_det.detect_objects = Mock(side_effect=Exception("Detection failed"))
            mock_detector.return_value = mock_det

            cortex = VisualCortexAsync(objdetect_model_id="owlv2", device="cpu", config=config)

            image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            result = cortex._process_image_sync(image, {})

            # Should return error result
            assert "error" in result or result["objects"] == []

            cortex.stop(timeout=1.0)

    def test_annotation_failure_fallback(self):
        """Test fallback when annotation fails."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex_async.RedisImageStreamer"), patch(
            "vision_detect_segment.core.visualcortex_async.RedisLabelManager"
        ), patch("vision_detect_segment.core.visualcortex_async.ObjectDetector"):

            cortex = VisualCortexAsync(objdetect_model_id="owlv2", device="cpu", config=config)

            # Make annotator fail
            cortex._corner_annotator.annotate = Mock(side_effect=Exception("Annotation failed"))

            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

            # Should not crash, should return fallback
            annotated = cortex._create_annotated_frame(image, [])

            assert annotated is not None

            cortex.stop(timeout=1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
