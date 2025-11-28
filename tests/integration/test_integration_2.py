"""
Integration tests for vision_detect_segment package - Part 2.
Tests Redis integration, label management, and advanced workflows.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
import supervision as sv
import time

from vision_detect_segment.core.visualcortex import VisualCortex
from vision_detect_segment.core.object_detector import ObjectDetector
from vision_detect_segment.core.object_tracker import ObjectTracker
from vision_detect_segment.utils.config import create_test_config
from vision_detect_segment.utils.utils import create_test_image


class TestRedisIntegration:
    """Test Redis integration across components."""

    @pytest.fixture
    def cortex_with_redis_mock(self):
        """Create VisualCortex with mocked Redis."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer") as mock_redis:
                mock_streamer = Mock()
                mock_redis.return_value = mock_streamer

                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    config=config,
                )

                cortex._streamer = mock_streamer

                # Setup mock detector
                mock_detector = Mock()
                mock_detector.detect_objects = Mock(return_value=[])
                mock_detector.get_detections = Mock(return_value=None)
                cortex._object_detector = mock_detector

                yield cortex, mock_streamer

    def test_redis_image_retrieval(self, cortex_with_redis_mock):
        """Test retrieving images from Redis."""
        cortex, mock_streamer = cortex_with_redis_mock

        # Mock Redis image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        metadata = {"robot": "test_robot", "timestamp": time.time()}

        mock_streamer.get_latest_image.return_value = (test_image, metadata)

        result = cortex.detect_objects_from_redis()

        assert result is True
        assert mock_streamer.get_latest_image.called

    def test_redis_unavailable_handling(self, cortex_with_redis_mock):
        """Test handling when Redis is unavailable."""
        cortex, mock_streamer = cortex_with_redis_mock

        # Simulate Redis returning no image
        mock_streamer.get_latest_image.return_value = None

        result = cortex.detect_objects_from_redis()

        assert result is False

    def test_redis_connection_failure(self):
        """Test handling Redis connection failure."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer") as mock_redis:
                mock_redis.side_effect = Exception("Redis connection failed")

                # Should not raise, should handle gracefully
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    config=config,
                )

                assert cortex._streamer is None

    def test_detection_publishing_to_redis(self):
        """Test publishing detections to Redis."""
        config = create_test_config()

        with patch("vision_detect_segment.core.object_detector.ObjectDetector._load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            with patch("vision_detect_segment.core.object_detector.RedisMessageBroker") as mock_broker_class:
                mock_broker = Mock()
                mock_broker_class.return_value = mock_broker

                detector = ObjectDetector(
                    device="cpu",
                    model_id="owlv2",
                    object_labels=[["test"]],
                    config=config,
                )

                detector._redis_broker = mock_broker

                # Publish detections
                detections = [
                    {"label": "test", "confidence": 0.9, "bbox": {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 200}}
                ]

                detector._publish_detections(detections, "owlv2")

                assert mock_broker.publish_objects.called


class TestLabelManagementIntegration:
    """Test label management across components."""

    def test_dynamic_label_addition(self):
        """Test adding labels dynamically during runtime."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    config=config,
                )

                # Setup mock detector
                mock_detector = Mock()
                mock_detector.add_label = Mock()
                mock_detector.get_object_labels = Mock(return_value=[["initial"]])
                cortex._object_detector = mock_detector

                # Add new label
                cortex.add_detectable_object("new_object")

                assert mock_detector.add_label.called
                assert mock_detector.add_label.call_args[0][0] == "new_object"

    def test_label_consistency_across_frames(self):
        """Test label consistency when processing multiple frames."""
        config = create_test_config()
        labels = ["object1", "object2", "object3"]
        config.set_object_labels(labels)

        with patch("vision_detect_segment.core.object_detector.ObjectDetector._load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=config.get_object_labels(),
                config=config,
            )

            initial_labels = detector.get_object_labels()[0]

            # Process multiple frames
            for _ in range(5):
                # Labels should remain consistent
                current_labels = detector.get_object_labels()[0]
                assert current_labels == initial_labels

    def test_label_stabilization_with_tracking(self):
        """Test label stabilization feature with tracking."""
        mock_model = Mock()

        tracker = ObjectTracker(
            model=mock_model,
            model_id="owlv2",
            enable_tracking=True,
            verbose=False,
            stabilization_frames=3,
        )

        # Simulate detections with varying labels
        track_ids = np.array([1, 1, 1, 1, 1])
        labels_per_frame = [
            ["cat", "dog", "cat", "cat", "cat"],  # Track 1: majority should be "cat"
        ]

        stabilized_results = []
        for labels in labels_per_frame:
            stabilized = tracker.update_label_history(track_ids[:1], [labels[0]])
            stabilized_results.append(stabilized[0])

        # After stabilization, label should be locked
        assert stabilized_results[-1] == "cat"


class TestYOLOIntegration:
    """Test YOLO-specific integration scenarios."""

    @pytest.fixture
    def yolo_detector(self):
        """Create YOLO detector."""
        config = create_test_config()

        with patch("vision_detect_segment.core.object_detector.YOLO_AVAILABLE", True):
            with patch("vision_detect_segment.core.object_detector.YOLO") as mock_yolo:
                mock_model = Mock()
                mock_yolo.return_value = mock_model

                detector = ObjectDetector(
                    device="cpu",
                    model_id="yolo-world",
                    object_labels=[["object1", "object2"]],
                    config=config,
                )

                detector._model = mock_model

                yield detector

    def test_yolo_detection_with_tracking(self, yolo_detector):
        """Test YOLO detection with built-in tracking."""
        detector = yolo_detector
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Mock YOLO results with tracking
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.cls = np.array([0, 1])
        mock_result.boxes.conf = np.array([0.9, 0.85])
        mock_result.boxes.xyxy = torch.tensor([[10, 20, 100, 200], [150, 50, 250, 150]])
        mock_result.boxes.id = np.array([1, 2])
        mock_result.names = {0: "object1", 1: "object2"}

        # Mock the tracker's track method
        detector._tracker.track = Mock(return_value=[mock_result])

        # Also need to mock the model's track method which is actually called
        detector._model.track = Mock(return_value=[mock_result])

        results = detector.detect_objects(image)

        assert len(results) == 2
        assert "track_id" in results[0]

    def test_yolo_label_update(self, yolo_detector):
        """Test updating YOLO model labels."""
        detector = yolo_detector

        initial_labels = detector.get_object_labels()[0]
        len_init = len(initial_labels)

        # Add new label
        detector.add_label("new_object")

        updated_labels = detector.get_object_labels()[0]
        assert "new_object" in updated_labels
        assert len(updated_labels) == len_init + 1


class TestTransformerModelIntegration:
    """Test transformer model-specific integration."""

    @pytest.fixture
    def owlv2_detector(self):
        """Create OWL-V2 detector."""
        config = create_test_config()

        with patch("vision_detect_segment.core.object_detector.TRANSFORMERS_AVAILABLE", True):
            with patch("vision_detect_segment.core.object_detector.Owlv2Processor") as mock_proc:
                with patch("vision_detect_segment.core.object_detector.Owlv2ForObjectDetection") as mock_model:
                    mock_processor = Mock()
                    mock_proc.from_pretrained.return_value = mock_processor

                    mock_detection_model = Mock()
                    mock_model.from_pretrained.return_value = mock_detection_model

                    detector = ObjectDetector(
                        device="cpu",
                        model_id="owlv2",
                        object_labels=[["cat", "dog"]],
                        config=config,
                    )

                    detector._model = mock_detection_model
                    detector._processor = mock_processor

                    yield detector

    def test_owlv2_detection_flow(self, owlv2_detector):
        """Test OWL-V2 detection workflow."""
        detector = owlv2_detector
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Mock processor output
        mock_inputs = Mock()
        detector._processor.return_value = mock_inputs
        mock_inputs.to = Mock(return_value=mock_inputs)

        # Mock model output
        mock_outputs = Mock()
        detector._model.return_value = mock_outputs

        # Mock post-processing - returns a list with one result dict
        mock_results = {
            "boxes": torch.tensor([[10.0, 20.0, 100.0, 200.0]]),
            "scores": torch.tensor([0.95]),
            "labels": torch.tensor([0]),
        }
        detector._processor.post_process_object_detection = Mock(return_value=[mock_results])

        # Mock label extraction
        with patch.object(detector, "_extract_owlv2_labels", return_value=np.array(["cat"])):
            results = detector.detect_objects(image)

            assert len(results) > 0
            assert results[0]["label"] == "cat"


class TestMemoryManagement:
    """Test memory management across components."""

    def test_gpu_cache_clearing(self):
        """Test GPU cache clearing after operations."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cuda",
                    config=config,
                )

                with patch("vision_detect_segment.core.visualcortex.clear_gpu_cache") as mock_clear:
                    cortex.clear_cache()
                    assert mock_clear.called

    def test_memory_tracking(self):
        """Test memory usage tracking."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    config=config,
                )

                # Get memory stats
                stats = cortex.get_stats()

                assert "device" in stats
                assert "processed_frames" in stats


class TestAnnotationIntegration:
    """Test annotation integration with detection pipeline."""

    @pytest.fixture
    def cortex_for_annotation(self):
        """Create VisualCortex for annotation testing."""
        config = create_test_config()
        config.annotation.show_labels = True
        config.annotation.show_confidence = True

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    config=config,
                )

                # Setup mock detector
                mock_detector = Mock()
                cortex._object_detector = mock_detector

                yield cortex

    def test_annotation_with_multiple_detections(self, cortex_for_annotation):
        """Test annotation with multiple objects."""
        cortex = cortex_for_annotation

        image = create_test_image(shapes=["square", "circle"])

        # Mock multiple detections
        detections_list = [
            {"label": "square", "confidence": 0.95, "bbox": {"x_min": 50, "y_min": 50, "x_max": 130, "y_max": 130}},
            {"label": "circle", "confidence": 0.90, "bbox": {"x_min": 200, "y_min": 50, "x_max": 280, "y_max": 130}},
        ]

        cortex._object_detector.detect_objects = Mock(return_value=detections_list)

        # Mock supervision detections
        mock_detections = sv.Detections(
            xyxy=np.array([[50, 50, 130, 130], [200, 50, 280, 130]]),
            confidence=np.array([0.95, 0.90]),
            class_id=np.array([0, 1]),
        )
        cortex._object_detector.get_detections = Mock(return_value=mock_detections)
        cortex._object_detector.get_label_texts = Mock(return_value=np.array(["square (0.95)", "circle (0.90)"]))

        cortex.process_image_callback(image, {}, None)

        # Verify annotations created
        assert cortex._annotated_frame is not None
        assert cortex._annotated_frame.shape[0] > 0

    def test_annotation_with_segmentation_masks(self, cortex_for_annotation):
        """Test annotation with segmentation masks."""
        cortex = cortex_for_annotation

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Create detection with mask
        detections_list = [
            {
                "label": "object",
                "confidence": 0.95,
                "bbox": {"x_min": 50, "y_min": 50, "x_max": 150, "y_max": 150},
                "has_mask": True,
            }
        ]

        cortex._object_detector.detect_objects = Mock(return_value=detections_list)

        # Mock supervision detections with mask
        mask = np.random.rand(480, 640) > 0.5
        mock_detections = sv.Detections(
            xyxy=np.array([[50, 50, 150, 150]]),
            confidence=np.array([0.95]),
            class_id=np.array([0]),
        )
        mock_detections.mask = [mask]

        cortex._object_detector.get_detections = Mock(return_value=mock_detections)
        cortex._object_detector.get_label_texts = Mock(return_value=np.array(["object (0.95)"]))

        cortex.process_image_callback(image, {}, None)

        assert cortex._annotated_frame is not None


class TestPerformanceIntegration:
    """Test performance-related integration scenarios."""

    def test_detection_timing(self):
        """Test detection timing across components."""
        config = create_test_config()

        with patch("vision_detect_segment.core.object_detector.ObjectDetector._load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test"]],
                config=config,
                enable_tracking=False,
            )

            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            with patch.object(detector, "_detect_transformer_based") as mock_detect:
                mock_detect.return_value = []

                start_time = time.time()
                detector.detect_objects(image)
                elapsed = time.time() - start_time

                # Should complete quickly (within 1 second for mocked operation)
                assert elapsed < 1.0

    def test_batch_processing_performance(self):
        """Test processing multiple frames efficiently."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    config=config,
                )

                mock_detector = Mock()
                mock_detector.detect_objects = Mock(return_value=[])
                mock_detector.get_detections = Mock(return_value=None)
                cortex._object_detector = mock_detector

                # Process 10 frames
                num_frames = 10
                start_time = time.time()

                for i in range(num_frames):
                    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    cortex.process_image_callback(image, {"frame_id": i}, None)

                elapsed = time.time() - start_time

                # Should process all frames
                assert cortex._processed_frames == num_frames

                # Should be reasonably fast (< 5 seconds for 10 mocked frames)
                assert elapsed < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
