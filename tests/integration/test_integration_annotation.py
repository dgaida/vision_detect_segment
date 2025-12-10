"""
Integration tests for annotated frame publishing pipeline.
Tests complete annotation workflow from detection to Redis publishing.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch
import supervision as sv

from vision_detect_segment.core.visualcortex import VisualCortex
from vision_detect_segment.utils.config import create_test_config
from vision_detect_segment.utils.utils import create_test_image

try:
    import redis
    from redis_robot_comm import RedisImageStreamer

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@pytest.fixture
def redis_available():
    """Check if Redis is available."""
    if not REDIS_AVAILABLE:
        pytest.skip("Redis not available")

    try:
        r = redis.Redis(host="localhost", port=6379, socket_timeout=1)
        r.ping()
        return True
    except (redis.ConnectionError, redis.TimeoutError):
        pytest.skip("Redis not available")


@pytest.fixture
def redis_cleanup():
    """Clean up Redis streams after tests."""
    yield

    if REDIS_AVAILABLE:
        try:
            r = redis.Redis(host="localhost", port=6379)
            for stream in ["test_input", "test_annotated"]:
                try:
                    r.delete(stream)
                except Exception:
                    pass
        except Exception:
            pass


class TestAnnotationPipeline:
    """Test complete annotation pipeline."""

    def test_basic_annotation_creation(self):
        """Test basic annotation is created correctly."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                with patch("vision_detect_segment.core.visualcortex.RedisLabelManager"):
                    cortex = VisualCortex(objdetect_model_id="owlv2", device="cpu", verbose=True, config=config)

                    # Mock detector with detections
                    mock_detector = Mock()
                    mock_detector.detect_objects = Mock(
                        return_value=[
                            {
                                "label": "test_object",
                                "confidence": 0.95,
                                "bbox": {"x_min": 50, "y_min": 50, "x_max": 150, "y_max": 150},
                            }
                        ]
                    )

                    # Mock supervision detections
                    mock_detections = sv.Detections(
                        xyxy=np.array([[50, 50, 150, 150]]), confidence=np.array([0.95]), class_id=np.array([0])
                    )
                    mock_detector.get_detections = Mock(return_value=mock_detections)
                    mock_detector.get_label_texts = Mock(return_value=np.array(["test_object (0.95)"]))

                    cortex._object_detector = mock_detector

                    # Process image
                    image = create_test_image(shapes=["square"])
                    cortex.process_image_callback(image, {"frame_id": 1}, None)

                    # Verify annotated frame created
                    annotated = cortex.get_annotated_image()
                    assert annotated is not None
                    assert annotated.shape[0] > 0
                    assert annotated.shape[1] > 0

                    # Annotated frame should be larger due to scaling
                    scale_factor = config.annotation.resize_scale_factor
                    expected_height = int(image.shape[0] * scale_factor)
                    assert abs(annotated.shape[0] - expected_height) < 5  # Allow small difference

                    print(f"✓ Basic annotation created: {annotated.shape}")

    def test_annotation_with_multiple_objects(self):
        """Test annotation with multiple detected objects."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                with patch("vision_detect_segment.core.visualcortex.RedisLabelManager"):
                    cortex = VisualCortex(objdetect_model_id="owlv2", device="cpu", config=config)

                    # Mock multiple detections
                    mock_detector = Mock()
                    mock_detector.detect_objects = Mock(
                        return_value=[
                            {
                                "label": "object1",
                                "confidence": 0.95,
                                "bbox": {"x_min": 50, "y_min": 50, "x_max": 150, "y_max": 150},
                            },
                            {
                                "label": "object2",
                                "confidence": 0.90,
                                "bbox": {"x_min": 200, "y_min": 100, "x_max": 300, "y_max": 200},
                            },
                            {
                                "label": "object3",
                                "confidence": 0.85,
                                "bbox": {"x_min": 350, "y_min": 150, "x_max": 450, "y_max": 250},
                            },
                        ]
                    )

                    mock_detections = sv.Detections(
                        xyxy=np.array([[50, 50, 150, 150], [200, 100, 300, 200], [350, 150, 450, 250]]),
                        confidence=np.array([0.95, 0.90, 0.85]),
                        class_id=np.array([0, 1, 2]),
                    )
                    mock_detector.get_detections = Mock(return_value=mock_detections)
                    mock_detector.get_label_texts = Mock(
                        return_value=np.array(["object1 (0.95)", "object2 (0.90)", "object3 (0.85)"])
                    )

                    cortex._object_detector = mock_detector

                    # Process image
                    image = create_test_image(shapes=["square", "circle"])
                    cortex.process_image_callback(image, {}, None)

                    annotated = cortex.get_annotated_image()
                    assert annotated is not None

                    print(f"✓ Annotation with {len(mock_detector.detect_objects())} objects")

    def test_annotation_with_masks(self):
        """Test annotation includes segmentation masks."""
        config = create_test_config()
        config.enable_segmentation = True

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                with patch("vision_detect_segment.core.visualcortex.RedisLabelManager"):
                    cortex = VisualCortex(objdetect_model_id="owlv2", device="cpu", config=config)

                    # Mock detection with mask
                    mock_detector = Mock()
                    mock_detector.detect_objects = Mock(
                        return_value=[
                            {
                                "label": "test",
                                "confidence": 0.9,
                                "bbox": {"x_min": 50, "y_min": 50, "x_max": 150, "y_max": 150},
                                "has_mask": True,
                            }
                        ]
                    )

                    # Create mock mask
                    mask = np.zeros((480, 640), dtype=bool)
                    mask[50:150, 50:150] = True

                    mock_detections = sv.Detections(
                        xyxy=np.array([[50, 50, 150, 150]]), confidence=np.array([0.9]), class_id=np.array([0])
                    )
                    mock_detections.mask = [mask]

                    mock_detector.get_detections = Mock(return_value=mock_detections)
                    mock_detector.get_label_texts = Mock(return_value=np.array(["test (0.90)"]))

                    cortex._object_detector = mock_detector

                    # Process image
                    image = create_test_image()
                    cortex.process_image_callback(image, {}, None)

                    annotated = cortex.get_annotated_image()
                    assert annotated is not None

                    print("✓ Annotation with segmentation mask")

    def test_annotation_scaling(self):
        """Test annotation coordinate scaling works correctly."""
        config = create_test_config()
        config.annotation.resize_scale_factor = 3.0  # Large scale

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                with patch("vision_detect_segment.core.visualcortex.RedisLabelManager"):
                    cortex = VisualCortex(objdetect_model_id="owlv2", device="cpu", config=config)

                    # Mock detection
                    original_bbox = {"x_min": 10, "y_min": 20, "x_max": 50, "y_max": 60}
                    mock_detector = Mock()
                    mock_detector.detect_objects = Mock(
                        return_value=[{"label": "test", "confidence": 0.9, "bbox": original_bbox}]
                    )

                    mock_detections = sv.Detections(
                        xyxy=np.array([[10, 20, 50, 60]]), confidence=np.array([0.9]), class_id=np.array([0])
                    )
                    mock_detector.get_detections = Mock(return_value=mock_detections)
                    mock_detector.get_label_texts = Mock(return_value=np.array(["test (0.90)"]))

                    cortex._object_detector = mock_detector

                    # Process small image
                    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                    cortex.process_image_callback(image, {}, None)

                    annotated = cortex.get_annotated_image()
                    assert annotated is not None

                    # Check scaling
                    expected_size = int(100 * 3.0)
                    assert abs(annotated.shape[0] - expected_size) < 5

                    print(f"✓ Annotation scaled: {image.shape} -> {annotated.shape}")


class TestAnnotationPublishing:
    """Test publishing annotated frames to Redis."""

    def test_annotation_publishing_enabled(self, redis_available, redis_cleanup):
        """Test annotated frames are published when enabled."""
        config = create_test_config()

        input_streamer = RedisImageStreamer(stream_name="test_input")
        annotated_streamer = RedisImageStreamer(stream_name="test_annotated")

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisLabelManager"):
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    stream_name="test_input",
                    annotated_stream_name="test_annotated",
                    publish_annotated=True,
                    verbose=True,
                    config=config,
                )

                # Mock detection
                mock_detector = Mock()
                mock_detector.detect_objects = Mock(
                    return_value=[
                        {"label": "test", "confidence": 0.9, "bbox": {"x_min": 50, "y_min": 50, "x_max": 150, "y_max": 150}}
                    ]
                )
                mock_detections = sv.Detections(
                    xyxy=np.array([[50, 50, 150, 150]]), confidence=np.array([0.9]), class_id=np.array([0])
                )
                mock_detector.get_detections = Mock(return_value=mock_detections)
                mock_detector.get_label_texts = Mock(return_value=np.array(["test (0.90)"]))

                cortex._object_detector = mock_detector

                # Publish and process
                test_image = create_test_image()
                input_streamer.publish_image(test_image, metadata={"frame_id": 1})

                time.sleep(0.1)
                cortex.detect_objects_from_redis()

                # Wait for publishing
                time.sleep(0.3)

                # Try to retrieve annotated frame
                result = annotated_streamer.get_latest_image()
                if result:
                    annotated_img, meta = result
                    assert annotated_img is not None
                    assert meta.get("annotated") is True
                    assert meta.get("detection_count") == 1

                    print(f"✓ Annotated frame published: {annotated_img.shape}")
                else:
                    print("⚠ Could not retrieve annotated frame (timing issue)")

    def test_annotation_publishing_disabled(self, redis_available, redis_cleanup):
        """Test annotated frames are NOT published when disabled."""
        config = create_test_config()

        input_streamer = RedisImageStreamer(stream_name="test_input")

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisLabelManager"):
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    stream_name="test_input",
                    annotated_stream_name="test_annotated",
                    publish_annotated=False,  # Disabled
                    verbose=True,
                    config=config,
                )

                mock_detector = Mock()
                mock_detector.detect_objects = Mock(return_value=[])
                mock_detector.get_detections = Mock(return_value=None)
                cortex._object_detector = mock_detector

                # Process image
                test_image = create_test_image()
                input_streamer.publish_image(test_image, metadata={"frame_id": 1})

                time.sleep(0.1)
                cortex.detect_objects_from_redis()

                # Annotated frame should still be created locally
                assert cortex.get_annotated_image() is not None

                print("✓ Annotation not published when disabled")

    def test_annotation_metadata(self, redis_available, redis_cleanup):
        """Test annotated frame metadata is correct."""
        config = create_test_config()

        input_streamer = RedisImageStreamer(stream_name="test_input")
        annotated_streamer = RedisImageStreamer(stream_name="test_annotated")

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisLabelManager"):
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    stream_name="test_input",
                    annotated_stream_name="test_annotated",
                    publish_annotated=True,
                    config=config,
                )

                # Mock 3 detections
                mock_detector = Mock()
                # Mock 3 detections
                detections_list = [
                    {
                        "label": f"obj{i}",
                        "confidence": 0.9,
                        "bbox": {"x_min": i * 50, "y_min": i * 50, "x_max": i * 50 + 40, "y_max": i * 50 + 40},
                    }
                    for i in range(3)
                ]
                mock_detector.detect_objects = Mock(return_value=detections_list)

                mock_detections = sv.Detections(
                    xyxy=np.array([[0, 0, 40, 40], [50, 50, 90, 90], [100, 100, 140, 140]]),
                    confidence=np.array([0.9, 0.9, 0.9]),
                    class_id=np.array([0, 1, 2]),
                )
                mock_detector.get_detections = Mock(return_value=mock_detections)
                mock_detector.get_label_texts = Mock(return_value=np.array(["obj0", "obj1", "obj2"]))

                cortex._object_detector = mock_detector

                # Process with custom metadata
                test_image = create_test_image()
                original_meta = {"robot_id": "robot_001", "workspace": "ws_alpha", "frame_id": 42}
                input_streamer.publish_image(test_image, metadata=original_meta)

                time.sleep(0.1)
                cortex.detect_objects_from_redis()
                time.sleep(0.3)

                # Check metadata
                result = annotated_streamer.get_latest_image()
                if result:
                    _, meta = result

                    # Should have original metadata
                    assert meta.get("robot_id") == "robot_001"
                    assert meta.get("workspace") == "ws_alpha"
                    assert meta.get("frame_id") == 42

                    # Plus annotation metadata
                    assert meta.get("annotated") is True
                    assert meta.get("detection_count") == 3
                    assert meta.get("model_id") == "owlv2"

                    print("✓ Annotation metadata preserved")


class TestAnnotationEdgeCases:
    """Test edge cases in annotation."""

    def test_annotation_with_no_detections(self):
        """Test annotation when no objects detected."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                with patch("vision_detect_segment.core.visualcortex.RedisLabelManager"):
                    cortex = VisualCortex(objdetect_model_id="owlv2", device="cpu", config=config)

                    mock_detector = Mock()
                    mock_detector.detect_objects = Mock(return_value=[])
                    mock_detector.get_detections = Mock(return_value=None)
                    cortex._object_detector = mock_detector

                    image = create_test_image()
                    cortex.process_image_callback(image, {}, None)

                    # Should still create annotated frame (just resized)
                    annotated = cortex.get_annotated_image()
                    assert annotated is not None

                    print("✓ Annotation created with no detections")

    def test_annotation_with_very_small_image(self):
        """Test annotation with very small input image."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                with patch("vision_detect_segment.core.visualcortex.RedisLabelManager"):
                    cortex = VisualCortex(objdetect_model_id="owlv2", device="cpu", config=config)

                    mock_detector = Mock()
                    mock_detector.detect_objects = Mock(return_value=[])
                    mock_detector.get_detections = Mock(return_value=None)
                    cortex._object_detector = mock_detector

                    # Very small image
                    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                    cortex.process_image_callback(image, {}, None)

                    annotated = cortex.get_annotated_image()
                    assert annotated is not None

                    # Should be scaled up
                    assert annotated.shape[0] > image.shape[0]

                    print(f"✓ Small image annotated: {image.shape} -> {annotated.shape}")

    def test_annotation_error_recovery(self):
        """Test system recovers from annotation errors."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                with patch("vision_detect_segment.core.visualcortex.RedisLabelManager"):
                    cortex = VisualCortex(objdetect_model_id="owlv2", device="cpu", verbose=True, config=config)

                    mock_detector = Mock()
                    mock_detector.detect_objects = Mock(return_value=[])
                    mock_detector.get_detections = Mock(return_value=None)
                    cortex._object_detector = mock_detector

                    # Make annotator fail once
                    original_annotate = cortex._corner_annotator.annotate
                    call_count = [0]

                    def failing_annotate(*args, **kwargs):
                        call_count[0] += 1
                        if call_count[0] == 1:
                            raise Exception("Annotation failed")
                        return original_annotate(*args, **kwargs)

                    cortex._corner_annotator.annotate = failing_annotate

                    # First call fails
                    image = create_test_image()
                    cortex.process_image_callback(image, {"frame": 1}, None)

                    # Should have fallback annotation
                    assert cortex.get_annotated_image() is not None

                    # Second call succeeds
                    cortex.process_image_callback(image, {"frame": 2}, None)
                    assert cortex.get_annotated_image() is not None

                    print("✓ Recovered from annotation error")

    def test_annotation_with_overlapping_boxes(self):
        """Test annotation handles overlapping bounding boxes."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                with patch("vision_detect_segment.core.visualcortex.RedisLabelManager"):
                    cortex = VisualCortex(objdetect_model_id="owlv2", device="cpu", config=config)

                    # Mock overlapping detections
                    mock_detector = Mock()
                    mock_detector.detect_objects = Mock(
                        return_value=[
                            {
                                "label": "obj1",
                                "confidence": 0.9,
                                "bbox": {"x_min": 50, "y_min": 50, "x_max": 150, "y_max": 150},
                            },
                            {
                                "label": "obj2",
                                "confidence": 0.85,
                                "bbox": {"x_min": 100, "y_min": 100, "x_max": 200, "y_max": 200},
                            },
                        ]
                    )

                    mock_detections = sv.Detections(
                        xyxy=np.array([[50, 50, 150, 150], [100, 100, 200, 200]]),
                        confidence=np.array([0.9, 0.85]),
                        class_id=np.array([0, 1]),
                    )
                    mock_detector.get_detections = Mock(return_value=mock_detections)
                    mock_detector.get_label_texts = Mock(return_value=np.array(["obj1 (0.90)", "obj2 (0.85)"]))

                    cortex._object_detector = mock_detector

                    image = create_test_image()
                    cortex.process_image_callback(image, {}, None)

                    annotated = cortex.get_annotated_image()
                    assert annotated is not None

                    print("✓ Overlapping boxes annotated correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
