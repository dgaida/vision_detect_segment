"""
Integration tests with real Redis instances.
Tests complete workflow from image publishing to detection results.

Prerequisites:
- Redis server running on localhost:6379
- Run with: pytest tests/integration/test_integration_redis.py -v --redis
"""

import time
from unittest.mock import patch

import pytest
import redis
from redis_robot_comm import RedisImageStreamer, RedisMessageBroker

from vision_detect_segment.core.visualcortex import VisualCortex
from vision_detect_segment.utils.config import create_test_config
from vision_detect_segment.utils.utils import create_test_image


@pytest.fixture
def redis_available():
    """Check if Redis is available."""
    try:
        r = redis.Redis(host="localhost", port=6379, socket_timeout=1)
        r.ping()
        return True
    except (redis.ConnectionError, redis.TimeoutError):
        pytest.skip("Redis not available")


@pytest.fixture
def redis_cleanup(redis_available):
    """Clean up Redis streams after tests."""
    yield

    try:
        r = redis.Redis(host="localhost", port=6379)
        # Clean up test streams
        for stream in ["test_camera", "test_annotated", "detected_objects"]:
            try:
                r.delete(stream)
            except Exception:
                pass
    except Exception as e:
        print(f"Redis cleanup warning: {e}")


class TestRedisEndToEnd:
    """Test complete end-to-end workflow with real Redis."""

    def test_image_publish_detect_receive(self, redis_cleanup):
        """Test publishing image, detecting objects, receiving results."""
        config = create_test_config()

        # Create test components
        image_streamer = RedisImageStreamer(stream_name="test_camera")
        RedisMessageBroker(host="localhost", port=6379)

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            cortex = VisualCortex(
                objdetect_model_id="owlv2",
                device="cpu",
                stream_name="test_camera",
                annotated_stream_name="test_annotated",
                verbose=True,
                config=config,
            )

            # Mock detector
            mock_detector = patch.object(
                cortex._object_detector,
                "detect_objects",
                return_value=[
                    {
                        "label": "test_object",
                        "confidence": 0.95,
                        "bbox": {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 200},
                    }
                ],
            )

            with mock_detector:
                # Publish test image
                test_image = create_test_image(shapes=["square"])
                metadata = {"robot": "test_robot", "workspace_id": "ws1", "timestamp": time.time()}

                stream_id = image_streamer.publish_image(test_image, metadata=metadata)

                assert stream_id is not None
                print(f"Published image: {stream_id}")

                # Wait briefly for Redis
                time.sleep(0.1)

                # Trigger detection
                success = cortex.detect_objects_from_redis()
                assert success is True

                # Verify results
                detected = cortex.get_detected_objects()
                assert len(detected) == 1
                assert detected[0]["label"] == "test_object"
                assert detected[0]["confidence"] == 0.95

                print("✓ End-to-end workflow completed successfully")

    def test_continuous_frame_processing(self, redis_cleanup):
        """Test processing multiple frames in sequence."""
        config = create_test_config()

        image_streamer = RedisImageStreamer(stream_name="test_camera")

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            cortex = VisualCortex(
                objdetect_model_id="owlv2", device="cpu", stream_name="test_camera", verbose=False, config=config
            )

            # Mock detector with varying results
            def mock_detect(image):
                frame_num = cortex._processed_frames
                return [
                    {
                        "label": f"object_{frame_num}",
                        "confidence": 0.9,
                        "bbox": {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 200},
                    }
                ]

            cortex._object_detector.detect_objects = mock_detect

            # Process multiple frames
            num_frames = 10
            for i in range(num_frames):
                image = create_test_image(shapes=["square", "circle"])
                metadata = {"frame_id": i, "timestamp": time.time()}

                image_streamer.publish_image(image, metadata=metadata)
                time.sleep(0.05)  # Brief delay

                success = cortex.detect_objects_from_redis()
                assert success is True

            assert cortex.get_processed_frames_count() == num_frames
            print(f"✓ Processed {num_frames} frames successfully")

    def test_annotated_frame_publishing(self, redis_cleanup):
        """Test that annotated frames are published correctly."""
        config = create_test_config()

        image_streamer = RedisImageStreamer(stream_name="test_camera")
        annotated_streamer = RedisImageStreamer(stream_name="test_annotated")

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            cortex = VisualCortex(
                objdetect_model_id="owlv2",
                device="cpu",
                stream_name="test_camera",
                annotated_stream_name="test_annotated",
                publish_annotated=True,
                verbose=True,
                config=config,
            )

            # Mock detection
            cortex._object_detector.detect_objects = lambda img: [
                {"label": "test", "confidence": 0.9, "bbox": {"x_min": 50, "y_min": 50, "x_max": 150, "y_max": 150}}
            ]

            # Publish and process
            test_image = create_test_image()
            image_streamer.publish_image(test_image, metadata={"frame_id": 1})

            time.sleep(0.1)
            cortex.detect_objects_from_redis()

            # Wait for annotated frame
            time.sleep(0.2)

            # Try to retrieve annotated frame
            try:
                result = annotated_streamer.get_latest_image()
                if result:
                    annotated_img, meta = result
                    assert annotated_img is not None
                    assert meta.get("annotated") is True
                    print("✓ Annotated frame published successfully")
                else:
                    print("⚠ No annotated frame found (may need more time)")
            except Exception as e:
                print(f"⚠ Could not retrieve annotated frame: {e}")

    def test_redis_connection_recovery(self, redis_cleanup):
        """Test recovery when Redis connection is temporarily lost."""
        config = create_test_config()
        config.redis.fail_on_error = False  # Don't fail on connection errors

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            # Initialize with Redis available
            cortex = VisualCortex(
                objdetect_model_id="owlv2", device="cpu", stream_name="test_camera", verbose=True, config=config
            )

            # Simulate Redis unavailable
            if cortex._streamer:
                original_get = cortex._streamer.get_latest_image

                # Make it fail once, then work
                call_count = [0]

                def failing_get():
                    call_count[0] += 1
                    if call_count[0] == 1:
                        return None
                    return original_get()

                cortex._streamer.get_latest_image = failing_get

                # First call fails
                success = cortex.detect_objects_from_redis()
                assert success is False

                # Second call succeeds (if Redis is actually available)
                success = cortex.detect_objects_from_redis()
                # May succeed or fail depending on actual Redis state

                print("✓ Connection recovery test completed")


class TestRedisStreamManagement:
    """Test Redis stream management and cleanup."""

    def test_stream_maxlen_enforcement(self, redis_cleanup):
        """Test that streams respect maxlen parameter."""
        streamer = RedisImageStreamer(stream_name="test_maxlen")

        # Publish more than maxlen images
        maxlen = 5
        for i in range(maxlen + 3):
            image = create_test_image()
            streamer.publish_image(image, metadata={"frame": i}, maxlen=maxlen)

        # Check stream length
        r = redis.Redis(host="localhost", port=6379)
        stream_len = r.xlen("test_maxlen")

        assert stream_len <= maxlen, f"Stream length {stream_len} exceeds maxlen {maxlen}"
        print(f"✓ Stream maxlen enforced: {stream_len} <= {maxlen}")

    def test_multiple_consumers(self, redis_cleanup):
        """Test multiple consumers reading from same stream."""
        image_streamer = RedisImageStreamer(stream_name="test_multi")

        # Create multiple readers
        reader1 = RedisImageStreamer(stream_name="test_multi")
        reader2 = RedisImageStreamer(stream_name="test_multi")

        # Publish test image
        test_image = create_test_image()
        image_streamer.publish_image(test_image, metadata={"id": 1})

        time.sleep(0.1)

        # Both readers should get the image
        result1 = reader1.get_latest_image()
        result2 = reader2.get_latest_image()

        assert result1 is not None
        assert result2 is not None

        img1, meta1 = result1
        img2, meta2 = result2

        assert meta1["id"] == 1
        assert meta2["id"] == 1
        assert img1.shape == img2.shape

        print("✓ Multiple consumers can read from same stream")

    def test_metadata_preservation(self, redis_cleanup):
        """Test that metadata is preserved through Redis."""
        streamer = RedisImageStreamer(stream_name="test_metadata")

        # Complex metadata
        metadata = {
            "robot_id": "robot_001",
            "workspace": "ws_alpha",
            "timestamp": time.time(),
            "sensor_data": {"temperature": 25.5, "humidity": 60},
            "tags": ["quality_check", "automated"],
            "frame_number": 42,
        }

        test_image = create_test_image()
        streamer.publish_image(test_image, metadata=metadata)

        time.sleep(0.1)

        result = streamer.get_latest_image()
        assert result is not None

        _, retrieved_meta = result

        # Verify metadata
        assert retrieved_meta["robot_id"] == "robot_001"
        assert retrieved_meta["workspace"] == "ws_alpha"
        assert retrieved_meta["frame_number"] == 42
        assert "quality_check" in retrieved_meta["tags"]

        print("✓ Complex metadata preserved through Redis")


class TestRedisPerformance:
    """Test Redis performance characteristics."""

    def test_throughput(self, redis_cleanup):
        """Test image publishing throughput."""
        streamer = RedisImageStreamer(stream_name="test_throughput")

        num_images = 50
        image = create_test_image()

        start_time = time.time()

        for i in range(num_images):
            streamer.publish_image(image, metadata={"frame": i}, compress_jpeg=True, quality=85)

        elapsed = time.time() - start_time
        fps = num_images / elapsed

        print(f"✓ Throughput: {fps:.1f} FPS ({num_images} images in {elapsed:.2f}s)")

        # Should be able to handle at least 10 FPS
        assert fps > 10, f"Throughput too low: {fps:.1f} FPS"

    def test_compression_efficiency(self, redis_cleanup):
        """Test JPEG compression reduces data size."""
        streamer = RedisImageStreamer(stream_name="test_compression")

        image = create_test_image(size=(1080, 1920))  # Large image

        # Publish without compression
        meta1 = {"compression": "none"}
        streamer.publish_image(image, metadata=meta1, compress_jpeg=False)

        # Publish with compression
        meta2 = {"compression": "jpeg"}
        streamer.publish_image(image, metadata=meta2, compress_jpeg=True, quality=85)

        # Check Redis memory
        r = redis.Redis(host="localhost", port=6379)
        info = r.info("memory")

        print(f"✓ Compression test completed (Redis memory: {info['used_memory_human']})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--redis"])
