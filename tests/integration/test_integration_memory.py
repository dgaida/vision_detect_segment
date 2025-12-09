"""
Integration tests for memory leak detection over extended runs.
Tests memory stability during long-running processing.

Note: These tests may take longer to run and are marked with 'slow'.
Run with: pytest tests/integration/test_integration_memory.py -v -m slow
"""

import pytest
import numpy as np
import time
import gc
import torch
from unittest.mock import Mock, patch

from vision_detect_segment.core.visualcortex import VisualCortex
from vision_detect_segment.core.object_detector import ObjectDetector
from vision_detect_segment.utils.config import create_test_config
from vision_detect_segment.utils.utils import create_test_image, get_memory_usage, clear_gpu_cache


@pytest.mark.slow
class TestMemoryLeaks:
    """Test for memory leaks during extended processing."""

    def test_no_memory_leak_detector_only(self):
        """Test detector doesn't leak memory over many iterations."""
        config = create_test_config()

        with patch.object(ObjectDetector, "_load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test"]],
                config=config,
                enable_tracking=False,  # Disable tracking to test detector only
            )

            # Mock detection
            detector._detect_transformer_based = Mock(return_value=[])

            # Record initial memory
            gc.collect()
            initial_memory = get_memory_usage()

            # Process many frames
            num_iterations = 100
            memory_samples = []

            for i in range(num_iterations):
                image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                detector.detect_objects(image)

                # Sample memory every 10 iterations
                if i % 10 == 0:
                    gc.collect()
                    mem = get_memory_usage()
                    memory_samples.append(mem["rss_mb"])

            # Final memory check
            gc.collect()
            final_memory = get_memory_usage()

            memory_increase = final_memory["rss_mb"] - initial_memory["rss_mb"]

            print(
                f"Memory: Initial={initial_memory['rss_mb']:.1f}MB, "
                f"Final={final_memory['rss_mb']:.1f}MB, "
                f"Increase={memory_increase:.1f}MB"
            )
            print(f"Memory samples: {memory_samples}")

            # Allow some increase but not excessive
            # Acceptable increase: < 50MB for 100 iterations
            assert memory_increase < 50, f"Excessive memory increase: {memory_increase:.1f}MB"

            print("✓ No significant memory leak detected in detector")

    def test_no_memory_leak_with_tracking(self):
        """Test tracking doesn't leak memory."""
        config = create_test_config()

        with patch("vision_detect_segment.core.object_detector.YOLO_AVAILABLE", True):
            with patch("vision_detect_segment.core.object_detector.YOLO") as mock_yolo:
                mock_model = Mock()
                mock_yolo.return_value = mock_model

                detector = ObjectDetector(
                    device="cpu", model_id="yolo-world", object_labels=[["cat", "dog"]], config=config, enable_tracking=True
                )

                # Mock YOLO results
                def mock_track_result():
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

                    return [mock_result]

                detector._tracker.track = Mock(side_effect=mock_track_result)
                detector._tracker.update_label_history = Mock(side_effect=lambda ids, labels: labels)

                # Record initial memory
                gc.collect()
                initial_memory = get_memory_usage()

                # Process many frames
                num_iterations = 100

                for i in range(num_iterations):
                    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    detector.detect_objects(image)

                # Final memory check
                gc.collect()
                final_memory = get_memory_usage()

                memory_increase = final_memory["rss_mb"] - initial_memory["rss_mb"]

                print(
                    f"Tracking Memory: Initial={initial_memory['rss_mb']:.1f}MB, "
                    f"Final={final_memory['rss_mb']:.1f}MB, "
                    f"Increase={memory_increase:.1f}MB"
                )

                # Allow some increase for tracking history
                assert memory_increase < 100, f"Excessive memory increase with tracking: {memory_increase:.1f}MB"

                print("✓ No significant memory leak with tracking")

    def test_no_memory_leak_visual_cortex(self):
        """Test VisualCortex doesn't leak memory."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(objdetect_model_id="owlv2", device="cpu", verbose=False, config=config)

                # Mock detector
                mock_detector = Mock()
                mock_detector.detect_objects = Mock(return_value=[])
                mock_detector.get_detections = Mock(return_value=None)
                cortex._object_detector = mock_detector

                # Record initial memory
                gc.collect()
                initial_memory = get_memory_usage()

                # Process many frames
                num_iterations = 50

                for i in range(num_iterations):
                    image = create_test_image()
                    metadata = {"frame_id": i}
                    cortex.process_image_callback(image, metadata, None)

                # Final memory check
                gc.collect()
                final_memory = get_memory_usage()

                memory_increase = final_memory["rss_mb"] - initial_memory["rss_mb"]

                print(
                    f"VisualCortex Memory: Initial={initial_memory['rss_mb']:.1f}MB, "
                    f"Final={final_memory['rss_mb']:.1f}MB, "
                    f"Increase={memory_increase:.1f}MB"
                )

                assert memory_increase < 100, f"Excessive memory increase in VisualCortex: {memory_increase:.1f}MB"

                print("✓ No significant memory leak in VisualCortex")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_cleanup(self):
        """Test GPU memory is properly released."""
        config = create_test_config()

        with patch.object(ObjectDetector, "_load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(device="cuda", model_id="owlv2", object_labels=[["test"]], config=config)

            # Record initial GPU memory
            initial_gpu = torch.cuda.memory_allocated() / 1024**2  # MB

            # Process some frames
            for i in range(10):
                image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                with patch.object(detector, "_detect_transformer_based", return_value=[]):
                    detector.detect_objects(image)

            # Clear cache
            clear_gpu_cache()

            # Check GPU memory
            final_gpu = torch.cuda.memory_allocated() / 1024**2

            print(f"GPU Memory: Initial={initial_gpu:.1f}MB, Final={final_gpu:.1f}MB")

            # GPU memory should be released
            # Allow some variance due to cached allocations
            assert final_gpu < initial_gpu + 100, f"GPU memory not released: {final_gpu:.1f}MB"

            print("✓ GPU memory properly cleaned")


@pytest.mark.slow
class TestMemoryStability:
    """Test memory stability over time."""

    def test_memory_stable_over_time(self):
        """Test memory remains stable over extended period."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(objdetect_model_id="owlv2", device="cpu", verbose=False, config=config)

                mock_detector = Mock()
                mock_detector.detect_objects = Mock(return_value=[])
                mock_detector.get_detections = Mock(return_value=None)
                cortex._object_detector = mock_detector

                # Sample memory at intervals
                memory_samples = []
                num_iterations = 200
                sample_interval = 20

                for i in range(num_iterations):
                    image = create_test_image()
                    cortex.process_image_callback(image, {"frame_id": i}, None)

                    if i % sample_interval == 0:
                        gc.collect()
                        mem = get_memory_usage()
                        memory_samples.append(mem["rss_mb"])
                        print(f"Frame {i}: {mem['rss_mb']:.1f}MB")

                # Check memory trend
                # Calculate linear regression slope
                x = np.arange(len(memory_samples))
                y = np.array(memory_samples)
                slope = np.polyfit(x, y, 1)[0]

                print(f"Memory slope: {slope:.2f} MB/sample")
                print(f"Memory samples: {memory_samples}")

                # Slope should be close to zero (stable)
                # Allow slight upward trend due to Python overhead
                assert abs(slope) < 2.0, f"Memory not stable, slope: {slope:.2f} MB/sample"

                print(f"✓ Memory stable over {num_iterations} iterations")

    def test_no_memory_accumulation_with_detections(self):
        """Test no memory accumulation when repeatedly detecting objects."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(objdetect_model_id="owlv2", device="cpu", verbose=False, config=config)

                # Mock detector with actual detections
                mock_detector = Mock()
                mock_detector.detect_objects = Mock(
                    return_value=[
                        {"label": "test", "confidence": 0.9, "bbox": {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 200}}
                    ]
                )
                mock_detector.get_detections = Mock(return_value=None)
                cortex._object_detector = mock_detector

                # Record memory samples
                memory_samples = []

                for i in range(100):
                    image = create_test_image()
                    cortex.process_image_callback(image, {"frame_id": i}, None)

                    if i % 10 == 0:
                        gc.collect()
                        mem = get_memory_usage()
                        memory_samples.append(mem["rss_mb"])

                # Check memory variance
                memory_std = np.std(memory_samples)
                print(f"Memory std dev: {memory_std:.2f}MB")
                print(f"Memory range: {min(memory_samples):.1f} - {max(memory_samples):.1f}MB")

                # Standard deviation should be small
                assert memory_std < 10.0, f"High memory variance: {memory_std:.2f}MB"

                print("✓ No memory accumulation with detections")


@pytest.mark.slow
class TestResourceCleanup:
    """Test proper resource cleanup."""

    def test_detector_cleanup_on_delete(self):
        """Test detector properly cleans up when deleted."""
        config = create_test_config()

        # Record initial memory
        gc.collect()
        initial_memory = get_memory_usage()

        # Create and use detector
        with patch.object(ObjectDetector, "_load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(device="cpu", model_id="owlv2", object_labels=[["test"]], config=config)

            # Use detector
            for _ in range(10):
                image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                with patch.object(detector, "_detect_transformer_based", return_value=[]):
                    detector.detect_objects(image)

            # Delete detector
            del detector

        # Force garbage collection
        gc.collect()
        time.sleep(0.1)

        # Check memory is released
        final_memory = get_memory_usage()
        memory_diff = final_memory["rss_mb"] - initial_memory["rss_mb"]

        print(
            f"Memory after cleanup: Initial={initial_memory['rss_mb']:.1f}MB, "
            f"Final={final_memory['rss_mb']:.1f}MB, Diff={memory_diff:.1f}MB"
        )

        # Memory should be mostly released
        assert memory_diff < 20, f"Memory not properly released: {memory_diff:.1f}MB remaining"

        print("✓ Detector cleanup successful")

    def test_cortex_cleanup_on_delete(self):
        """Test VisualCortex properly cleans up."""
        config = create_test_config()

        gc.collect()
        initial_memory = get_memory_usage()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(objdetect_model_id="owlv2", device="cpu", config=config)

                # Use cortex
                mock_detector = Mock()
                mock_detector.detect_objects = Mock(return_value=[])
                mock_detector.get_detections = Mock(return_value=None)
                cortex._object_detector = mock_detector

                for i in range(10):
                    image = create_test_image()
                    cortex.process_image_callback(image, {"frame": i}, None)

                # Cleanup
                cortex.cleanup()
                del cortex

        gc.collect()
        time.sleep(0.1)

        final_memory = get_memory_usage()
        memory_diff = final_memory["rss_mb"] - initial_memory["rss_mb"]

        print(f"VisualCortex cleanup: Diff={memory_diff:.1f}MB")

        assert memory_diff < 30, f"VisualCortex memory not released: {memory_diff:.1f}MB"

        print("✓ VisualCortex cleanup successful")

    def test_multiple_creation_deletion_cycles(self):
        """Test multiple create/delete cycles don't accumulate memory."""
        config = create_test_config()

        memory_samples = []

        for cycle in range(5):
            gc.collect()
            get_memory_usage()

            with patch.object(ObjectDetector, "_load_model") as mock_load:
                mock_load.return_value = (Mock(), Mock())

                detector = ObjectDetector(device="cpu", model_id="owlv2", object_labels=[["test"]], config=config)

                # Use detector
                for _ in range(5):
                    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                    with patch.object(detector, "_detect_transformer_based", return_value=[]):
                        detector.detect_objects(image)

                del detector

            gc.collect()
            time.sleep(0.1)

            mem_after = get_memory_usage()
            memory_samples.append(mem_after["rss_mb"])

            print(f"Cycle {cycle}: {mem_after['rss_mb']:.1f}MB")

        # Check memory doesn't grow across cycles
        memory_growth = memory_samples[-1] - memory_samples[0]

        print(f"Memory growth over {len(memory_samples)} cycles: {memory_growth:.1f}MB")

        assert memory_growth < 20, f"Memory accumulating across cycles: {memory_growth:.1f}MB"

        print(f"✓ No accumulation over {len(memory_samples)} cycles")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])
