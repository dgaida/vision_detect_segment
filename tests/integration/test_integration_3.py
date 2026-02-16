"""
Integration tests for vision_detect_segment package - Part 3.
Tests edge cases, stress scenarios, and complex workflows.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import supervision as sv
import torch

from vision_detect_segment.core.object_detector import ObjectDetector
from vision_detect_segment.core.object_segmenter import ObjectSegmenter
from vision_detect_segment.core.object_tracker import ObjectTracker
from vision_detect_segment.core.visualcortex import VisualCortex
from vision_detect_segment.utils.config import create_test_config
from vision_detect_segment.utils.exceptions import (
    DetectionError,
    ImageProcessingError,
)


class TestEdgeCaseImages:
    """Test handling of edge case images."""

    @pytest.fixture
    def detector_for_edge_cases(self):
        """Create detector for edge case testing."""
        config = create_test_config()

        with patch("vision_detect_segment.core.object_detector.ObjectDetector._load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test"]],
                config=config,
            )

            yield detector

    def test_very_small_image(self, detector_for_edge_cases):
        """Test detection on very small image."""
        detector = detector_for_edge_cases

        # 32x32 is minimum size
        small_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)

        with patch.object(detector, "_detect_transformer_based") as mock_detect:
            mock_detect.return_value = []

            results = detector.detect_objects(small_image)
            assert isinstance(results, list)

    def test_very_large_image(self, detector_for_edge_cases):
        """Test detection on very large image."""
        detector = detector_for_edge_cases

        # Test with 4K resolution
        large_image = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)

        with patch.object(detector, "_detect_transformer_based") as mock_detect:
            mock_detect.return_value = []

            results = detector.detect_objects(large_image)
            assert isinstance(results, list)

    def test_grayscale_image(self, detector_for_edge_cases):
        """Test detection on grayscale image."""
        detector = detector_for_edge_cases

        gray_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

        with patch.object(detector, "_detect_transformer_based") as mock_detect:
            mock_detect.return_value = []

            # Should handle or convert grayscale appropriately
            try:
                detector.detect_objects(gray_image)
            except (DetectionError, ImageProcessingError):
                # Expected for grayscale without conversion
                pass

    def test_unusual_aspect_ratio(self, detector_for_edge_cases):
        """Test detection on image with unusual aspect ratio."""
        detector = detector_for_edge_cases

        # Very wide image
        wide_image = np.random.randint(0, 255, (100, 2000, 3), dtype=np.uint8)

        with patch.object(detector, "_detect_transformer_based") as mock_detect:
            mock_detect.return_value = []

            results = detector.detect_objects(wide_image)
            assert isinstance(results, list)


class TestHighLoadScenarios:
    """Test system behavior under high load."""

    def test_many_objects_in_frame(self):
        """Test detection with many objects in single frame."""
        config = create_test_config()

        with patch("vision_detect_segment.core.object_detector.ObjectDetector._load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["object"]],
                config=config,
            )

            image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

            # Mock 50 detections
            many_detections = [
                {
                    "label": f"object_{i}",
                    "confidence": 0.9,
                    "bbox": {"x_min": i * 30, "y_min": i * 20, "x_max": i * 30 + 50, "y_max": i * 20 + 50},
                }
                for i in range(50)
            ]

            with patch.object(detector, "_detect_transformer_based") as mock_detect:
                mock_detect.return_value = many_detections

                results = detector.detect_objects(image)
                assert len(results) <= 50  # May be limited by max_detections

    def test_rapid_frame_succession(self):
        """Test processing many frames in rapid succession."""
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

                # Process 100 frames rapidly
                for i in range(100):
                    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    cortex.process_image_callback(image, {"frame_id": i}, None)

                assert cortex._processed_frames == 100

    def test_long_running_session(self):
        """Test stability over long running session."""
        config = create_test_config()

        with patch("vision_detect_segment.core.object_detector.ObjectDetector._load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test"]],
                config=config,
            )

            # Simulate 1000 detections
            for _ in range(1000):
                image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

                with patch.object(detector, "_detect_transformer_based") as mock_detect:
                    mock_detect.return_value = []

                    detector.detect_objects(image)

            # Should still be functional
            labels = detector.get_object_labels()
            assert len(labels) > 0


class TestTrackingEdgeCases:
    """Test tracking in edge case scenarios."""

    def test_object_entering_leaving_frame(self):
        """Test tracking when objects enter and leave frame."""
        mock_model = Mock()

        tracker = ObjectTracker(
            model=mock_model,
            model_id="owlv2",
            enable_tracking=True,
            verbose=False,
        )

        # Frame 1: Object present
        track_ids_1 = np.array([1])
        labels_1 = ["cat"]
        tracker.update_label_history(track_ids_1, labels_1)

        # Frame 2: Object still present
        track_ids_2 = np.array([1])
        labels_2 = ["cat"]
        tracker.update_label_history(track_ids_2, labels_2)

        # Frame 3: Object left (empty frame) - track ID 1 is missing
        track_ids_3 = np.array([])  # Empty array - no tracks present
        lost = tracker.detect_lost_tracks(track_ids_3)
        assert 1 in lost  # track 1 is temporarily lost, but only in one frame, this can be compensated

        # Frame 4: New object enters with same ID - this object is re-identified as the previous object.
        track_ids_4 = np.array([1])
        labels_4 = ["dog"]
        display_labels = tracker.update_label_history(track_ids_4, labels_4)

        print(display_labels)

        # Should treat as new track - no it does not, it looks at majority vote. in the first 2 frames it was a cat,
        # now it is a dog. so the frame count is 3 and the majority vote is cat
        info = tracker.get_track_info(1)
        assert info["frame_count"] == 3

    def test_occlusion_handling(self):
        """Test tracking through occlusion."""
        mock_model = Mock()

        tracker = ObjectTracker(
            model=mock_model,
            model_id="owlv2",
            enable_tracking=True,
            verbose=False,
        )

        # Simulate object visible -> occluded -> visible again
        frames = [
            (np.array([1, 2]), ["cat", "dog"]),  # Both visible
            (np.array([2]), ["dog"]),  # Cat occluded
            (np.array([1, 2]), ["cat", "dog"]),  # Both visible again
        ]

        for track_ids, labels in frames:
            tracker.update_label_history(track_ids, labels)

        # Both tracks should still exist
        info_1 = tracker.get_track_info(1)
        info_2 = tracker.get_track_info(2)

        assert info_1 is not None
        assert info_2 is not None

    def test_id_switch_detection(self):
        """Test detection of track ID switches."""
        mock_model = Mock()

        tracker = ObjectTracker(
            model=mock_model,
            model_id="owlv2",
            enable_tracking=True,
            verbose=False,
        )

        # Track object with consistent ID
        for _ in range(5):
            tracker.update_label_history(np.array([1]), ["cat"])

        # Sudden ID change (tracker reassignment)
        tracker.update_label_history(np.array([2]), ["cat"])

        # Old track should be detected as lost
        lost = tracker.detect_lost_tracks(np.array([2]))
        assert 1 in lost


class TestSegmentationEdgeCases:
    """Test segmentation edge cases."""

    def test_segmentation_with_tiny_bbox(self):
        """Test segmentation with very small bounding box."""
        with patch("vision_detect_segment.core.object_segmenter.FASTSAM_AVAILABLE", True):
            with patch("vision_detect_segment.core.object_segmenter.FastSAM") as mock_fastsam:
                mock_model = Mock()
                mock_fastsam.return_value = mock_model

                segmenter = ObjectSegmenter(device="cpu", verbose=False)
                segmenter._model_id = "fastsam"
                segmenter._segmenter = mock_model

                image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

                # 5x5 pixel box
                tiny_box = torch.tensor([100, 100, 105, 105])

                # Mock result
                mock_result = Mock()
                mock_result.masks = Mock()
                mock_result.masks.data = [torch.ones((480, 640)) * 0.9]
                mock_model.return_value = [mock_result]

                mask_8u, mask_binary = segmenter.segment_box_in_image(tiny_box, image)

                # Should handle tiny box
                assert mask_8u is None or mask_8u.shape == image.shape[:2]

    def test_segmentation_with_overlapping_boxes(self):
        """Test segmentation with overlapping bounding boxes."""
        with patch("vision_detect_segment.core.object_segmenter.FASTSAM_AVAILABLE", True):
            with patch("vision_detect_segment.core.object_segmenter.FastSAM") as mock_fastsam:
                mock_model = Mock()
                mock_fastsam.return_value = mock_model

                segmenter = ObjectSegmenter(device="cpu", verbose=False)
                segmenter._model_id = "fastsam"
                segmenter._segmenter = mock_model

                image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

                # Create detections with overlapping boxes
                detections = sv.Detections(
                    xyxy=np.array(
                        [
                            [50, 50, 150, 150],
                            [100, 100, 200, 200],  # Overlaps first box
                        ]
                    ),
                    confidence=np.array([0.9, 0.85]),
                    class_id=np.array([0, 1]),
                )

                # Mock segmentation for each box
                def mock_segment(box, img):
                    mask = np.random.randint(0, 255, img.shape[:2], dtype=np.uint8)
                    return mask, mask > 127

                segmenter.segment_box_in_image = Mock(side_effect=mock_segment)

                result = segmenter.segment_objects(image, detections)

                assert result is not None


class TestConfigurationEdgeCases:
    """Test configuration edge cases."""

    def test_extreme_confidence_thresholds(self):
        """Test with extreme confidence threshold values."""
        config = create_test_config()

        # Very low threshold
        config.model.confidence_threshold = 0.01

        with patch("vision_detect_segment.core.object_detector.ObjectDetector._load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test"]],
                config=config,
            )

            assert detector._config.model.confidence_threshold == 0.01

        # Very high threshold
        config.model.confidence_threshold = 0.99

        with patch("vision_detect_segment.core.object_detector.ObjectDetector._load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test"]],
                config=config,
            )

            assert detector._config.model.confidence_threshold == 0.99

    def test_empty_label_list(self):
        """Test handling of empty label list."""
        config = create_test_config()
        config.set_object_labels([])

        with patch("vision_detect_segment.core.object_detector.ObjectDetector._load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            # Should handle empty labels gracefully
            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[[]],
                config=config,
            )

            labels = detector.get_object_labels()
            assert len(labels) == 1
            assert len(labels[0]) == 0

    def test_very_long_label_list(self):
        """Test with very long label list."""
        config = create_test_config()

        # 1000 labels
        many_labels = [f"object_{i}" for i in range(1000)]
        config.set_object_labels(many_labels)

        with patch("vision_detect_segment.core.object_detector.ObjectDetector._load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=config.get_object_labels(),
                config=config,
            )

            labels = detector.get_object_labels()[0]
            assert len(labels) == 1000


class TestConcurrentOperations:
    """Test concurrent operations and race conditions."""

    def test_concurrent_label_updates(self):
        """Test concurrent label updates don't cause issues."""
        config = create_test_config()

        with patch("vision_detect_segment.core.object_detector.ObjectDetector._load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["initial"]],
                config=config,
            )

            # Rapidly add labels
            for i in range(100):
                detector.add_label(f"object_{i}")

            labels = detector.get_object_labels()[0]
            assert len(labels) >= 100

    def test_detection_during_label_update(self):
        """Test detection while labels are being updated."""
        config = create_test_config()

        with patch("vision_detect_segment.core.object_detector.ObjectDetector._load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test"]],
                config=config,
            )

            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # Add label
            detector.add_label("new_object")

            # Immediately run detection
            with patch.object(detector, "_detect_transformer_based") as mock_detect:
                mock_detect.return_value = []

                results = detector.detect_objects(image)
                assert isinstance(results, list)


class TestResourceCleanup:
    """Test proper resource cleanup."""

    def test_detector_cleanup(self):
        """Test detector properly cleans up resources."""
        config = create_test_config()

        with patch("vision_detect_segment.core.object_detector.ObjectDetector._load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test"]],
                config=config,
            )

            # Simulate cleanup
            del detector

            # Should not cause issues

    def test_cortex_cleanup(self):
        """Test VisualCortex properly cleans up resources."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    config=config,
                )

                # Process some frames
                for i in range(10):
                    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    cortex.process_image_callback(image, {}, None)

                # Clear cache
                cortex.clear_cache()

                # Cleanup
                del cortex


class TestBackwardCompatibility:
    """Test backward compatibility with deprecated APIs."""

    def test_deprecated_detector_methods(self):
        """Test deprecated detector methods still work."""
        config = create_test_config()

        with patch("vision_detect_segment.core.object_detector.ObjectDetector._load_model") as mock_load:
            mock_load.return_value = (Mock(), Mock())

            detector = ObjectDetector(
                device="cpu",
                model_id="owlv2",
                object_labels=[["test"]],
                config=config,
            )

            # Test deprecated methods
            detector.detections()
            detector.label_texts()
            detector.object_labels()

            # Should not raise errors

    def test_deprecated_cortex_methods(self):
        """Test deprecated VisualCortex methods still work."""
        config = create_test_config()

        with patch("vision_detect_segment.core.visualcortex.ObjectDetector"):
            with patch("vision_detect_segment.core.visualcortex.RedisImageStreamer"):
                cortex = VisualCortex(
                    objdetect_model_id="owlv2",
                    device="cpu",
                    config=config,
                )

                cortex._img_work = np.zeros((100, 100, 3), dtype=np.uint8)
                cortex._annotated_frame = np.zeros((200, 200, 3), dtype=np.uint8)

                # Test deprecated methods
                img = cortex.img_work()
                frame = cortex.annotated_frame()

                assert img is not None
                assert frame is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
