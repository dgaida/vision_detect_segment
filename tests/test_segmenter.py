"""
Extended unit tests for ObjectSegmenter class to increase coverage.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
import supervision as sv

from vision_detect_segment.core.object_segmenter import ObjectSegmenter
from vision_detect_segment.utils.exceptions import (
    SegmentationError,
)


class TestObjectSegmenterInitialization:
    """Tests for ObjectSegmenter initialization."""

    def test_initialization_default(self):
        """Test default initialization."""
        with patch.object(ObjectSegmenter, "_initialize_segmenter"):
            segmenter = ObjectSegmenter(device="cpu", verbose=False)
            assert segmenter._device in ["cpu", "cuda"]

    def test_initialization_with_sam2(self):
        """Test initialization with SAM2 model."""
        with patch("vision_detect_segment.core.object_segmenter.SAM2_AVAILABLE", True):
            with patch("vision_detect_segment.core.object_segmenter.SAM2ImagePredictor") as mock_sam:
                mock_sam.from_pretrained.return_value = Mock()

                segmenter = ObjectSegmenter(
                    segmentation_model="facebook/sam2.1-hiera-tiny",
                    device="cpu",
                    verbose=False,
                )

                assert segmenter._model_id == "sam2.1-hiera-tiny"
                assert segmenter._segmenter is not None

    def test_initialization_with_fastsam(self):
        """Test initialization with FastSAM model."""
        with patch("vision_detect_segment.core.object_segmenter.FASTSAM_AVAILABLE", True):
            with patch("vision_detect_segment.core.object_segmenter.FastSAM") as mock_fastsam:
                mock_fastsam.return_value = Mock()

                segmenter = ObjectSegmenter(
                    segmentation_model=None,
                    device="cpu",
                    verbose=False,
                )

                assert segmenter._model_id == "fastsam"
                assert segmenter._segmenter is not None

    def test_initialization_no_model_available(self):
        """Test initialization when no segmentation model available."""
        with patch("vision_detect_segment.core.object_segmenter.SAM2_AVAILABLE", False):
            with patch("vision_detect_segment.core.object_segmenter.FASTSAM_AVAILABLE", False):
                with patch("vision_detect_segment.core.object_segmenter.EDGETAM_AVAILABLE", False):
                    segmenter = ObjectSegmenter(device="cpu", verbose=False)

                    # Should handle gracefully
                    assert segmenter._segmenter is None

    def test_initialization_with_cuda(self):
        """Test initialization with CUDA device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        with patch.object(ObjectSegmenter, "_initialize_segmenter"):
            segmenter = ObjectSegmenter(device="cuda", verbose=False)
            assert segmenter._device == "cuda"


class TestObjectSegmenterSegmentation:
    """Tests for segmentation methods."""

    @pytest.fixture
    def mock_segmenter_fastsam(self):
        """Create mock segmenter with FastSAM."""
        with patch("vision_detect_segment.core.object_segmenter.FASTSAM_AVAILABLE", True):
            with patch("vision_detect_segment.core.object_segmenter.FastSAM") as mock_fastsam:
                mock_model = Mock()
                mock_fastsam.return_value = mock_model

                segmenter = ObjectSegmenter(device="cpu", verbose=False)
                segmenter._model_id = "fastsam"
                segmenter._segmenter = mock_model

                yield segmenter

    @pytest.fixture
    def mock_segmenter_sam2(self):
        """Create mock segmenter with SAM2."""
        with patch("vision_detect_segment.core.object_segmenter.SAM2_AVAILABLE", True):
            mock_model = Mock()

            segmenter = ObjectSegmenter.__new__(ObjectSegmenter)
            segmenter._device = "cpu"
            segmenter._model_id = "sam2.1-hiera-tiny"
            segmenter._segmenter = mock_model
            segmenter._logger = Mock()
            segmenter.verbose = False

            yield segmenter

    def test_segment_objects_basic(self, mock_segmenter_fastsam):
        """Test basic object segmentation."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )

        # Mock segmentation result
        mock_mask = np.random.randint(0, 255, (90, 90), dtype=np.uint8)
        mock_segmenter_fastsam.segment_box_in_image = Mock(return_value=(mock_mask, mock_mask > 0))

        result = mock_segmenter_fastsam.segment_objects(image, detections)

        assert result is not None
        assert isinstance(result, sv.Detections)

    def test_segment_objects_no_model(self):
        """Test segmentation when model not loaded."""
        segmenter = ObjectSegmenter.__new__(ObjectSegmenter)
        segmenter._segmenter = None

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 50, 60]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )

        with pytest.raises(SegmentationError):
            segmenter.segment_objects(image, detections)

    def test_segment_objects_invalid_image(self, mock_segmenter_fastsam):
        """Test segmentation with invalid image."""
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )

        with pytest.raises(SegmentationError):
            mock_segmenter_fastsam.segment_objects(None, detections)

    def test_segment_box_in_image_fastsam(self, mock_segmenter_fastsam):
        """Test single box segmentation with FastSAM."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        box = torch.tensor([10, 20, 100, 200])

        # Mock FastSAM results
        mock_result = Mock()
        mock_result.masks = Mock()
        mock_result.masks.data = [torch.ones((480, 640)) * 0.8]

        mock_segmenter_fastsam._segmenter.return_value = [mock_result]

        mask_8u, mask_binary = mock_segmenter_fastsam.segment_box_in_image(box, image)

        assert mask_8u is not None
        assert mask_binary is not None
        assert mask_8u.dtype == np.uint8

    def test_segment_box_in_image_sam2(self, mock_segmenter_sam2):
        """Test single box segmentation with SAM2."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        box = torch.tensor([10.0, 20.0, 100.0, 200.0])

        # Mock SAM2 prediction - return tuple of (masks, scores, logits)
        mock_masks = np.random.rand(3, 480, 640) > 0.5
        mock_scores = np.array([0.9, 0.8, 0.7])
        mock_segmenter_sam2._segmenter.predict = Mock(return_value=(mock_masks, mock_scores, None))
        mock_segmenter_sam2._segmenter.set_image = Mock()

        # Mock the _run_sam2_inference to return proper values
        with patch.object(mock_segmenter_sam2, "_run_sam2_inference") as mock_inference:
            mock_mask = (mock_masks[0] * 255).astype(np.uint8)
            mock_inference.return_value = (mock_mask, mock_masks[0] > 0)

            mask_8u, mask_binary = mock_segmenter_sam2.segment_box_in_image(box, image)

            assert mask_8u is not None
            assert mask_binary is not None

    def test_segment_box_in_image_error_handling(self, mock_segmenter_fastsam):
        """Test error handling during segmentation."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        box = torch.tensor([10, 20, 50, 60])

        # Mock segmentation to raise exception
        mock_segmenter_fastsam._segmenter.side_effect = Exception("Segmentation failed")

        # Should raise SegmentationError
        with pytest.raises(SegmentationError):
            mock_segmenter_fastsam.segment_box_in_image(box, image)


class TestObjectSegmenterFastSAMSpecific:
    """Tests specific to FastSAM segmentation."""

    @pytest.fixture
    def fastsam_segmenter(self):
        """Create FastSAM segmenter."""
        with patch("vision_detect_segment.core.object_segmenter.FASTSAM_AVAILABLE", True):
            with patch("vision_detect_segment.core.object_segmenter.FastSAM") as mock_fastsam:
                mock_model = Mock()
                mock_fastsam.return_value = mock_model

                segmenter = ObjectSegmenter(device="cpu", verbose=False)
                segmenter._model_id = "fastsam"
                segmenter._segmenter = mock_model

                yield segmenter

    def test_segment_box_with_fastsam_success(self, fastsam_segmenter):
        """Test successful FastSAM segmentation."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        box = torch.tensor([50, 50, 150, 150])

        # Mock FastSAM output
        mock_result = Mock()
        mock_result.masks = Mock()
        mock_result.masks.data = [torch.ones((480, 640)) * 0.9]

        fastsam_segmenter._segmenter.return_value = [mock_result]

        mask_8u, mask_binary = fastsam_segmenter._segment_box_with_fastsam(box, image)

        assert mask_8u is not None
        assert mask_8u.shape == image.shape[:2]

    def test_segment_box_with_fastsam_no_masks(self, fastsam_segmenter):
        """Test FastSAM with no masks returned."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        box = torch.tensor([10, 10, 50, 50])

        # Mock empty result
        mock_result = Mock()
        mock_result.masks = None

        fastsam_segmenter._segmenter.return_value = [mock_result]

        mask_8u, mask_binary = fastsam_segmenter._segment_box_with_fastsam(box, image)

        assert mask_8u is None
        assert mask_binary is None

    def test_create_mask8u(self):
        """Test mask8u creation for FastSAM."""
        img_work = np.zeros((400, 400, 3), dtype=np.uint8)
        input_box = [50, 50, 150, 150]

        # Create mock masks
        mock_masks = Mock()
        mock_masks.data = [torch.ones((400, 400)) * 0.8]

        mask_8u = ObjectSegmenter._create_mask8u(img_work, input_box, mock_masks)

        assert mask_8u.shape == (400, 400)
        assert mask_8u.dtype == np.uint8


class TestObjectSegmenterSAM2Specific:
    """Tests specific to SAM2 segmentation."""

    @pytest.fixture
    def sam2_segmenter(self):
        """Create SAM2 segmenter."""
        segmenter = ObjectSegmenter.__new__(ObjectSegmenter)
        segmenter._device = "cpu"
        segmenter._model_id = "sam2.1-hiera-tiny"
        segmenter._segmenter = Mock()
        segmenter._logger = Mock()
        segmenter.verbose = False

        yield segmenter

    def test_segment_box_with_sam2_success(self, sam2_segmenter):
        """Test successful SAM2 segmentation."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        box = torch.tensor([50.0, 50.0, 150.0, 150.0])

        # Mock SAM2 prediction
        mock_masks = np.random.rand(3, 480, 640) > 0.5
        mock_scores = np.array([0.95, 0.85, 0.75])

        sam2_segmenter._segmenter.predict = Mock(return_value=(mock_masks, mock_scores, None))
        sam2_segmenter._segmenter.set_image = Mock()

        # Mock _run_sam2_inference to return proper result
        with patch.object(sam2_segmenter, "_run_sam2_inference") as mock_inference:
            mock_mask = (mock_masks[0] * 255).astype(np.uint8)
            mock_inference.return_value = (mock_mask, mock_masks[0] > 0)

            mask_8u, mask_binary = sam2_segmenter._segment_box_with_sam2(box, image)

            assert mask_8u is not None
            assert mask_binary is not None
            assert sam2_segmenter._segmenter.set_image.called

    def test_run_sam2_inference(self, sam2_segmenter):
        """Test SAM2 inference run."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        box = torch.tensor([20.0, 20.0, 80.0, 80.0])

        # Mock prediction
        mock_masks = np.random.rand(2, 200, 200) > 0.5
        mock_scores = np.array([0.9, 0.7])

        sam2_segmenter._segmenter.set_image = Mock()
        sam2_segmenter._segmenter.predict = Mock(return_value=(mock_masks, mock_scores, None))

        mask_8u, mask_binary = sam2_segmenter._run_sam2_inference(box, image)

        assert mask_8u is not None
        # Should choose highest scoring mask
        assert sam2_segmenter._segmenter.predict.called


class TestObjectSegmenterUtilities:
    """Tests for utility methods."""

    def test_get_segmenter(self):
        """Test getting segmenter instance."""
        with patch.object(ObjectSegmenter, "_initialize_segmenter"):
            segmenter = ObjectSegmenter(device="cpu", verbose=False)
            segmenter._segmenter = Mock()

            result = segmenter.get_segmenter()

            assert result is not None

    def test_get_model_id(self):
        """Test getting model ID."""
        with patch.object(ObjectSegmenter, "_initialize_segmenter"):
            segmenter = ObjectSegmenter(device="cpu", verbose=False)
            segmenter._model_id = "test_model"

            assert segmenter.get_model_id() == "test_model"

    def test_get_device(self):
        """Test getting device."""
        with patch.object(ObjectSegmenter, "_initialize_segmenter"):
            segmenter = ObjectSegmenter(device="cpu", verbose=False)

            assert segmenter.get_device() in ["cpu", "cuda"]

    def test_is_available_true(self):
        """Test checking if segmentation is available."""
        with patch.object(ObjectSegmenter, "_initialize_segmenter"):
            segmenter = ObjectSegmenter(device="cpu", verbose=False)
            segmenter._segmenter = Mock()

            assert segmenter.is_available() is True

    def test_is_available_false(self):
        """Test checking availability when segmenter is None."""
        with patch.object(ObjectSegmenter, "_initialize_segmenter"):
            segmenter = ObjectSegmenter(device="cpu", verbose=False)
            segmenter._segmenter = None

            assert segmenter.is_available() is False

    def test_add_masks2detections(self):
        """Test adding masks to detections."""
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )

        # Add mock masks
        mask = np.random.rand(480, 640) > 0.5
        detections.mask = [mask]

        result = ObjectSegmenter.add_masks2detections(detections)

        assert result is not None
        assert isinstance(result, sv.Detections)

    def test_add_masks2detections_no_masks(self):
        """Test adding masks when no masks present."""
        detections = sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )

        result = ObjectSegmenter.add_masks2detections(detections)

        assert result is not None


class TestObjectSegmenterErrorHandling:
    """Tests for error handling in segmentation."""

    def test_segment_objects_segmentation_failure(self):
        """Test handling of segmentation failures."""
        with patch.object(ObjectSegmenter, "_initialize_segmenter"):
            segmenter = ObjectSegmenter(device="cpu", verbose=True)
            segmenter._segmenter = Mock()
            segmenter._logger = Mock()

            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            detections = sv.Detections(
                xyxy=np.array([[10, 20, 50, 60]]),
                confidence=np.array([0.9]),
                class_id=np.array([0]),
            )

            # Mock segmentation failure
            segmenter.segment_box_in_image = Mock(side_effect=Exception("Fail"))

            result = segmenter.segment_objects(image, detections)

            # Should handle error gracefully
            assert result is not None

    def test_invalid_bbox_error(self):
        """Test segmentation with invalid bounding box."""
        with patch("vision_detect_segment.core.object_segmenter.FASTSAM_AVAILABLE", True):
            with patch("vision_detect_segment.core.object_segmenter.FastSAM") as mock_fastsam:
                mock_fastsam.return_value = Mock()

                segmenter = ObjectSegmenter(device="cpu", verbose=False)
                segmenter._model_id = "fastsam"

                image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                # Invalid box (coordinates outside image)
                box = torch.tensor([10, 10, 500, 500])

                # Should handle gracefully or raise SegmentationError
                try:
                    segmenter.segment_box_in_image(box, image)
                except SegmentationError:
                    pass  # Expected


class TestObjectSegmenterBackwardCompatibility:
    """Tests for backward compatibility methods."""

    def test_deprecated_segmenter_method(self):
        """Test deprecated segmenter() method."""
        with patch.object(ObjectSegmenter, "_initialize_segmenter"):
            segmenter = ObjectSegmenter(device="cpu", verbose=False)
            segmenter._segmenter = Mock()

            result = segmenter.segmenter()

            assert result is not None

    def test_deprecated_verbose_method(self):
        """Test accessing verbose attribute."""
        with patch.object(ObjectSegmenter, "_initialize_segmenter"):
            segmenter = ObjectSegmenter(device="cpu", verbose=True)

            assert segmenter.verbose is True


class TestObjectSegmenterEdgeCases:
    """Tests for edge cases in segmentation."""

    def test_segment_very_small_box(self):
        """Test segmentation with very small bounding box."""
        with patch("vision_detect_segment.core.object_segmenter.FASTSAM_AVAILABLE", True):
            with patch("vision_detect_segment.core.object_segmenter.FastSAM") as mock_fastsam:
                mock_model = Mock()
                mock_fastsam.return_value = mock_model

                segmenter = ObjectSegmenter(device="cpu", verbose=False)
                segmenter._model_id = "fastsam"
                segmenter._segmenter = mock_model

                image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
                # Very small box
                box = torch.tensor([50, 50, 52, 52])

                mock_result = Mock()
                mock_result.masks = Mock()
                mock_result.masks.data = [torch.ones((200, 200)) * 0.9]
                mock_model.return_value = [mock_result]

                mask_8u, mask_binary = segmenter.segment_box_in_image(box, image)

                assert mask_8u is not None

    def test_segment_full_image_box(self):
        """Test segmentation with box covering entire image."""
        with patch("vision_detect_segment.core.object_segmenter.FASTSAM_AVAILABLE", True):
            with patch("vision_detect_segment.core.object_segmenter.FastSAM") as mock_fastsam:
                mock_model = Mock()
                mock_fastsam.return_value = mock_model

                segmenter = ObjectSegmenter(device="cpu", verbose=False)
                segmenter._model_id = "fastsam"
                segmenter._segmenter = mock_model

                image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                # Full image box
                box = torch.tensor([0, 0, 100, 100])

                mock_result = Mock()
                mock_result.masks = Mock()
                mock_result.masks.data = [torch.ones((100, 100)) * 0.9]
                mock_model.return_value = [mock_result]

                mask_8u, mask_binary = segmenter.segment_box_in_image(box, image)

                assert mask_8u is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
