"""
Unit tests for utility functions.
"""

import pytest
import numpy as np
import cv2
import time
from pathlib import Path
import tempfile
import torch

from vision_detect_segment.utils.utils import (
    validate_image, resize_image, create_test_image, load_image_safe,
    get_optimal_device, check_dependencies, validate_model_requirements,
    validate_bbox, validate_confidence_threshold, Timer,
    format_detection_results, convert_bbox_format, get_memory_usage,
    clear_gpu_cache
)
from vision_detect_segment.utils.exceptions import (
    ImageProcessingError, ConfigurationError, DependencyError
)


class TestValidateImage:
    """Tests for validate_image function."""
    
    def test_valid_color_image(self):
        """Test validation of valid color image."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        assert validate_image(image) is True
    
    def test_valid_grayscale_image(self):
        """Test validation of valid grayscale image."""
        image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        assert validate_image(image) is True
    
    def test_none_image(self):
        """Test that None image raises error."""
        with pytest.raises(ImageProcessingError, match="Image is None"):
            validate_image(None)
    
    def test_wrong_type(self):
        """Test that wrong type raises error."""
        with pytest.raises(ImageProcessingError, match="Expected numpy array"):
            validate_image([1, 2, 3])
    
    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise error."""
        image = np.random.randint(0, 255, (10,), dtype=np.uint8)
        with pytest.raises(ImageProcessingError, match="Invalid image dimensions"):
            validate_image(image)
    
    def test_too_small_image(self):
        """Test that too small image raises error."""
        image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        with pytest.raises(ImageProcessingError, match="Image too small"):
            validate_image(image, min_size=(32, 32))


class TestResizeImage:
    """Tests for resize_image function."""
    
    def test_scale_up(self):
        """Test scaling image up."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        resized, scale_x, scale_y = resize_image(image, scale_factor=2.0)
        
        assert resized.shape[0] == 200
        assert resized.shape[1] == 200
        assert scale_x == 2.0
        assert scale_y == 2.0
    
    def test_scale_down(self):
        """Test scaling image down."""
        image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        resized, scale_x, scale_y = resize_image(image, scale_factor=0.5)
        
        assert resized.shape[0] == 200
        assert resized.shape[1] == 200
        assert scale_x == 0.5
        assert scale_y == 0.5
    
    def test_max_size_constraint(self):
        """Test resizing with max size constraint."""
        image = np.random.randint(0, 255, (800, 1200, 3), dtype=np.uint8)
        resized, scale_x, scale_y = resize_image(image, max_size=(640, 480))
        
        assert resized.shape[1] <= 640
        assert resized.shape[0] <= 480
    
    def test_invalid_image(self):
        """Test that invalid image raises error."""
        with pytest.raises(ImageProcessingError):
            resize_image(None, scale_factor=2.0)


class TestCreateTestImage:
    """Tests for create_test_image function."""
    
    def test_default_creation(self):
        """Test creating test image with defaults."""
        image = create_test_image()
        assert image.shape == (480, 640, 3)
        assert image.dtype == np.uint8
    
    def test_custom_size(self):
        """Test creating test image with custom size."""
        image = create_test_image(size=(240, 320))
        assert image.shape == (240, 320, 3)
    
    def test_custom_shapes(self):
        """Test creating test image with specific shapes."""
        image = create_test_image(shapes=["square", "circle", "rectangle"])
        assert image is not None
        assert image.shape[2] == 3
    
    def test_image_not_empty(self):
        """Test that created image is not entirely black."""
        image = create_test_image(shapes=["square"])
        assert np.any(image > 0)


class TestLoadImageSafe:
    """Tests for load_image_safe function."""
    
    def test_load_existing_image(self):
        """Test loading an existing image file."""
        # Create temporary image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(tmp.name, test_image)
            tmp_path = tmp.name
        
        try:
            loaded = load_image_safe(tmp_path)
            assert loaded is not None
            assert loaded.shape[:2] == (100, 100)
        finally:
            Path(tmp_path).unlink()
    
    def test_load_nonexistent_with_fallback(self):
        """Test loading nonexistent file with fallback."""
        fallback = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = load_image_safe("nonexistent.png", fallback_image=fallback)
        assert np.array_equal(result, fallback)
    
    def test_load_nonexistent_without_fallback(self):
        """Test that loading nonexistent file without fallback raises error."""
        with pytest.raises(ImageProcessingError, match="File not found"):
            load_image_safe("nonexistent.png")


class TestGetOptimalDevice:
    """Tests for get_optimal_device function."""
    
    def test_prefer_gpu_available(self):
        """Test device selection when preferring GPU."""
        device = get_optimal_device(prefer_gpu=True)
        if torch.cuda.is_available():
            assert device == "cuda"
        else:
            assert device == "cpu"
    
    def test_cpu_only(self):
        """Test forcing CPU device."""
        device = get_optimal_device(prefer_gpu=False)
        assert device == "cpu"


class TestCheckDependencies:
    """Tests for check_dependencies function."""
    
    def test_available_packages(self):
        """Test checking available packages."""
        result = check_dependencies(["numpy", "cv2"])
        assert result["numpy"] is True
        assert result["cv2"] is True
    
    def test_unavailable_package(self):
        """Test checking unavailable package."""
        result = check_dependencies(["nonexistent_package_12345"])
        assert result["nonexistent_package_12345"] is False


class TestValidateModelRequirements:
    """Tests for validate_model_requirements function."""
    
    def test_unknown_model(self):
        """Test that unknown model raises error."""
        with pytest.raises(ConfigurationError, match="Unknown model"):
            validate_model_requirements("unknown_model")
    
    # Note: Testing actual model requirements would require installing dependencies
    # These tests verify the error handling logic


class TestValidateBbox:
    """Tests for validate_bbox function."""
    
    def test_valid_bbox(self):
        """Test validation of valid bounding box."""
        bbox = {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 200}
        image_shape = (480, 640, 3)
        assert validate_bbox(bbox, image_shape) is True
    
    def test_missing_key(self):
        """Test that missing key raises error."""
        bbox = {"x_min": 10, "y_min": 20, "x_max": 100}  # Missing y_max
        with pytest.raises(ConfigurationError, match="Missing key"):
            validate_bbox(bbox, (480, 640, 3))
    
    def test_invalid_coordinates(self):
        """Test that invalid coordinates raise error."""
        bbox = {"x_min": 100, "y_min": 20, "x_max": 50, "y_max": 200}  # x_min > x_max
        with pytest.raises(ConfigurationError, match="Invalid coordinates"):
            validate_bbox(bbox, (480, 640, 3))
    
    def test_out_of_bounds(self):
        """Test that out of bounds bbox raises error."""
        bbox = {"x_min": 10, "y_min": 20, "x_max": 1000, "y_max": 200}
        with pytest.raises(ConfigurationError, match="outside image bounds"):
            validate_bbox(bbox, (480, 640, 3))


class TestValidateConfidenceThreshold:
    """Tests for validate_confidence_threshold function."""
    
    def test_valid_threshold(self):
        """Test valid confidence thresholds."""
        assert validate_confidence_threshold(0.5) is True
        assert validate_confidence_threshold(0.0) is True
        assert validate_confidence_threshold(1.0) is True
    
    def test_invalid_type(self):
        """Test that non-numeric threshold raises error."""
        with pytest.raises(ConfigurationError, match="Must be a number"):
            validate_confidence_threshold("0.5")
    
    def test_out_of_range(self):
        """Test that out of range threshold raises error."""
        with pytest.raises(ConfigurationError, match="between 0.0 and 1.0"):
            validate_confidence_threshold(1.5)
        
        with pytest.raises(ConfigurationError, match="between 0.0 and 1.0"):
            validate_confidence_threshold(-0.1)


class TestTimer:
    """Tests for Timer context manager."""
    
    def test_timer_basic(self):
        """Test basic timer functionality."""
        with Timer("test operation") as timer:
            time.sleep(0.1)
        
        elapsed = timer.elapsed()
        assert elapsed >= 0.1
    
    def test_timer_elapsed(self):
        """Test elapsed time calculation."""
        timer = Timer("test")
        with timer:
            time.sleep(0.05)
        
        assert 0.04 <= timer.elapsed() <= 0.1


class TestFormatDetectionResults:
    """Tests for format_detection_results function."""
    
    def test_empty_detections(self):
        """Test formatting empty detections."""
        result = format_detection_results([])
        assert result == "No objects detected"
    
    def test_single_detection(self):
        """Test formatting single detection."""
        detections = [{
            "label": "test_object",
            "confidence": 0.95,
            "bbox": {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 200}
        }]
        result = format_detection_results(detections)
        assert "Found 1 objects" in result
        assert "test_object" in result
        assert "0.95" in result
    
    def test_multiple_detections(self):
        """Test formatting multiple detections."""
        detections = [
            {"label": "obj1", "confidence": 0.9, "bbox": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}},
            {"label": "obj2", "confidence": 0.8, "bbox": {"x_min": 20, "y_min": 20, "x_max": 30, "y_max": 30}}
        ]
        result = format_detection_results(detections)
        assert "Found 2 objects" in result
        assert "obj1" in result
        assert "obj2" in result
    
    def test_max_items_limit(self):
        """Test that max_items limits output."""
        detections = [{"label": f"obj{i}", "confidence": 0.9, 
                      "bbox": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}}
                     for i in range(20)]
        result = format_detection_results(detections, max_items=5)
        assert "... and 15 more" in result


class TestConvertBboxFormat:
    """Tests for convert_bbox_format function."""
    
    def test_dict_to_list(self):
        """Test converting dict to list format."""
        bbox = {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 200}
        result = convert_bbox_format(bbox, "dict", "list")
        assert result == [10, 20, 100, 200]
    
    def test_list_to_dict(self):
        """Test converting list to dict format."""
        bbox = [10, 20, 100, 200]
        result = convert_bbox_format(bbox, "list", "dict")
        assert result == {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 200}
    
    def test_list_to_tuple(self):
        """Test converting list to tuple format."""
        bbox = [10, 20, 100, 200]
        result = convert_bbox_format(bbox, "list", "tuple")
        assert result == (10, 20, 100, 200)
    
    def test_invalid_source_format(self):
        """Test that invalid source format raises error."""
        with pytest.raises(ConfigurationError):
            convert_bbox_format([10, 20, 100, 200], "invalid", "list")
    
    def test_invalid_target_format(self):
        """Test that invalid target format raises error."""
        with pytest.raises(ConfigurationError):
            convert_bbox_format([10, 20, 100, 200], "list", "invalid")


class TestGetMemoryUsage:
    """Tests for get_memory_usage function."""
    
    def test_memory_usage_format(self):
        """Test that memory usage returns expected format."""
        result = get_memory_usage()
        assert "rss_mb" in result
        assert "vms_mb" in result
        assert "percent" in result
        assert all(isinstance(v, (int, float)) for v in result.values())


class TestClearGpuCache:
    """Tests for clear_gpu_cache function."""
    
    def test_clear_cache(self):
        """Test clearing GPU cache."""
        result = clear_gpu_cache()
        assert isinstance(result, bool)
        # Returns True if CUDA available, False otherwise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
