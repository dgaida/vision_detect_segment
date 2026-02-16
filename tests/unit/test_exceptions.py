"""
Unit tests for custom exceptions.
"""

import pytest

from vision_detect_segment.utils.exceptions import (
    AnnotationError,
    ConfigurationError,
    DependencyError,
    DetectionError,
    ImageProcessingError,
    ModelLoadError,
    RedisConnectionError,
    SegmentationError,
    VisionDetectionError,
    handle_detection_error,
    handle_model_loading_error,
    handle_redis_error,
)


class TestVisionDetectionError:
    """Tests for base VisionDetectionError class."""

    def test_basic_creation(self):
        """Test basic error creation."""
        error = VisionDetectionError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.details is None

    def test_with_details(self):
        """Test error creation with details."""
        details = {"key": "value", "number": 42}
        error = VisionDetectionError("Test error", details=details)
        assert error.details == details
        assert "Details:" in str(error)


class TestModelLoadError:
    """Tests for ModelLoadError class."""

    def test_basic_creation(self):
        """Test basic ModelLoadError creation."""
        error = ModelLoadError("owlv2", "Missing dependencies")
        assert error.model_name == "owlv2"
        assert error.reason == "Missing dependencies"
        assert "Failed to load model 'owlv2'" in str(error)

    def test_with_details(self):
        """Test ModelLoadError with details."""
        error = ModelLoadError("yolo-world", "CUDA error", details="Out of memory")
        assert error.details == "Out of memory"
        assert "Details:" in str(error)


class TestDetectionError:
    """Tests for DetectionError class."""

    def test_basic_creation(self):
        """Test basic DetectionError creation."""
        error = DetectionError("Detection failed")
        assert "Detection failed" in str(error)

    def test_with_image_shape(self):
        """Test DetectionError with image shape."""
        error = DetectionError("Detection failed", image_shape=(480, 640, 3))
        assert error.image_shape == (480, 640, 3)
        assert error.details is not None
        assert "image_shape" in error.details

    def test_with_model_name(self):
        """Test DetectionError with model name."""
        error = DetectionError("Detection failed", model_name="owlv2")
        assert error.model_name == "owlv2"
        assert "model_name" in error.details


class TestSegmentationError:
    """Tests for SegmentationError class."""

    def test_basic_creation(self):
        """Test basic SegmentationError creation."""
        error = SegmentationError("Segmentation failed")
        assert "Segmentation failed" in str(error)

    def test_with_bbox(self):
        """Test SegmentationError with bounding box."""
        bbox = (10, 20, 100, 200)
        error = SegmentationError("Segmentation failed", bbox=bbox)
        assert error.bbox == bbox
        assert "bbox" in error.details

    def test_with_model(self):
        """Test SegmentationError with segmentation model."""
        error = SegmentationError("Segmentation failed", segmentation_model="sam2")
        assert error.segmentation_model == "sam2"
        assert "segmentation_model" in error.details


class TestRedisConnectionError:
    """Tests for RedisConnectionError class."""

    def test_basic_creation(self):
        """Test basic RedisConnectionError creation."""
        error = RedisConnectionError("connection", "localhost", 6379)
        assert error.operation == "connection"
        assert error.host == "localhost"
        assert error.port == 6379
        assert "Redis connection failed on localhost:6379" in str(error)

    def test_with_original_error(self):
        """Test RedisConnectionError with original error."""
        original = Exception("Connection refused")
        error = RedisConnectionError("publish", "192.168.1.1", 6380, original_error=original)
        assert error.original_error == original
        assert "Connection refused" in str(error)


class TestImageProcessingError:
    """Tests for ImageProcessingError class."""

    def test_basic_creation(self):
        """Test basic ImageProcessingError creation."""
        error = ImageProcessingError("resize")
        assert error.operation == "resize"
        assert "Image processing failed during resize" in str(error)

    def test_with_image_info(self):
        """Test ImageProcessingError with image info."""
        info = {"shape": (480, 640, 3), "dtype": "uint8"}
        error = ImageProcessingError("load", image_info=info)
        assert error.image_info == info
        assert "image_info" in error.details

    def test_with_original_error(self):
        """Test ImageProcessingError with original error."""
        original = ValueError("Invalid dimensions")
        error = ImageProcessingError("validation", original_error=original)
        assert error.original_error == original
        assert "Invalid dimensions" in str(error)


class TestConfigurationError:
    """Tests for ConfigurationError class."""

    def test_basic_creation(self):
        """Test basic ConfigurationError creation."""
        error = ConfigurationError("threshold", 1.5, "Must be between 0 and 1")
        assert error.parameter == "threshold"
        assert error.value == 1.5
        assert error.reason == "Must be between 0 and 1"
        assert "Invalid configuration for 'threshold'" in str(error)


class TestDependencyError:
    """Tests for DependencyError class."""

    def test_basic_creation(self):
        """Test basic DependencyError creation."""
        error = DependencyError("transformers", "model loading")
        assert error.dependency == "transformers"
        assert error.operation == "model loading"
        assert "Missing dependency 'transformers' for model loading" in str(error)

    def test_with_suggestion(self):
        """Test DependencyError with installation suggestion."""
        suggestion = "Install with: pip install transformers"
        error = DependencyError("transformers", "model loading", suggestion=suggestion)
        assert error.suggestion == suggestion
        assert suggestion in str(error)


class TestAnnotationError:
    """Tests for AnnotationError class."""

    def test_basic_creation(self):
        """Test basic AnnotationError creation."""
        error = AnnotationError("bounding_box", 5)
        assert error.annotation_type == "bounding_box"
        assert error.detection_count == 5
        assert "Failed to create bounding_box annotations for 5 detections" in str(error)

    def test_with_original_error(self):
        """Test AnnotationError with original error."""
        original = ValueError("Invalid detection format")
        error = AnnotationError("label", 3, original_error=original)
        assert error.original_error == original
        assert "Invalid detection format" in str(error)


class TestErrorHandlers:
    """Tests for error handling utility functions."""

    def test_handle_model_loading_error_gpu(self):
        """Test handling GPU-related model loading error."""
        original = RuntimeError("CUDA out of memory")
        result = handle_model_loading_error("owlv2", original)

        assert isinstance(result, ModelLoadError)
        assert result.model_name == "owlv2"
        assert "GPU/CUDA related error" in result.reason

    def test_handle_model_loading_error_memory(self):
        """Test handling memory-related model loading error."""
        original = MemoryError("Cannot allocate memory")
        result = handle_model_loading_error("yolo-world", original)

        assert isinstance(result, ModelLoadError)
        assert "Insufficient memory" in result.reason

    def test_handle_model_loading_error_import(self):
        """Test handling import-related model loading error."""
        original = ImportError("No module named 'transformers'")
        result = handle_model_loading_error("owlv2", original)

        assert isinstance(result, ModelLoadError)
        assert "Missing dependencies" in result.reason

    def test_handle_detection_error(self):
        """Test handling detection error."""
        original = RuntimeError("Detection failed")
        image_shape = (480, 640, 3)
        result = handle_detection_error(original, image_shape, "owlv2")

        assert isinstance(result, DetectionError)
        assert result.image_shape == image_shape
        assert result.model_name == "owlv2"
        assert "Detection failed" in str(result)

    def test_handle_redis_error(self):
        """Test handling Redis error."""
        original = ConnectionError("Connection refused")
        result = handle_redis_error("publish", "localhost", 6379, original)

        assert isinstance(result, RedisConnectionError)
        assert result.operation == "publish"
        assert result.host == "localhost"
        assert result.port == 6379
        assert result.original_error == original


class TestExceptionInheritance:
    """Tests for exception class inheritance."""

    def test_all_inherit_from_base(self):
        """Test that all custom exceptions inherit from VisionDetectionError."""
        exception_classes = [
            ModelLoadError,
            DetectionError,
            SegmentationError,
            RedisConnectionError,
            ImageProcessingError,
            ConfigurationError,
            DependencyError,
            AnnotationError,
        ]

        for exc_class in exception_classes:
            # Create minimal instance
            if exc_class == ModelLoadError:
                exc = exc_class("model", "reason")
            elif exc_class == RedisConnectionError:
                exc = exc_class("op", "host", 6379)
            elif exc_class == DetectionError:
                exc = exc_class("message")
            elif exc_class == SegmentationError:
                exc = exc_class("message")
            elif exc_class == ImageProcessingError:
                exc = exc_class("op")
            elif exc_class == ConfigurationError:
                exc = exc_class("param", "value", "reason")
            elif exc_class == DependencyError:
                exc = exc_class("dep", "op")
            elif exc_class == AnnotationError:
                exc = exc_class("type", 0)

            assert isinstance(exc, VisionDetectionError)
            assert isinstance(exc, Exception)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
