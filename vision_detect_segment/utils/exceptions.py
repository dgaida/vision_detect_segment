"""
Custom exceptions for the vision_detect_segment package.
Provides specific error types for better error handling and debugging.
"""

from typing import Optional, Any


class VisionDetectionError(Exception):
    """
    Base exception class for all vision detection errors.

    All other custom exceptions in this package should inherit from this class.
    """

    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class ModelLoadError(VisionDetectionError):
    """
    Exception raised when a detection model fails to load.

    This can happen due to:
    - Missing model files
    - Incompatible model versions
    - Insufficient GPU memory
    - Missing dependencies
    """

    def __init__(self, model_name: str, reason: str, details: Optional[Any] = None):
        self.model_name = model_name
        self.reason = reason
        message = f"Failed to load model '{model_name}': {reason}"
        super().__init__(message, details)


class DetectionError(VisionDetectionError):
    """
    Exception raised when object detection fails during inference.

    This can happen due to:
    - Invalid input image
    - Model inference errors
    - GPU out of memory
    - Preprocessing failures
    """

    def __init__(self, message: str, image_shape: Optional[tuple] = None, model_name: Optional[str] = None):
        self.image_shape = image_shape
        self.model_name = model_name

        details = {}
        if image_shape:
            details["image_shape"] = image_shape
        if model_name:
            details["model_name"] = model_name

        super().__init__(message, details if details else None)


class SegmentationError(VisionDetectionError):
    """
    Exception raised when image segmentation fails.

    This can happen due to:
    - Segmentation model not available
    - Invalid bounding box coordinates
    - Segmentation model errors
    """

    def __init__(self, message: str, bbox: Optional[tuple] = None, segmentation_model: Optional[str] = None):
        self.bbox = bbox
        self.segmentation_model = segmentation_model

        details = {}
        if bbox:
            details["bbox"] = bbox
        if segmentation_model:
            details["segmentation_model"] = segmentation_model

        super().__init__(message, details if details else None)


class RedisConnectionError(VisionDetectionError):
    """
    Exception raised when Redis connection or operations fail.

    This can happen due to:
    - Redis server not running
    - Network connectivity issues
    - Authentication failures
    - Invalid Redis configurations
    """

    def __init__(self, operation: str, host: str, port: int, original_error: Optional[Exception] = None):
        self.operation = operation
        self.host = host
        self.port = port
        self.original_error = original_error

        message = f"Redis {operation} failed on {host}:{port}"
        if original_error:
            message += f" - {str(original_error)}"

        details = {
            "operation": operation,
            "host": host,
            "port": port,
            "original_error": str(original_error) if original_error else None,
        }

        super().__init__(message, details)


class ImageProcessingError(VisionDetectionError):
    """
    Exception raised when image processing operations fail.

    This can happen due to:
    - Invalid image format
    - Corrupted image data
    - Unsupported image dimensions
    - Memory allocation failures
    """

    def __init__(self, operation: str, image_info: Optional[dict] = None, original_error: Optional[Exception] = None):
        self.operation = operation
        self.image_info = image_info
        self.original_error = original_error

        message = f"Image processing failed during {operation}"
        if original_error:
            message += f": {str(original_error)}"

        details = {
            "operation": operation,
            "image_info": image_info,
            "original_error": str(original_error) if original_error else None,
        }

        super().__init__(message, details)


class ConfigurationError(VisionDetectionError):
    """
    Exception raised when there are configuration-related errors.

    This can happen due to:
    - Invalid configuration parameters
    - Missing required configuration values
    - Incompatible configuration combinations
    """

    def __init__(self, parameter: str, value: Any, reason: str):
        self.parameter = parameter
        self.value = value
        self.reason = reason

        message = f"Invalid configuration for '{parameter}': {reason}"
        details = {"parameter": parameter, "value": value, "reason": reason}

        super().__init__(message, details)


class DependencyError(VisionDetectionError):
    """
    Exception raised when required dependencies are missing or incompatible.

    This can happen due to:
    - Missing Python packages
    - Incompatible package versions
    - Missing system libraries
    """

    def __init__(self, dependency: str, operation: str, suggestion: Optional[str] = None):
        self.dependency = dependency
        self.operation = operation
        self.suggestion = suggestion

        message = f"Missing dependency '{dependency}' for {operation}"
        if suggestion:
            message += f". {suggestion}"

        details = {"dependency": dependency, "operation": operation, "suggestion": suggestion}

        super().__init__(message, details)


class AnnotationError(VisionDetectionError):
    """
    Exception raised when image annotation fails.

    This can happen due to:
    - Invalid detection data
    - Annotation library errors
    - Image format incompatibilities
    """

    def __init__(self, annotation_type: str, detection_count: int, original_error: Optional[Exception] = None):
        self.annotation_type = annotation_type
        self.detection_count = detection_count
        self.original_error = original_error

        message = f"Failed to create {annotation_type} annotations for {detection_count} detections"
        if original_error:
            message += f": {str(original_error)}"

        details = {
            "annotation_type": annotation_type,
            "detection_count": detection_count,
            "original_error": str(original_error) if original_error else None,
        }

        super().__init__(message, details)


# Utility functions for exception handling
def handle_model_loading_error(model_name: str, error: Exception) -> ModelLoadError:
    """
    Convert a generic exception to a ModelLoadError with useful context.

    Args:
        model_name: Name of the model that failed to load
        error: The original exception

    Returns:
        ModelLoadError: Wrapped exception with additional context
    """
    if "CUDA" in str(error) or "GPU" in str(error):
        reason = "GPU/CUDA related error"
    elif "memory" in str(error).lower():
        reason = "Insufficient memory"
    elif "import" in str(error).lower() or "module" in str(error).lower():
        reason = "Missing dependencies"
    else:
        reason = "Unknown error"

    return ModelLoadError(model_name, reason, str(error))


def handle_detection_error(error: Exception, image_shape: tuple, model_name: str) -> DetectionError:
    """
    Convert a generic exception to a DetectionError with useful context.

    Args:
        error: The original exception
        image_shape: Shape of the input image
        model_name: Name of the detection model

    Returns:
        DetectionError: Wrapped exception with additional context
    """
    message = f"Detection failed: {str(error)}"
    return DetectionError(message, image_shape, model_name)


def handle_redis_error(operation: str, host: str, port: int, error: Exception) -> RedisConnectionError:
    """
    Convert a generic exception to a RedisConnectionError with useful context.

    Args:
        operation: The Redis operation that failed
        host: Redis host
        port: Redis port
        error: The original exception

    Returns:
        RedisConnectionError: Wrapped exception with additional context
    """
    return RedisConnectionError(operation, host, port, error)
