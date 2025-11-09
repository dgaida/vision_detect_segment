"""
Utility functions for the vision_detect_segment package.
Contains helper functions for image processing, logging, validation, and other common operations.
"""

import cv2
import numpy as np
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import torch

from .exceptions import ImageProcessingError, ConfigurationError, DependencyError


# Logging setup
def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the vision detection system.
    Handles Unicode output gracefully on Windows consoles.
    """
    import sys

    logger = logging.getLogger("vision_detect_segment")

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # ---- Handle console encoding problems (Windows cp1252 etc.) ----
    try:
        stream = sys.stdout
        encoding = getattr(stream, "encoding", None)

        # If Windows console with cp1252, wrap stream to use UTF-8 with replacement
        if encoding is None or encoding.lower() in ["cp1252", "ansi_x3.4-1968"]:
            import io
            stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

        console_handler = logging.StreamHandler(stream)
    except Exception:
        # Fallback to default stream handler
        console_handler = logging.StreamHandler()

    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # ---- Optional file logging ----
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not create log file {log_file}: {e}")

    return logger


# Image processing utilities
def validate_image(image: np.ndarray, min_size: Tuple[int, int] = (32, 32)) -> bool:
    """
    Validate that an image is suitable for processing.
    
    Args:
        image: Input image array
        min_size: Minimum required (height, width)
        
    Returns:
        bool: True if image is valid
        
    Raises:
        ImageProcessingError: If image is invalid
    """
    if image is None:
        raise ImageProcessingError("validation", {"error": "Image is None"})
    
    if not isinstance(image, np.ndarray):
        raise ImageProcessingError("validation",
                                   {"error": f"Expected numpy array, got {type(image)}"})
    
    if len(image.shape) not in [2, 3]:
        raise ImageProcessingError("validation",
                                   {"error": f"Invalid image dimensions: {image.shape}"})
    
    height, width = image.shape[:2]
    if height < min_size[0] or width < min_size[1]:
        raise ImageProcessingError("validation",
                                   {"error": f"Image too small: {image.shape}, minimum: {min_size}"})
    
    return True


def resize_image(image: np.ndarray, scale_factor: float = 2.0, 
                 max_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, float, float]:
    """
    Resize image with given scale factor or to maximum size.
    
    Args:
        image: Input image
        scale_factor: Scaling factor for both dimensions
        max_size: Optional maximum (width, height) - overrides scale_factor
        
    Returns:
        Tuple of (resized_image, actual_scale_x, actual_scale_y)
        
    Raises:
        ImageProcessingError: If resizing fails
    """
    try:
        validate_image(image)
        
        height, width = image.shape[:2]
        
        if max_size:
            # Calculate scale to fit within max_size
            max_width, max_height = max_size
            scale_x = max_width / width
            scale_y = max_height / height
            scale = min(scale_x, scale_y)  # Use smaller scale to fit both dimensions
        else:
            scale = scale_factor

        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        actual_scale_x = new_width / width
        actual_scale_y = new_height / height
        
        return resized, actual_scale_x, actual_scale_y

    except Exception as e:
        image_info = {
            "original_shape": image.shape if image is not None else None,
            "scale_factor": scale_factor
        }
        raise ImageProcessingError("resize", image_info, e)


def create_test_image(shapes: Optional[List[str]] = None, 
                      size: Tuple[int, int] = (480, 640)) -> np.ndarray:
    """
    Create a test image with colored geometric shapes.
    
    Args:
        shapes: List of shape types to include ("square", "circle", "rectangle")
        size: Image size as (height, width)
        
    Returns:
        np.ndarray: Test image
    """
    if shapes is None:
        shapes = ["square", "circle"]
    
    height, width = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Colors: BGR format for OpenCV
    colors = {
        "blue": (255, 0, 0),
        "red": (0, 0, 255), 
        "green": (0, 255, 0),
        "yellow": (0, 255, 255),
        "magenta": (255, 0, 255),
        "cyan": (255, 255, 0)
    }
    
    color_names = list(colors.keys())
    shape_positions = []
    
    # Calculate positions to avoid overlap
    margin = 50
    shape_size = 80
    
    for i, shape in enumerate(shapes):
        if i >= 6:  # Limit to available colors
            break
            
        color_name = color_names[i]
        color = colors[color_name]
        
        # Position shapes in a grid
        col = i % 3
        row = i // 3
        
        center_x = margin + col * (shape_size + margin) + shape_size // 2
        center_y = margin + row * (shape_size + margin) + shape_size // 2
        
        if shape == "square" or shape == "rectangle":
            top_left = (center_x - shape_size // 2, center_y - shape_size // 2)
            bottom_right = (center_x + shape_size // 2, center_y + shape_size // 2)
            cv2.rectangle(image, top_left, bottom_right, color, -1)
            
        elif shape == "circle":
            cv2.circle(image, (center_x, center_y), shape_size // 2, color, -1)
        
        # Add text label
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{color_name} {shape}"
        text_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y + shape_size // 2 + 20
        
        cv2.putText(image, text, (text_x, text_y), font, 0.4, (255, 255, 255), 1)
        
        shape_positions.append({
            "shape": shape,
            "color": color_name,
            "center": (center_x, center_y),
            "size": shape_size
        })
    
    return image


def load_image_safe(image_path: Union[str, Path], 
                    fallback_image: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Safely load an image with fallback options.
    
    Args:
        image_path: Path to image file
        fallback_image: Optional fallback image if loading fails
        
    Returns:
        np.ndarray: Loaded image or fallback
        
    Raises:
        ImageProcessingError: If loading fails and no fallback provided
    """
    try:
        image_path = Path(image_path)
        
        if not image_path.exists():
            if fallback_image is not None:
                return fallback_image
            else:
                raise ImageProcessingError("load",
                                           {"path": str(image_path), "error": "File not found"})
        
        image = cv2.imread(str(image_path))
        
        if image is None:
            if fallback_image is not None:
                return fallback_image
            else:
                raise ImageProcessingError("load",
                                           {"path": str(image_path), "error": "Could not decode image"})
        
        validate_image(image)
        return image
        
    except ImageProcessingError:
        raise
    except Exception as e:
        if fallback_image is not None:
            return fallback_image
        else:
            raise ImageProcessingError("load", {"path": str(image_path)}, e)


# Device and dependency utilities
def get_optimal_device(prefer_gpu: bool = True) -> str:
    """
    Get the optimal device for computation.
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
        
    Returns:
        str: Device string ("cuda" or "cpu")
    """
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def check_dependencies(requirements: List[str]) -> Dict[str, bool]:
    """
    Check if required packages are available.
    
    Args:
        requirements: List of package names to check
        
    Returns:
        Dict mapping package name to availability
    """
    availability = {}
    
    for package in requirements:
        try:
            __import__(package)
            availability[package] = True
        except ImportError:
            availability[package] = False
    
    return availability


def validate_model_requirements(model_name: str) -> None:
    """
    Validate that requirements for a specific model are met.
    
    Args:
        model_name: Name of the model to validate
        
    Raises:
        DependencyError: If required dependencies are missing
    """
    requirements_map = {
        "owlv2": ["transformers", "torch"],
        "grounding_dino": ["transformers", "torch"],
        "yolo-world": ["ultralytics", "torch"]
    }
    
    if model_name not in requirements_map:
        raise ConfigurationError("model_name", model_name,
                                 f"Unknown model. Available: {list(requirements_map.keys())}")
    
    required = requirements_map[model_name]
    availability = check_dependencies(required)
    
    missing = [pkg for pkg, available in availability.items() if not available]
    
    if missing:
        suggestion = f"Install with: pip install {' '.join(missing)}"
        raise DependencyError(', '.join(missing), f"model {model_name}", suggestion)


# Validation utilities  
def validate_bbox(bbox: Dict[str, int], image_shape: Tuple[int, ...]) -> bool:
    """
    Validate that a bounding box is within image boundaries.
    
    Args:
        bbox: Bounding box with keys x_min, y_min, x_max, y_max
        image_shape: Shape of the image
        
    Returns:
        bool: True if valid
        
    Raises:
        ConfigurationError: If bbox is invalid
    """
    height, width = image_shape[:2]
    
    required_keys = ["x_min", "y_min", "x_max", "y_max"]
    for key in required_keys:
        if key not in bbox:
            raise ConfigurationError("bbox", bbox, f"Missing key: {key}")
    
    x_min, y_min, x_max, y_max = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]
    
    if x_min >= x_max or y_min >= y_max:
        raise ConfigurationError("bbox", bbox, "Invalid coordinates: min >= max")
    
    if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
        raise ConfigurationError("bbox", bbox, f"Bbox outside image bounds {width}x{height}")
    
    return True


def validate_confidence_threshold(threshold: float) -> bool:
    """
    Validate confidence threshold value.
    
    Args:
        threshold: Confidence threshold
        
    Returns:
        bool: True if valid
        
    Raises:
        ConfigurationError: If threshold is invalid
    """
    if not isinstance(threshold, (int, float)):
        raise ConfigurationError("confidence_threshold", threshold, "Must be a number")
    
    if not 0.0 <= threshold <= 1.0:
        raise ConfigurationError("confidence_threshold", threshold, "Must be between 0.0 and 1.0")
    
    return True


# Performance utilities
class Timer:
    """Simple context manager for timing operations."""
    
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        self._name = name
        self._logger = logger
        self._start_time = None
    
    def __enter__(self):
        self._start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self._start_time
        message = f"{self._name} took {duration:.3f} seconds"
        
        if self._logger:
            self._logger.info(message)
        else:
            print(message)
    
    def elapsed(self) -> float:
        """Get elapsed time since start."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time


def format_detection_results(detections: List[Dict], max_items: int = 10) -> str:
    """
    Format detection results for human-readable output.
    
    Args:
        detections: List of detection dictionaries
        max_items: Maximum number of items to include
        
    Returns:
        str: Formatted detection summary
    """
    if not detections:
        return "No objects detected"
    
    lines = [f"Found {len(detections)} objects:"]
    
    for i, det in enumerate(detections[:max_items]):
        label = det.get("label", "unknown")
        confidence = det.get("confidence", 0.0)
        
        bbox = det.get("bbox", det)  # Handle both formats
        if isinstance(bbox, dict):
            coords = f"[{bbox.get('x_min', 0)}, {bbox.get('y_min', 0)}, {bbox.get('x_max', 0)}, {bbox.get('y_max', 0)}]"
        else:
            coords = f"[{det.get('x_min', 0)}, {det.get('y_min', 0)}, {det.get('x_max', 0)}, {det.get('y_max', 0)}]"
        
        lines.append(f"  {i+1}. {label} (confidence: {confidence:.2f}) at {coords}")
    
    if len(detections) > max_items:
        lines.append(f"  ... and {len(detections) - max_items} more")
    
    return "\n".join(lines)


def convert_bbox_format(bbox: Union[Dict, List, Tuple], 
                        from_format: str, to_format: str) -> Union[Dict, List]:
    """
    Convert bounding box between different formats.
    
    Args:
        bbox: Bounding box in source format
        from_format: Source format ("dict", "list", "tuple")
        to_format: Target format ("dict", "list", "tuple")
        
    Returns:
        Bounding box in target format
        
    Raises:
        ConfigurationError: If format conversion is invalid
    """
    # Extract coordinates based on source format
    if from_format == "dict":
        if not isinstance(bbox, dict):
            raise ConfigurationError("bbox_format", bbox, "Expected dict for dict format")
        x_min = bbox.get("x_min", 0)
        y_min = bbox.get("y_min", 0) 
        x_max = bbox.get("x_max", 0)
        y_max = bbox.get("y_max", 0)
    
    elif from_format in ["list", "tuple"]:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ConfigurationError("bbox_format", bbox, f"Expected 4-element {from_format}")
        x_min, y_min, x_max, y_max = bbox
    
    else:
        raise ConfigurationError("bbox_format", from_format,
                                 "Unsupported format. Use 'dict', 'list', or 'tuple'")
    
    # Convert to target format
    if to_format == "dict":
        return {"x_min": int(x_min), "y_min": int(y_min), 
                "x_max": int(x_max), "y_max": int(y_max)}
    
    elif to_format == "list":
        return [int(x_min), int(y_min), int(x_max), int(y_max)]
    
    elif to_format == "tuple":
        return (int(x_min), int(y_min), int(x_max), int(y_max))
    
    else:
        raise ConfigurationError("bbox_format", to_format,
                                 "Unsupported format. Use 'dict', 'list', or 'tuple'")


# Memory management utilities
def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.
    
    Returns:
        Dict with memory usage in MB
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
        "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        "percent": process.memory_percent()
    }


def clear_gpu_cache():
    """Clear GPU cache if CUDA is available."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return True
    except Exception:
        pass
    return False
