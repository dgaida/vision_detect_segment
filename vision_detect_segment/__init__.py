# vision_detect_segment/__init__.py
"""
Vision detection and segmentation package for object detection and tracking.
"""

try:
    from .core.object_detector import ObjectDetector
    from .core.object_segmenter import ObjectSegmenter
    from .core.object_tracker import ObjectTracker
    from .core.visualcortex import VisualCortex
    from .utils.config import VisionConfig, get_default_config

    __all__ = ["VisualCortex", "ObjectDetector", "ObjectSegmenter", "ObjectTracker", "VisionConfig", "get_default_config"]
except ImportError as e:
    # Graceful degradation if dependencies are missing
    import warnings

    warnings.warn(f"Could not import vision_detect_segment components: {e}")

    __all__ = []

__version__ = "0.2.0"
