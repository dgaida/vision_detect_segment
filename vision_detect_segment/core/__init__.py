# vision_detect_segment/core/__init__.py
"""
Core detection and segmentation modules.
"""

try:
    from .visualcortex import VisualCortex
    from .object_detector import ObjectDetector
    from .object_segmenter import ObjectSegmenter
    from .object_tracker import ObjectTracker

    __all__ = ["VisualCortex", "ObjectDetector", "ObjectSegmenter", "ObjectTracker"]
except ImportError as e:
    # Graceful degradation
    import warnings

    warnings.warn(f"Could not import core modules: {e}")
    __all__ = []
