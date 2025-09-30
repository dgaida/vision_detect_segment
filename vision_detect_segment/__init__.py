# __init__.py
from .visualcortex import VisualCortex
from .object_detector import ObjectDetector
from .object_segmenter import ObjectSegmenter
from .config import VisionConfig

__version__ = "0.2.0"
__all__ = ["VisualCortex", "ObjectDetector", "ObjectSegmenter", "VisionConfig"]
