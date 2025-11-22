"""
Configuration module for vision_detect_segment package.
Contains all configurable parameters and default values.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch


@dataclass
class ModelConfig:
    """Configuration for detection models."""

    name: str
    confidence_threshold: float = 0.3
    max_detections: int = 20
    device_preference: str = "auto"  # "auto", "cuda", "cpu"

    # Model-specific parameters
    model_params: Dict[str, Any] = field(default_factory=dict)

    def get_device(self) -> str:
        """Get the actual device to use based on preference and availability."""
        if self.device_preference == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device_preference


@dataclass
class RedisConfig:
    """Configuration for Redis connections."""

    host: str = "localhost"
    port: int = 6379
    stream_name: str = "robot_camera"
    detection_stream: str = "detected_objects"
    connection_timeout: int = 5
    retry_attempts: int = 3


@dataclass
class AnnotationConfig:
    """Configuration for image annotation."""

    text_scale: float = 0.5
    text_padding: int = 3
    box_thickness: int = 2
    resize_scale_factor: float = 2.0
    show_confidence: bool = True
    show_labels: bool = True


@dataclass
class VisionConfig:
    """Main configuration class for the vision detection system."""

    # Sub-configurations
    model: ModelConfig = field(default_factory=lambda: ModelConfig("owlv2"))
    redis: RedisConfig = field(default_factory=RedisConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)

    # General settings
    verbose: bool = False
    enable_segmentation: bool = True

    # Object labels - organized by category
    _object_labels: Optional[List[str]] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize default object labels if none provided."""
        if self._object_labels is None:
            self._object_labels = self._get_default_labels()

    def get_object_labels(self) -> List[List[str]]:
        """Get object labels in the format expected by ObjectDetector."""
        return [self._object_labels]

    def add_object_label(self, label: str):
        """Add a new object label."""
        if label.lower() not in [lbl.lower() for lbl in self._object_labels]:
            self._object_labels.append(label.lower())

    def set_object_labels(self, labels: List[str]):
        """Set custom object labels."""
        self._object_labels = [label.lower() for label in labels]

    @staticmethod
    def _get_default_labels() -> List[str]:
        """Get the default set of object labels."""
        geometric_shapes = [
            "blue circle",
            "blue square",
            "blue box",
            "blue cube",
            "red circle",
            "red square",
            "red box",
            "red cube",
            "green circle",
            "green coin",
            "green cylinder",
            "green square",
            "orange cube",
            "purple cube",
            "yellow cube",
            "green cube",
        ]

        office_items = [
            "black pen",
            "black ballpoint pen",
            "pen",
            "pencil",
            "book",
            "computer mouse",
            "usb stick",
            "remote control",
            "battery",
            "batteries",
            "screwdriver",
        ]

        food_items = [
            "chocolate bar",
            "bounty",
            "snickers",
            "mars",
            "milky way",
            "twix",
            "snickers bar",
            "sweets",
            "mandarin",
            "apple",
            "coke can",
        ]

        lighting_items = ["lighter", "cigarette lighter", "philips batteries"]

        # Combine all categories
        all_labels = geometric_shapes + office_items + food_items + lighting_items
        return all_labels


# Predefined model configurations
MODEL_CONFIGS = {
    "owlv2": ModelConfig(
        name="owlv2",
        confidence_threshold=0.3,
        model_params={"model_path": "google/owlv2-base-patch16-ensemble", "requires_transformers": True},
    ),
    "yolo-world": ModelConfig(
        name="yolo-world",
        confidence_threshold=0.25,
        model_params={"model_path": "yolov8x-worldv2.pt", "requires_ultralytics": True},
    ),
    "grounding_dino": ModelConfig(
        name="grounding_dino",
        confidence_threshold=0.3,
        model_params={
            "model_path": "IDEA-Research/grounding-dino-base",
            "requires_transformers": True,
            "text_preprocessing": {"lowercase": True, "join_with_periods": True},
        },
    ),
    # NEW: YOLOE configurations
    "yoloe-11s": ModelConfig(
        name="yoloe-11s",
        confidence_threshold=0.25,
        model_params={
            "model_path": "yoloe-11s-seg.pt",
            "requires_ultralytics": True,
            "has_builtin_segmentation": True,  # YOLOE has integrated segmentation
            "supports_prompts": True,  # Supports text and visual prompts
            "prompt_free_variant": "yoloe-11s-seg-pf.pt",  # Prompt-free version
        },
    ),
    "yoloe-11m": ModelConfig(
        name="yoloe-11m",
        confidence_threshold=0.25,
        model_params={
            "model_path": "yoloe-11m-seg.pt",
            "requires_ultralytics": True,
            "has_builtin_segmentation": True,
            "supports_prompts": True,
            "prompt_free_variant": "yoloe-11m-seg-pf.pt",
        },
    ),
    "yoloe-11l": ModelConfig(
        name="yoloe-11l",
        confidence_threshold=0.25,
        model_params={
            "model_path": "yoloe-11l-seg.pt",
            "requires_ultralytics": True,
            "has_builtin_segmentation": True,
            "supports_prompts": True,
            "prompt_free_variant": "yoloe-11l-seg-pf.pt",
        },
    ),
    "yoloe-v8s": ModelConfig(
        name="yoloe-v8s",
        confidence_threshold=0.25,
        model_params={
            "model_path": "yoloe-v8s-seg.pt",
            "requires_ultralytics": True,
            "has_builtin_segmentation": True,
            "supports_prompts": True,
            "prompt_free_variant": "yoloe-v8s-seg-pf.pt",
        },
    ),
    "yoloe-v8m": ModelConfig(
        name="yoloe-v8m",
        confidence_threshold=0.25,
        model_params={
            "model_path": "yoloe-v8m-seg.pt",
            "requires_ultralytics": True,
            "has_builtin_segmentation": True,
            "supports_prompts": True,
            "prompt_free_variant": "yoloe-v8m-seg-pf.pt",
        },
    ),
    "yoloe-v8l": ModelConfig(
        name="yoloe-v8l",
        confidence_threshold=0.25,
        model_params={
            "model_path": "yoloe-v8l-seg.pt",
            "requires_ultralytics": True,
            "has_builtin_segmentation": True,
            "supports_prompts": True,
            "prompt_free_variant": "yoloe-v8l-seg-pf.pt",
        },
    ),
    # Prompt-free variants (for use without text/visual prompts)
    "yoloe-11s-pf": ModelConfig(
        name="yoloe-11s-pf",
        confidence_threshold=0.25,
        model_params={
            "model_path": "yoloe-11s-seg-pf.pt",
            "requires_ultralytics": True,
            "has_builtin_segmentation": True,
            "supports_prompts": False,  # Prompt-free uses internal vocabulary
            "is_prompt_free": True,
        },
    ),
    "yoloe-11m-pf": ModelConfig(
        name="yoloe-11m-pf",
        confidence_threshold=0.25,
        model_params={
            "model_path": "yoloe-11m-seg-pf.pt",
            "requires_ultralytics": True,
            "has_builtin_segmentation": True,
            "supports_prompts": False,
            "is_prompt_free": True,
        },
    ),
    "yoloe-11l-pf": ModelConfig(
        name="yoloe-11l-pf",
        confidence_threshold=0.25,
        model_params={
            "model_path": "yoloe-11l-seg-pf.pt",
            "requires_ultralytics": True,
            "has_builtin_segmentation": True,
            "supports_prompts": False,
            "is_prompt_free": True,
        },
    ),
}


def get_default_config(model_name: str = "owlv2") -> VisionConfig:
    """
    Get a default configuration for the specified model.

    Args:
        model_name: Name of the detection model to use

    Returns:
        VisionConfig: Default configuration instance

    Raises:
        ValueError: If model_name is not supported
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}. " f"Available models: {list(MODEL_CONFIGS.keys())}")

    config = VisionConfig()
    config.model = MODEL_CONFIGS[model_name]
    return config


def create_test_config() -> VisionConfig:
    """
    Create a configuration optimized for testing.

    Returns:
        VisionConfig: Test configuration with reduced object labels
    """
    config = get_default_config("owlv2")
    config.verbose = True
    config.model.confidence_threshold = 0.2  # Lower threshold for testing

    # Use only a subset of labels for faster testing
    test_labels = ["blue square", "chocolate bar", "mars", "snickers", "red circle", "pen", "book"]
    config.set_object_labels(test_labels)

    return config
