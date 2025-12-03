# API Reference

Complete API documentation for the `vision_detect_segment` package.

---

## Table of Contents

- [Core Classes](#core-classes)
  - [VisualCortex](#visualcortex)
  - [ObjectDetector](#objectdetector)
  - [ObjectSegmenter](#objectsegmenter)
  - [ObjectTracker](#objecttracker)
- [Configuration](#configuration)
  - [VisionConfig](#visionconfig)
  - [ModelConfig](#modelconfig)
  - [RedisConfig](#redisconfig)
  - [AnnotationConfig](#annotationconfig)
- [Data Formats](#data-formats)
  - [Detection Results](#detection-results-format)
  - [Metadata Format](#metadata-format)
- [Exceptions](#exceptions)
- [Utility Functions](#utility-functions)

---

## UML Class Diagram

![UML Class Diagram](vision_uml_diagram.png)

---

## Core Classes

### VisualCortex

Main orchestration class for vision processing. Coordinates all detection, tracking, and segmentation operations.

#### Initialization

```python
from vision_detect_segment import VisualCortex, get_default_config

cortex = VisualCortex(
    objdetect_model_id: str,                    # Model identifier
    device: str = "auto",                        # Computation device
    stream_name: str = "robot_camera",          # Input Redis stream
    annotated_stream_name: str = "annotated_camera",  # Output Redis stream
    publish_annotated: bool = True,             # Publish annotated frames
    verbose: bool = False,                      # Enable verbose logging
    config: Optional[VisionConfig] = None       # Custom configuration
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `objdetect_model_id` | str | required | Model identifier: "owlv2", "yolo-world", "yoloe-11s/m/l", "grounding_dino" |
| `device` | str | "auto" | Computation device: "auto", "cuda", or "cpu" |
| `stream_name` | str | "robot_camera" | Redis stream name for input images |
| `annotated_stream_name` | str | "annotated_camera" | Redis stream name for annotated frames |
| `publish_annotated` | bool | True | Whether to publish annotated frames to Redis |
| `verbose` | bool | False | Enable detailed logging |
| `config` | VisionConfig | None | Custom configuration (uses default if None) |

#### Methods

##### Detection Methods

```python
# Trigger detection from Redis stream
success: bool = cortex.detect_objects_from_redis()
```

Returns `True` if detection was successful, `False` otherwise.

---

##### Data Retrieval Methods

```python
# Get detected objects (returns copy)
objects: List[Dict] = cortex.get_detected_objects()

# Get annotated image with bounding boxes and labels
annotated: np.ndarray = cortex.get_annotated_image()

# Get current raw image
image: np.ndarray = cortex.get_current_image(resize: bool = True)

# Get detectable object labels
labels: List[List[str]] = cortex.get_object_labels()

# Get processing statistics
stats: Dict = cortex.get_stats()
```

---

##### Configuration Methods

```python
# Add new detectable object type
cortex.add_detectable_object(object_name: str)

# Enable/disable annotated frame publishing
cortex.enable_annotated_publishing(enable: bool = True)

# Get annotated stream name
stream_name: str = cortex.get_annotated_stream_name()

# Check if annotated publishing is enabled
enabled: bool = cortex.is_annotated_publishing_enabled()
```

---

##### Utility Methods

```python
# Get number of processed frames
count: int = cortex.get_processed_frames_count()

# Get computation device
device: str = cortex.get_device()

# Clear GPU cache
cortex.clear_cache()

# Get memory usage information
memory: Dict = cortex.get_memory_usage()

# Cleanup resources (stop background threads)
cortex.cleanup()
```

---

#### Example Usage

```python
from vision_detect_segment import VisualCortex, get_default_config
from redis_robot_comm import RedisImageStreamer
import cv2

# Initialize
config = get_default_config("yoloe-11l")
cortex = VisualCortex("yoloe-11l", device="cuda", config=config)

# Publish image
streamer = RedisImageStreamer()
image = cv2.imread("workspace.jpg")
streamer.publish_image(image)

# Detect objects
if cortex.detect_objects_from_redis():
    objects = cortex.get_detected_objects()
    annotated = cortex.get_annotated_image()

    # Display results
    print(f"Found {len(objects)} objects")
    cv2.imshow("Detection", annotated)
    cv2.waitKey(0)

# Cleanup
cortex.cleanup()
```

---

### ObjectDetector

Multi-model object detection engine supporting YOLOE, YOLO-World, OWL-V2, and Grounding-DINO.

#### Initialization

```python
from vision_detect_segment.core import ObjectDetector
from vision_detect_segment.utils import VisionConfig

detector = ObjectDetector(
    device: str,                              # "cuda" or "cpu"
    model_id: str,                            # Model identifier
    object_labels: List[List[str]],          # Nested list of labels
    redis_host: str = "localhost",           # Redis server host
    redis_port: int = 6379,                  # Redis server port
    stream_name: str = "detected_objects",   # Output stream name
    verbose: bool = False,                   # Enable verbose logging
    config: Optional[VisionConfig] = None,   # Custom configuration
    enable_tracking: bool = True             # Enable object tracking
)
```

#### Methods

```python
# Run detection on RGB image
objects: List[Dict] = detector.detect_objects(
    image: np.ndarray,                        # RGB image
    confidence_threshold: Optional[float] = None  # Override default threshold
)

# Add new detectable label
detector.add_label(label: str)

# Get current detections (supervision format)
detections: sv.Detections = detector.get_detections()

# Get detection labels with track IDs
labels: np.ndarray = detector.get_label_texts()

# Get object labels list
labels: List[List[str]] = detector.get_object_labels()

# Get device and model info
device: str = detector.get_device()
model_id: str = detector.get_model_id()
```

#### Supported Models

| Model ID | Description | Speed | Segmentation |
|----------|-------------|-------|--------------|
| `yoloe-11s/m/l` | YOLO11-based open-vocabulary | ⚡⚡⚡ Fast | Built-in ✅ |
| `yoloe-v8s/m/l` | YOLOv8-based open-vocabulary | ⚡⚡⚡ Fast | Built-in ✅ |
| `yoloe-*-pf` | Prompt-free variants (1200+ classes) | ⚡⚡⚡ Fast | Built-in ✅ |
| `yolo-world` | Real-time open-vocabulary detection | ⚡⚡⚡ Fast | External |
| `owlv2` | Open-vocabulary detection | ⚡⚡ Medium | External |
| `grounding_dino` | Text-guided detection | ⚡ Slow | External |

---

### ObjectSegmenter

Instance segmentation using SAM2, FastSAM, or YOLOE built-in segmentation.

#### Initialization

```python
from vision_detect_segment.core import ObjectSegmenter

segmenter = ObjectSegmenter(
    segmentation_model: Optional[str] = "facebook/sam2.1-hiera-tiny",
    device: str = "cuda",
    verbose: bool = False,
    config: Optional[VisionConfig] = None,
    edgetam_config_path: Optional[str] = None,     # For EdgeTAM
    edgetam_weights_path: Optional[str] = None      # For EdgeTAM
)
```

#### Methods

```python
# Segment all detected objects
detections: sv.Detections = segmenter.segment_objects(
    image: np.ndarray,
    detections: sv.Detections
)

# Segment single bounding box
mask_8u, mask_binary = segmenter.segment_box_in_image(
    box: torch.Tensor,      # [x_min, y_min, x_max, y_max]
    img_work: np.ndarray    # Input image
)

# Check availability
available: bool = segmenter.is_available()

# Get model info
segmenter_obj = segmenter.get_segmenter()
model_id: str = segmenter.get_model_id()
device: str = segmenter.get_device()
```

#### Segmentation Models

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| **YOLOE (built-in)** | ⚡⚡⚡ Fast (~6-10ms) | ⭐⭐⭐ Excellent | Real-time unified detection & segmentation |
| **FastSAM** | ⚡⚡ Medium (~50-100ms) | ⭐⭐ Good | Fast external segmentation |
| **SAM2** | ⚡ Slow (~200-500ms) | ⭐⭐⭐ Excellent | High-quality masks |

---

### ObjectTracker

Persistent object tracking with progressive label stabilization.

#### Initialization

```python
from vision_detect_segment.core import ObjectTracker

tracker = ObjectTracker(
    model: Any,                              # Detection model
    model_id: str,                           # Model identifier
    enable_tracking: bool = False,          # Enable tracking
    verbose: bool = False,                  # Verbose logging
    stabilization_frames: int = 10,         # Frames before label lock
    min_frames_for_display: int = 1         # Min frames before showing label
)
```

#### Methods

##### Label Stabilization

```python
# Update label history and get stabilized labels
stabilized_labels: List[str] = tracker.update_label_history(
    track_ids: np.ndarray,
    labels: List[str]
)

# Detect lost tracks
lost_ids: List[int] = tracker.detect_lost_tracks(
    current_track_ids: np.ndarray
)

# Cleanup lost tracks
tracker.cleanup_lost_tracks(lost_track_ids: List[int])
```

##### Tracking Information

```python
# Get info for specific track
info: Dict = tracker.get_track_info(track_id: int)
# Returns: {
#     "track_id": int,
#     "frame_count": int,
#     "label_history": List[str],
#     "label_distribution": Dict[str, int],
#     "current_majority": str,
#     "stabilized_label": Optional[str],
#     "is_stabilized": bool
# }

# Get stats for all tracks
all_stats: Dict[int, Dict] = tracker.get_all_track_stats()
```

##### Tracking Operations

```python
# YOLO tracking (for YOLO/YOLOE models)
results = tracker.track(
    image: np.ndarray,
    threshold: float = 0.25,
    max_det: int = 50
)

# ByteTrack update (for transformer models)
tracked_detections: sv.Detections = tracker.update_with_detections(
    detections: sv.Detections
)

# Reset tracking state
tracker.reset()
```

---

## Configuration

### VisionConfig

Main configuration class for the vision system.

```python
from vision_detect_segment import VisionConfig

config = VisionConfig()

# Sub-configurations
config.model: ModelConfig           # Model settings
config.redis: RedisConfig          # Redis settings
config.annotation: AnnotationConfig  # Annotation settings

# General settings
config.verbose: bool = False
config.enable_segmentation: bool = True
```

#### Label Management

```python
# Get labels (returns List[List[str]])
labels = config.get_object_labels()

# Set custom labels
config.set_object_labels(["red cube", "blue sphere"])

# Add single label
config.add_object_label("green cylinder")
```

---

### ModelConfig

Configuration for detection models.

```python
from vision_detect_segment.utils import ModelConfig

model_config = ModelConfig(
    name: str,                              # Model name
    confidence_threshold: float = 0.3,     # Detection threshold
    max_detections: int = 20,              # Maximum detections per image
    device_preference: str = "auto",       # Device preference
    model_params: Dict[str, Any] = {}      # Model-specific parameters
)

# Get actual device based on availability
device: str = model_config.get_device()
```

---

### RedisConfig

Configuration for Redis connections.

```python
from vision_detect_segment.utils import RedisConfig

redis_config = RedisConfig(
    host: str = "localhost",
    port: int = 6379,
    stream_name: str = "robot_camera",
    detection_stream: str = "detected_objects",
    connection_timeout: int = 5,
    retry_attempts: int = 3
)
```

---

### AnnotationConfig

Configuration for image annotation.

```python
from vision_detect_segment.utils import AnnotationConfig

annotation_config = AnnotationConfig(
    text_scale: float = 0.5,
    text_padding: int = 3,
    box_thickness: int = 2,
    resize_scale_factor: float = 2.0,
    show_confidence: bool = True,
    show_labels: bool = True
)
```

---

### Configuration Helper Functions

```python
from vision_detect_segment import get_default_config, create_test_config

# Get default config for specific model
config = get_default_config(model_name: str = "owlv2")

# Create test config with reduced labels
test_config = create_test_config()
```

---

## Data Formats

### Detection Results Format

Each detected object is represented as a dictionary:

```python
detected_object = {
    "label": str,              # Object class name
    "confidence": float,       # Detection confidence (0.0-1.0)
    "bbox": {                  # Bounding box coordinates
        "x_min": int,          # Left coordinate
        "y_min": int,          # Top coordinate
        "x_max": int,          # Right coordinate
        "y_max": int           # Bottom coordinate
    },
    "track_id": int,           # Optional: Persistent tracking ID
    "has_mask": bool,          # Whether segmentation mask available
    "mask_data": str,          # Optional: Base64-encoded mask
    "mask_shape": List[int],   # Optional: [height, width]
    "mask_dtype": str          # Optional: Mask data type (e.g., "uint8")
}
```

#### Example

```python
{
    "label": "red cube",
    "confidence": 0.95,
    "bbox": {
        "x_min": 100,
        "y_min": 150,
        "x_max": 200,
        "y_max": 250
    },
    "track_id": 1,
    "has_mask": True,
    "mask_data": "iVBORw0KGgoAAAANSUhEUgAA...",
    "mask_shape": [100, 100],
    "mask_dtype": "uint8"
}
```

---

### Metadata Format

Metadata attached to images and detection results:

#### Image Metadata

```python
image_metadata = {
    "robot": str,              # Robot identifier
    "workspace": str,          # Workspace name
    "workspace_id": str,       # Workspace ID
    "robot_pose": {            # Robot pose information
        "x": float,
        "y": float,
        "z": float,
        "roll": float,
        "pitch": float,
        "yaw": float
    },
    "frame_id": int,           # Frame number
    "timestamp": float         # Unix timestamp
}
```

#### Detection Metadata

```python
detection_metadata = {
    "timestamp": float,        # Unix timestamp
    "object_count": int,       # Number of detections
    "detection_method": str,   # Model used
    "model_id": str           # Model identifier
}
```

---

## Exceptions

All exceptions inherit from `VisionDetectionError`.

### Exception Hierarchy

```python
VisionDetectionError (base)
├── ModelLoadError           # Model loading failures
├── DetectionError          # Detection inference failures
├── SegmentationError       # Segmentation failures
├── RedisConnectionError    # Redis connection/operation failures
├── ImageProcessingError    # Image processing failures
├── ConfigurationError      # Invalid configuration
├── DependencyError         # Missing dependencies
└── AnnotationError         # Annotation failures
```

### Using Exceptions

```python
from vision_detect_segment.utils.exceptions import (
    VisionDetectionError,
    ModelLoadError,
    DetectionError,
    RedisConnectionError
)

try:
    cortex = VisualCortex("owlv2")
    cortex.detect_objects_from_redis()

except ModelLoadError as e:
    print(f"Model loading failed: {e}")
    print(f"Model: {e.model_name}, Reason: {e.reason}")

except DetectionError as e:
    print(f"Detection failed: {e}")
    print(f"Image shape: {e.image_shape}")

except RedisConnectionError as e:
    print(f"Redis error: {e}")
    print(f"Operation: {e.operation}, Host: {e.host}:{e.port}")

except VisionDetectionError as e:
    print(f"Vision error: {e}")
    print(f"Details: {e.details}")
```

---

## Utility Functions

### Image Processing

```python
from vision_detect_segment.utils import (
    validate_image,
    resize_image,
    create_test_image,
    load_image_safe
)

# Validate image
validate_image(image: np.ndarray, min_size: Tuple[int, int] = (32, 32))

# Resize image
resized, scale_x, scale_y = resize_image(
    image: np.ndarray,
    scale_factor: float = 2.0,
    max_size: Optional[Tuple[int, int]] = None
)

# Create test image
test_img = create_test_image(
    shapes: Optional[List[str]] = None,
    size: Tuple[int, int] = (480, 640)
)

# Load image safely with fallback
image = load_image_safe(
    image_path: Union[str, Path],
    fallback_image: Optional[np.ndarray] = None
)
```

---

### Device and Dependencies

```python
from vision_detect_segment.utils import (
    get_optimal_device,
    check_dependencies,
    validate_model_requirements
)

# Get optimal device
device = get_optimal_device(prefer_gpu: bool = True)

# Check dependencies
availability = check_dependencies(requirements: List[str])

# Validate model requirements
validate_model_requirements(model_name: str)
```

---

### Validation

```python
from vision_detect_segment.utils import (
    validate_bbox,
    validate_confidence_threshold,
    convert_bbox_format
)

# Validate bounding box
validate_bbox(bbox: Dict[str, int], image_shape: Tuple[int, ...])

# Validate confidence threshold
validate_confidence_threshold(threshold: float)

# Convert bbox format
bbox_dict = convert_bbox_format(
    bbox: Union[Dict, List, Tuple],
    from_format: str,  # "dict", "list", or "tuple"
    to_format: str
)
```

---

### Performance

```python
from vision_detect_segment.utils import (
    Timer,
    get_memory_usage,
    clear_gpu_cache,
    format_detection_results
)

# Time operations
with Timer("Detection", logger) as t:
    results = detector.detect_objects(image)
print(f"Elapsed: {t.elapsed():.3f}s")

# Get memory usage
memory = get_memory_usage()
print(f"Memory: {memory['rss_mb']:.1f} MB")

# Clear GPU cache
clear_gpu_cache()

# Format detection results
summary = format_detection_results(detections, max_items=10)
print(summary)
```

---

### Logging

```python
from vision_detect_segment.utils import setup_logging

# Setup logger
logger = setup_logging(
    verbose: bool = False,
    log_file: Optional[str] = None
)

# Use logger
logger.info("Processing started")
logger.debug("Detailed debug info")
logger.warning("Warning message")
logger.error("Error occurred")
```

---

## Complete Example

```python
from vision_detect_segment import VisualCortex, get_default_config
from redis_robot_comm import RedisImageStreamer
import cv2

# 1. Initialize with configuration
config = get_default_config("yoloe-11l")
config.model.confidence_threshold = 0.3
config.enable_segmentation = True
config.verbose = True

cortex = VisualCortex(
    objdetect_model_id="yoloe-11l",
    device="cuda",
    config=config,
    publish_annotated=True
)

# 2. Setup image streaming
streamer = RedisImageStreamer(stream_name="robot_camera")

# 3. Process image
image = cv2.imread("workspace.jpg")
metadata = {
    "robot": "arm1",
    "workspace": "station_A",
    "timestamp": time.time()
}
streamer.publish_image(image, metadata=metadata)

# 4. Run detection
if cortex.detect_objects_from_redis():
    # Get results
    objects = cortex.get_detected_objects()
    annotated = cortex.get_annotated_image()

    # Process results
    for obj in objects:
        print(f"Found {obj['label']} at {obj['bbox']}")
        if 'track_id' in obj:
            print(f"  Track ID: {obj['track_id']}")
        if obj['has_mask']:
            print(f"  Has segmentation mask")

    # Display
    cv2.imshow("Detection", annotated)
    cv2.waitKey(0)

    # Get statistics
    stats = cortex.get_stats()
    print(f"Processed {stats['processed_frames']} frames")

# 5. Cleanup
cortex.cleanup()
cv2.destroyAllWindows()
```

---

## See Also

- [Main README](../README.md) - Package overview and installation
- [Workflow Documentation](vision_workflow_doc.md) - Complete workflow details
- [Examples](../examples/) - Code examples and tutorials

---

For questions or issues, please see the [main repository](https://github.com/dgaida/vision_detect_segment).
