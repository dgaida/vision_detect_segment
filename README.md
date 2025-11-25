# vision_detect_segment

A flexible Python package for real-time object detection and segmentation with Redis-based streaming support. Designed for robotics applications, this package provides an easy-to-use interface for detecting and segmenting objects in images using state-of-the-art vision models.

## Badges

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/dgaida/vision_detect_segment/branch/master/graph/badge.svg)](https://codecov.io/gh/dgaida/vision_detect_segment)
[![Code Quality](https://github.com/dgaida/vision_detect_segment/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/vision_detect_segment/actions/workflows/lint.yml)
[![Tests](https://github.com/dgaida/vision_detect_segment/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/vision_detect_segment/actions/workflows/tests.yml)
[![CodeQL](https://github.com/dgaida/vision_detect_segment/actions/workflows/codeql.yml/badge.svg)](https://github.com/dgaida/vision_detect_segment/actions/workflows/codeql.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Features

- **Multiple Detection Backends**: Support for OWL-V2, YOLO-World, and Grounding-DINO
- **Optional Segmentation**: Integrated support for SAM2 and FastSAM segmentation models
- **Redis Streaming**: Built-in Redis support for real-time image streaming and detection results
- **Flexible Configuration**: Easy-to-use configuration system with sensible defaults
- **GPU Support**: Automatic GPU detection and utilization when available
- **Robust Error Handling**: Comprehensive exception handling with detailed error messages
- **Performance Monitoring**: Built-in timing and memory usage tracking

## Installation

### Basic Installation

```bash
git clone https://github.com/dgaida/vision_detect_segment.git
cd vision_detect_segment
pip install -e .
```

### Dependencies

Core dependencies:
```bash
pip install torch torchvision
pip install opencv-python numpy
pip install supervision
pip install redis
```

Model-specific dependencies:

**For OWL-V2 and Grounding-DINO:**
```bash
pip install transformers
```

**For YOLO-World:**
```bash
pip install ultralytics
```

**For SAM2 segmentation (optional):**
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

**For FastSAM segmentation (optional):**
```bash
pip install ultralytics  # Already included if using YOLO-World
```

### Redis Server

A Redis server is required for image streaming functionality:

```bash
# Using Docker (recommended)
docker run -p 6379:6379 redis:alpine

# Or install locally
# Ubuntu/Debian:
sudo apt-get install redis-server

# macOS:
brew install redis
```

## Quick Start

### Basic Usage

```python
from vision_detect_segment import VisualCortex
from vision_detect_segment.config import create_test_config
import cv2

# Initialize with test configuration
config = create_test_config()
cortex = VisualCortex("owlv2", device="auto", config=config)

# Load an image
image = cv2.imread("example.jpg")

# Detect objects
from vision_detect_segment import RedisImageStreamer
streamer = RedisImageStreamer(stream_name="robot_camera")
streamer.publish_image(image, metadata={"source": "camera1"})

# Get detection results
success = cortex.detect_objects_from_redis()
if success:
    detected_objects = cortex.get_detected_objects()
    annotated_image = cortex.get_annotated_image()

    # Display results
    cv2.imshow("Detections", annotated_image)
    cv2.waitKey(0)
```

### Using Custom Object Labels

```python
from vision_detect_segment import VisualCortex
from vision_detect_segment.config import VisionConfig

# Create custom configuration
config = VisionConfig()
config.set_object_labels([
    "red cube", "blue sphere", "green cylinder",
    "robot gripper", "workpiece"
])

cortex = VisualCortex("owlv2", config=config)
```

### Running the Test Script

```bash
# Make sure Redis is running first
docker run -p 6379:6379 redis:alpine

# In another terminal, run the test script
python main.py
```

## Supported Models

### Object Detection Models

| Model | Description | Best For | Speed |
|-------|-------------|----------|-------|
| **owlv2** | Open-vocabulary detection | Custom object classes | Medium |
| **yolo-world** | Real-time detection | Speed-critical applications | Fast |
| **grounding_dino** | Text-guided detection | Complex queries | Slow |

### Segmentation Models

| Model | Description | Requirements |
|-------|-------------|--------------|
| **SAM2** | High-quality segmentation | `pip install segment-anything-2` |
| **FastSAM** | Fast segmentation | Included with ultralytics |

## Configuration

### Using Configuration Objects

```python
from vision_detect_segment.config import VisionConfig, ModelConfig

# Create configuration
config = VisionConfig()

# Configure model settings
config.model.confidence_threshold = 0.25
config.model.max_detections = 30

# Configure Redis
config.redis.host = "localhost"
config.redis.port = 6379

# Configure annotation
config.annotation.show_confidence = True
config.annotation.resize_scale_factor = 2.0

# Enable/disable features
config.enable_segmentation = True
config.verbose = True
```

### Default Configurations

```python
from vision_detect_segment.config import get_default_config, create_test_config

# Get default configuration for a model
config = get_default_config("owlv2")

# Get test configuration (reduced object labels for faster testing)
test_config = create_test_config()
```

## Architecture

### Key Components

- **VisualCortex**: Main interface for vision processing
- **ObjectDetector**: Handles object detection with multiple backend support
- **ObjectSegmenter**: Optional segmentation functionality
- **VisionConfig**: Configuration management
- **Custom Exceptions**: Detailed error handling

## API Reference

### VisualCortex

Main class for vision processing.

```python
cortex = VisualCortex(
    objdetect_model_id: str,          # "owlv2", "yolo-world", or "grounding_dino"
    device: str = "auto",              # "auto", "cuda", or "cpu"
    stream_name: str = "robot_camera", # Redis stream name
    verbose: bool = True,              # Enable detailed logging
    config: Optional[VisionConfig] = None  # Custom configuration
)
```

**Methods:**

- `detect_objects_from_redis()` - Trigger detection from Redis stream
- `get_detected_objects()` - Get list of detected objects
- `get_annotated_image()` - Get annotated visualization
- `get_current_image()` - Get current raw image
- `add_detectable_object(label)` - Add new object label
- `get_stats()` - Get processing statistics
- `clear_cache()` - Clear GPU memory cache

### Detection Results Format

```python
detected_object = {
    "label": str,              # Object class name
    "confidence": float,       # Detection confidence (0-1)
    "bbox": {                  # Bounding box coordinates
        "x_min": int,
        "y_min": int,
        "x_max": int,
        "y_max": int
    },
    "has_mask": bool,          # Whether segmentation mask is available
    "mask_data": str or None   # Base64-encoded mask data (if available)
}
```

## Error Handling

The package uses custom exception types for better error handling:

```python
from vision_detect_segment.exceptions import (
    VisionDetectionError,      # Base exception
    ModelLoadError,            # Model loading failed
    DetectionError,            # Detection failed
    SegmentationError,         # Segmentation failed
    RedisConnectionError,      # Redis connection issues
    ImageProcessingError,      # Image processing failed
    ConfigurationError,        # Configuration invalid
    DependencyError           # Missing dependencies
)

try:
    cortex = VisualCortex("owlv2")
    cortex.detect_objects_from_redis()
except ModelLoadError as e:
    print(f"Model loading failed: {e}")
    print(f"Details: {e.details}")
except DetectionError as e:
    print(f"Detection failed: {e}")
```

## Performance Optimization

### GPU Memory Management

```python
# Clear GPU cache after processing
cortex.clear_cache()

# Monitor memory usage
memory_info = cortex.get_memory_usage()
print(f"Memory: {memory_info['rss_mb']:.1f} MB")
```

### Batch Processing

```python
from vision_detect_segment.utils import Timer

with Timer("Batch processing", logger):
    for image_path in image_paths:
        image = cv2.imread(image_path)
        streamer.publish_image(image)
        cortex.detect_objects_from_redis()
```

### Using Test Configuration

For faster testing with fewer object labels:

```python
from vision_detect_segment.config import create_test_config

config = create_test_config()  # Only 7 labels instead of 50+
cortex = VisualCortex("owlv2", config=config)
```

## Utilities

The package includes various utility functions:

```python
from vision_detect_segment.utils import (
    create_test_image,        # Generate test images
    resize_image,             # Resize with scale factor
    validate_image,           # Validate image format
    format_detection_results, # Format results for display
    Timer,                    # Performance timing
    get_optimal_device,       # Auto-detect best device
    clear_gpu_cache          # Free GPU memory
)
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```python
# Use CPU instead
cortex = VisualCortex("owlv2", device="cpu")

# Or clear cache regularly
cortex.clear_cache()
```

**Redis Connection Failed:**
```bash
# Check if Redis is running
docker ps | grep redis

# Or start Redis
docker run -p 6379:6379 redis:alpine
```

**Model Loading Fails:**
```bash
# Check dependencies
pip install transformers  # For OWL-V2/Grounding-DINO
pip install ultralytics   # For YOLO-World
```

## Development

### Project Structure

```
vision_detect_segment/
├── .github/
│   └── workflows/
│       ├── codeql.yml
│       ├── dependency-review.yml
│       ├── lint.yml
│       └── release.yml
├── examples/
│   └── example.png
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_detector.py
│   ├── test_segmenter.py
│   ├── test_tracker.py
│   ├── test_utils.py
│   └── test_visualcortex.py
├── vision_detect_segment/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── object_detector.py
│   │   ├── object_segmenter.py
│   │   ├── object_tracker.py
│   │   └── visualcortex.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── exceptions.py
│       └── utils.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── main.py                      # Test script
├── README.md
├── pyproject.toml
└── requirements.txt
```

### Running Tests

```bash
python main.py  # Runs comprehensive test suite
```

### Adding New Models

To add a new detection model backend:

1. Add model configuration to `config.py`:
```python
MODEL_CONFIGS["new_model"] = ModelConfig(...)
```

2. Implement loading in `ObjectDetector._load_model()`
3. Implement detection in `ObjectDetector._detect_new_model()`

## Contributing

Contributions are welcome! Please ensure:

- Code follows existing style conventions
- Private variables use `_` prefix
- New features include error handling
- Tests pass successfully

## License

MIT License - see LICENSE file for details

## Citation

If you use this package in your research, please cite:

```bibtex
@software{vision_detect_segment,
  author = {Gaida, Daniel},
  title = {vision_detect_segment: Object Detection and Segmentation for Robotics},
  year = {2025},
  url = {https://github.com/dgaida/vision_detect_segment}
}
```

## Acknowledgments

This package builds upon:
- [Supervision](https://github.com/roboflow/supervision) for annotations
- [Transformers](https://github.com/huggingface/transformers) for OWL-V2 and Grounding-DINO
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO-World and FastSAM
- [SAM2](https://github.com/facebookresearch/segment-anything-2) for segmentation

## Contact

Daniel Gaida - daniel.gaida@th-koeln.de

Project Link: https://github.com/dgaida/vision_detect_segment
