# vision_detect_segment

A flexible Python package for real-time object detection and segmentation with Redis-based streaming support. Designed for robotics applications, this package provides an easy-to-use interface for detecting and segmenting objects in images using state-of-the-art vision models.

## Badges

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Quality](https://github.com/dgaida/vision_detect_segment/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/vision_detect_segment/actions/workflows/lint.yml)
[![Tests](https://github.com/dgaida/vision_detect_segment/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/vision_detect_segment/actions/workflows/tests.yml)
[![CodeQL](https://github.com/dgaida/vision_detect_segment/actions/workflows/codeql.yml/badge.svg)](https://github.com/dgaida/vision_detect_segment/actions/workflows/codeql.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Features

- **Multiple Detection Backends**: Support for OWL-V2, YOLO-World, Grounding-DINO, and **YOLOE**
- **YOLOE Integration**: Unified detection and segmentation with open-vocabulary capabilities
- **Optional Segmentation**: Integrated support for SAM2, FastSAM, and YOLOE's built-in segmentation
- **Redis Streaming**: Built-in Redis support for real-time image streaming and detection results
- **Flexible Configuration**: Easy-to-use configuration system with sensible defaults
- **GPU Support**: Automatic GPU detection and utilization when available
- **Robust Error Handling**: Comprehensive exception handling with detailed error messages
- **Performance Monitoring**: Built-in timing and memory usage tracking

## What's New: YOLOE Support

YOLOE (Real-Time Seeing Anything) is the latest addition to our supported models, offering:

- **Unified Detection & Segmentation**: Single model for both tasks
- **Open-Vocabulary Detection**: Detect custom object classes without retraining
- **Multiple Prompting Modes**:
  - Text prompts for flexible class definition
  - Visual prompts for one-shot detection
  - Prompt-free mode with 1200+ built-in classes
- **Real-Time Performance**: Comparable speed to YOLO11 (~130 FPS on GPU)
- **Superior Accuracy**: +3.5 AP over YOLO-Worldv2 on LVIS benchmark

## Supported Models

### Object Detection Models

| Model | Description | Best For | Speed | Segmentation |
|-------|-------------|----------|-------|--------------|
| **yoloe-11s/m/l** | Open-vocabulary detection & segmentation | Real-time unified tasks | Fast | Built-in ✅ |
| **yoloe-v8s/m/l** | YOLOE based on YOLOv8 | Balanced performance | Fast | Built-in ✅ |
| **yoloe-*-pf** | Prompt-free variants | Large vocabulary (1200+ classes) | Fast | Built-in ✅ |
| **owlv2** | Open-vocabulary detection | Custom object classes | Medium | External |
| **yolo-world** | Real-time detection | Speed-critical applications | Fast | External |
| **grounding_dino** | Text-guided detection | Complex queries | Slow | External |

### Segmentation Models

| Model | Description | Requirements |
|-------|-------------|--------------|
| **YOLOE (Built-in)** | Integrated segmentation | Ultralytics >=8.3.0 |
| **SAM2** | High-quality segmentation | `pip install segment-anything-2` |
| **FastSAM** | Fast segmentation | Included with ultralytics |

## Installation

### Basic Installation

```bash
git clone https://github.com/dgaida/vision_detect_segment.git
cd vision_detect_segment
pip install -e .
```

### For YOLOE Support

```bash
# Upgrade ultralytics to latest version
pip install -U ultralytics>=8.3.0
```

### Redis Server

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

### Using YOLOE for Detection and Segmentation

```python
from vision_detect_segment import VisualCortex
from vision_detect_segment.config import get_default_config
from redis_robot_comm import RedisImageStreamer
import cv2

# Initialize with YOLOE (built-in segmentation)
config = get_default_config("yoloe-11l")
cortex = VisualCortex("yoloe-11l", device="auto", config=config)

# Load an image
image = cv2.imread("example.jpg")

# Publish image to Redis
streamer = RedisImageStreamer(stream_name="robot_camera")
streamer.publish_image(image, metadata={"source": "camera1"})

# Detect objects (includes segmentation automatically)
success = cortex.detect_objects_from_redis()
if success:
    detected_objects = cortex.get_detected_objects()
    annotated_image = cortex.get_annotated_image()

    # Check if segmentation masks are available
    for obj in detected_objects:
        print(f"Object: {obj['label']}, Has mask: {obj['has_mask']}")

    cv2.imshow("Detections", annotated_image)
    cv2.waitKey(0)
```

### YOLOE with Custom Text Prompts

```python
from vision_detect_segment import VisualCortex
from vision_detect_segment.config import VisionConfig

# Create custom configuration
config = VisionConfig()
config.set_object_labels([
    "red cube", "blue sphere", "green cylinder",
    "robot gripper", "safety cone"
])

# Initialize YOLOE with custom prompts
cortex = VisualCortex("yoloe-11m", config=config)

# Detection will now only look for specified objects
# And provide segmentation masks automatically
```

### Using YOLOE Prompt-Free Mode

```python
# Prompt-free variant uses internal vocabulary (1200+ classes)
config = get_default_config("yoloe-11l-pf")
cortex = VisualCortex("yoloe-11l-pf", device="auto", config=config)

# No need to set classes - detects from large internal vocabulary
success = cortex.detect_objects_from_redis()
```

## YOLOE Model Variants

### Prompted Models (Text/Visual Prompts)

```python
# Small model (fastest)
cortex = VisualCortex("yoloe-11s", device="auto")

# Medium model (balanced)
cortex = VisualCortex("yoloe-11m", device="auto")

# Large model (most accurate)
cortex = VisualCortex("yoloe-11l", device="auto")

# YOLOv8-based variants
cortex = VisualCortex("yoloe-v8s", device="auto")
cortex = VisualCortex("yoloe-v8m", device="auto")
cortex = VisualCortex("yoloe-v8l", device="auto")
```

### Prompt-Free Models (Internal Vocabulary)

```python
# Use when you don't need custom classes
# Automatically detects from 1200+ predefined categories
cortex = VisualCortex("yoloe-11s-pf", device="auto")
cortex = VisualCortex("yoloe-11m-pf", device="auto")
cortex = VisualCortex("yoloe-11l-pf", device="auto")
```

## Configuration

### YOLOE-Specific Configuration

```python
from vision_detect_segment.config import VisionConfig, get_default_config

# Get default YOLOE configuration
config = get_default_config("yoloe-11l")

# Configure detection parameters
config.model.confidence_threshold = 0.25
config.model.max_detections = 30

# Segmentation is built-in for YOLOE (always enabled)
# No need to configure external segmenter
config.enable_segmentation = True  # This uses YOLOE's built-in segmentation

# Set custom object labels (for prompted models)
config.set_object_labels([
    "workpiece", "tool", "safety equipment", "obstacle"
])
```

## Performance Comparison

### YOLOE vs Other Models

| Model | mAP50-95 (COCO) | Speed (T4 GPU) | Parameters | Segmentation |
|-------|-----------------|----------------|------------|--------------|
| YOLOE-L | 52.6% | 6.2 ms (161 FPS) | 26.2M | Built-in ✅ |
| YOLOv8-L | 52.9% | 9.06 ms (110 FPS) | 43.7M | External |
| YOLO11-L | 53.5% | 6.2 ms (161 FPS) | 26.2M | External |
| YOLO-World | 35.4% (LVIS) | ~8 ms | ~40M | External |
| OWL-V2 | ~30% (LVIS) | 100-200 ms | ~300M | External |

% TODO: reference for those numbers?

**Key Advantages of YOLOE:**
- Unified detection + segmentation in one model
- Open-vocabulary capabilities without performance penalty
- Comparable speed to fastest YOLO models
- Built-in segmentation eliminates need for separate SAM/FastSAM

## API Reference

### YOLOE-Specific Features

```python
# Initialize YOLOE
cortex = VisualCortex(
    objdetect_model_id="yoloe-11l",  # or any YOLOE variant
    device="auto",
    config=config
)

# Detect with automatic segmentation
detected_objects = cortex.get_detected_objects()

# Each detected object includes:
# - label: Object class name
# - confidence: Detection confidence (0-1)
# - bbox: Bounding box coordinates
# - has_mask: True (YOLOE always provides masks)
# - mask_data: Base64-encoded segmentation mask
# - mask_shape: Mask dimensions [height, width]
# - track_id: Optional tracking ID

# Check segmentation availability
for obj in detected_objects:
    if obj['has_mask']:
        print(f"{obj['label']}: mask shape {obj['mask_shape']}")
```

## Use Cases for YOLOE

### 1. Real-Time Robotics
```python
# Perfect for robot manipulation tasks requiring both
# detection and precise segmentation at high speed
config = get_default_config("yoloe-11m")
config.set_object_labels(["workpiece", "tool", "gripper"])
cortex = VisualCortex("yoloe-11m", config=config)
```

### 2. Open-World Detection
```python
# Detect objects not seen during training
config = get_default_config("yoloe-11l-pf")
cortex = VisualCortex("yoloe-11l-pf", config=config)
# Automatically detects from 1200+ categories
```

### 3. Quality Inspection
```python
# Fast detection and segmentation of defects
config = get_default_config("yoloe-11s")
config.set_object_labels(["scratch", "dent", "crack", "discoloration"])
cortex = VisualCortex("yoloe-11s", config=config)
```

### 4. Warehouse Automation
```python
# Track and segment various package types
config = get_default_config("yoloe-v8l")
config.set_object_labels(["box", "pallet", "container", "forklift"])
cortex = VisualCortex("yoloe-v8l", config=config)
```

## Troubleshooting

### YOLOE-Specific Issues

**Model Not Found:**
```bash
# YOLOE models are downloaded automatically
# Ensure you have internet connection and sufficient disk space
pip install -U ultralytics>=8.3.0
```

**CUDA Out of Memory:**
```python
# Use smaller YOLOE variant
cortex = VisualCortex("yoloe-11s", device="cuda")  # Instead of yoloe-11l

# Or use CPU
cortex = VisualCortex("yoloe-11m", device="cpu")
```

**Segmentation Quality Issues:**
```python
# YOLOE segmentation quality is tied to detection confidence
# Lower threshold may capture more objects but less accurate masks
config.model.confidence_threshold = 0.3  # Adjust as needed
```

## Migration Guide

### From SAM2/FastSAM to YOLOE

If you're currently using external segmentation models, YOLOE offers a simpler approach:

**Before (with external segmentation):**
```python
config = get_default_config("yolo-world")
config.enable_segmentation = True  # Uses SAM2 or FastSAM
cortex = VisualCortex("yolo-world", config=config)
```

**After (with YOLOE):**
```python
config = get_default_config("yoloe-11l")
# Segmentation is built-in, no external model needed
cortex = VisualCortex("yoloe-11l", config=config)
```

Benefits:
- Faster inference (single model vs. two separate models)
- Better mask-object alignment
- Simpler configuration

## Contributing

Contributions are welcome! When adding new model support:
1. Add model configuration to `config.py`
2. Implement loading method in `object_detector.py`
3. Add detection/segmentation logic
4. Update documentation and tests

## License

MIT License - see LICENSE file for details

## Citation

```bibtex
@software{vision_detect_segment,
  author = {Gaida, Daniel},
  title = {vision_detect_segment: Object Detection and Segmentation for Robotics},
  year = {2025},
  url = {https://github.com/dgaida/vision_detect_segment}
}

@misc{wang2025yoloerealtimeseeing,
  title={YOLOE: Real-Time Seeing Anything},
  author={Ao Wang and Lihao Liu and Hui Chen and Zijia Lin and Jungong Han and Guiguang Ding},
  year={2025},
  eprint={2503.07465},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## Acknowledgments

This package builds upon:
- [Supervision](https://github.com/roboflow/supervision) for annotations
- [Transformers](https://github.com/huggingface/transformers) for OWL-V2 and Grounding-DINO
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO-World, YOLOE, and FastSAM
- [SAM2](https://github.com/facebookresearch/segment-anything-2) for segmentation
- [YOLOE](https://github.com/THU-MIG/yoloe) for open-vocabulary detection and segmentation

## Contact

Daniel Gaida - daniel.gaida@th-koeln.de

Project Link: https://github.com/dgaida/vision_detect_segment
