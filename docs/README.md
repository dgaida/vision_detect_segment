# Vision Detection Workflow Documentation

This directory contains detailed documentation about the `vision_detect_segment` package architecture and workflow.

## Overview

The `vision_detect_segment` package provides a complete pipeline for real-time object detection, tracking, and segmentation in robotics applications. The system uses Redis Streams as a communication backbone to enable asynchronous processing of image data.

## Workflow Diagram

![Vision Detection Workflow](workflow_simple.png)

The workflow consists of the following main steps:

1. **Image Publishing** - An image source (camera, file, or simulation) captures images and publishes them to a Redis Stream using `RedisImageStreamer`. Images can be compressed as JPEG and include metadata like robot pose and workspace information.

2. **Image Retrieval** - The `VisualCortex` component monitors the Redis stream and retrieves the latest images for processing.

3. **Object Detection** - The `ObjectDetector` performs detection using multiple supported model backends:
   - **YOLOE (yoloe-11s/m/l, yoloe-v8s/m/l)** - Open-vocabulary detection with built-in segmentation
   - **YOLO-World** - Fast real-time detection
   - **OWL-V2** - Open-vocabulary detection for custom object classes
   - **Grounding-DINO** - Text-guided detection for complex queries

4. **Object Tracking (Optional)** - The `ObjectTracker` maintains persistent object IDs across frames with progressive label stabilization, enabling trajectory analysis and object persistence through occlusions.

5. **Instance Segmentation (Optional)** - The `ObjectSegmenter` generates pixel-level masks for detected objects using:
   - **YOLOE built-in segmentation** - Integrated detection and segmentation in one model
   - **SAM2** - High-quality segmentation for other detection models
   - **FastSAM** - Fast segmentation alternative

6. **Results Publishing** - Detection results (bounding boxes, labels, confidence scores, track IDs, and optional masks) are published back to Redis for downstream consumers.

7. **Visualization** - Annotated images with bounding boxes, labels, and segmentation masks are generated for debugging and monitoring. Can optionally be published to a separate Redis stream.

8. **Downstream Processing** - Other components can consume the detection results from Redis for robot control, decision-making, or data logging.

## Key Components

- **VisualCortex** - Main orchestrator that coordinates all processing steps, manages Redis connections, and publishes annotated frames
- **ObjectDetector** - Multi-model detection engine with support for YOLOE, YOLO-World, OWL-V2, and Grounding-DINO
- **ObjectTracker** - Persistent object tracking with progressive label stabilization using ByteTrack or YOLO built-in tracking
- **ObjectSegmenter** - Instance segmentation using YOLOE built-in, SAM2, or FastSAM
- **RedisImageStreamer** - Image publishing and retrieval (from `redis_robot_comm`)
- **RedisMessageBroker** - Detection results publishing (from `redis_robot_comm`)
- **RedisLabelManager** - Dynamic label management and monitoring (from `redis_robot_comm`)

## Detailed Documentation

For comprehensive information about the workflow, including:
- Detailed step-by-step process descriptions
- Code examples for each component
- Configuration options
- Performance optimization tips
- Troubleshooting guide

Please see: **[vision_workflow_doc.md](vision_workflow_doc.md)**

## Quick Start Example

```python
from vision_detect_segment import VisualCortex, get_default_config
from redis_robot_comm import RedisImageStreamer
import cv2

# Initialize components
streamer = RedisImageStreamer(stream_name="robot_camera")

# Option 1: Using YOLOE for unified detection and segmentation
config = get_default_config("yoloe-11l")
cortex = VisualCortex("yoloe-11l", device="auto", config=config)

# Option 2: Using OWL-V2 for open-vocabulary detection
# config = get_default_config("owlv2")
# cortex = VisualCortex("owlv2", device="auto", config=config)

# Publish image
image = cv2.imread("workspace.jpg")
streamer.publish_image(image, metadata={"robot": "arm1"})

# Detect objects
if cortex.detect_objects_from_redis():
    objects = cortex.get_detected_objects()
    annotated = cortex.get_annotated_image()

    # Display results
    print(f"Found {len(objects)} objects")
    for obj in objects:
        print(f"  - {obj['label']}: {obj['confidence']:.2f}")
        if 'track_id' in obj:
            print(f"    Track ID: {obj['track_id']}")
        if obj.get('has_mask'):
            print(f"    Has segmentation mask")

    cv2.imshow("Detections", annotated)
    cv2.waitKey(0)
```

## Documentation Files

- **README.md** (this file) - Quick overview and workflow diagram
- **vision_workflow_doc.md** - Complete workflow documentation with detailed examples
- **api.md** - API reference documentation
- **workflow_simple.png** - Simplified workflow diagram
- **workflow_detailed.png** - Detailed workflow with all processing steps

## Architecture Benefits

This architecture provides several advantages:

✅ **Decoupling** - Image producers and consumers operate independently  
✅ **Asynchronous** - Non-blocking processing enables real-time operation  
✅ **Scalability** - Multiple detectors can consume from the same stream  
✅ **Flexibility** - Easy to swap detection models or add processing steps  
✅ **Robustness** - Redis provides reliable message delivery and persistence  
✅ **Monitoring** - All data flows through Redis for easy debugging and logging  
✅ **Label Management** - Dynamic object label updates via Redis without restart  
✅ **Unified Segmentation** - YOLOE provides detection + segmentation in one model  
✅ **Progressive Tracking** - Label stabilization prevents flickering across frames

## Performance

Typical processing times on NVIDIA GPU:

| Model | Detection | Segmentation | Total FPS |
|-------|-----------|--------------|-----------|
| YOLOE-L | 6-10ms | Built-in | 100-160 FPS |
| YOLO-World | 20-50ms | 50-100ms (FastSAM) | 10-25 FPS |
| OWL-V2 | 100-200ms | 200-500ms (SAM2) | 1-3 FPS |
| Grounding-DINO | 200-400ms | 200-500ms (SAM2) | 1-2 FPS |

**Recommended Configurations:**
- **Real-time applications (>30 FPS):** YOLOE-11s/m with built-in segmentation
- **Balanced performance (10-25 FPS):** YOLO-World with FastSAM
- **High accuracy (<5 FPS):** OWL-V2 or Grounding-DINO with SAM2
- **Custom classes:** YOLOE or OWL-V2 with text prompts

## Requirements

- Python 3.8+
- Redis Server (local or remote)
- PyTorch with CUDA support (recommended)
- See `requirements.txt` for complete dependencies

### Model-Specific Requirements

- **YOLOE:** `pip install -U ultralytics>=8.3.0`
- **YOLO-World:** `pip install ultralytics`
- **OWL-V2 / Grounding-DINO:** `pip install transformers`
- **SAM2 (optional):** `pip install segment-anything-2`
- **FastSAM:** Included with ultralytics

## Related Documentation

- [Main README](../README.md) - Package overview and installation
- [API Documentation](api.md) - Detailed API reference
- [Workflow Documentation](vision_workflow_doc.md) - Complete workflow details
- [Examples](../examples/) - Code examples and tutorials

## New Features

### YOLOE Integration (v0.2.0+)

YOLOE brings several advantages:
- **Unified Model** - Single model for both detection and segmentation
- **Real-Time Performance** - Comparable speed to YOLO11 (~130 FPS)
- **Open Vocabulary** - Detect custom classes without retraining
- **Built-in Segmentation** - No need for separate SAM/FastSAM model

### Progressive Label Stabilization

The tracking system now includes:
- **Immediate Display** - Labels shown from first frame using majority vote
- **Gradual Stabilization** - Labels stabilize over configurable number of frames (default: 10)
- **Locked Labels** - Once stabilized, labels remain consistent even if detection varies
- **Lost Track Cleanup** - Automatic cleanup of tracks that disappear

### Dynamic Label Management

- Labels can be updated via Redis without restarting the detector
- `RedisLabelManager` enables remote label configuration
- Background monitoring thread detects label changes automatically

### Annotated Frame Publishing

- Optionally publish annotated frames to a separate Redis stream
- Useful for remote monitoring and debugging
- Configurable quality and compression settings

---

For questions or issues, please see the [main repository](https://github.com/dgaida/vision_detect_segment).
