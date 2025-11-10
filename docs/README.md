# Vision Detection Workflow Documentation

This directory contains detailed documentation about the `vision_detect_segment` package architecture and workflow.

## Overview

The `vision_detect_segment` package provides a complete pipeline for real-time object detection, tracking, and segmentation in robotics applications. The system uses Redis Streams as a communication backbone to enable asynchronous processing of image data.

## Workflow Diagram

![Vision Detection Workflow](workflow_simple.png)

The workflow consists of the following main steps:

1. **Image Publishing** - An image source (camera, file, or simulation) captures images and publishes them to a Redis Stream using `RedisImageStreamer`. Images can be compressed as JPEG and include metadata like robot pose and workspace information.

2. **Image Retrieval** - The `VisualCortex` component monitors the Redis stream and retrieves the latest images for processing.

3. **Object Detection** - The `ObjectDetector` performs detection using one of three supported model backends:
   - **OWL-V2** - Open-vocabulary detection for custom object classes
   - **YOLO-World** - Fast real-time detection
   - **Grounding-DINO** - Text-guided detection for complex queries

4. **Object Tracking (Optional)** - The `ObjectTracker` maintains persistent object IDs across frames, enabling trajectory analysis and object persistence through occlusions.

5. **Instance Segmentation (Optional)** - The `ObjectSegmenter` generates pixel-level masks for detected objects using SAM2 or FastSAM models.

6. **Results Publishing** - Detection results (bounding boxes, labels, confidence scores, track IDs, and optional masks) are published back to Redis for downstream consumers.

7. **Visualization** - Annotated images with bounding boxes, labels, and segmentation masks are generated for debugging and monitoring.

8. **Downstream Processing** - Other components can consume the detection results from Redis for robot control, decision-making, or data logging.

## Key Components

- **VisualCortex** - Main orchestrator that coordinates all processing steps
- **ObjectDetector** - Multi-model detection engine with tracking support
- **ObjectTracker** - Persistent object tracking using YOLO built-in or ByteTrack
- **ObjectSegmenter** - Instance segmentation using SAM2 or FastSAM
- **RedisImageStreamer** - Image publishing and retrieval (from `redis_robot_comm`)
- **RedisMessageBroker** - Detection results publishing (from `redis_robot_comm`)

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
from vision_detect_segment import VisualCortex
from vision_detect_segment.config import create_test_config
from redis_robot_comm import RedisImageStreamer
import cv2

# Initialize components
streamer = RedisImageStreamer(stream_name="robot_camera")
config = create_test_config()
cortex = VisualCortex("owlv2", device="auto", config=config)

# Publish image
image = cv2.imread("workspace.jpg")
streamer.publish_image(image, metadata={"robot": "arm1"})

# Detect objects
if cortex.detect_objects_from_redis():
    objects = cortex.get_detected_objects()
    annotated = cortex.get_annotated_image()
    
    # Display results
    print(f"Found {len(objects)} objects")
    cv2.imshow("Detections", annotated)
    cv2.waitKey(0)
```

## Documentation Files

- **README.md** (this file) - Quick overview and workflow diagram
- **vision_workflow_doc.md** - Complete workflow documentation with detailed examples
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

## Performance

Typical processing times on NVIDIA GPU:

| Model | Detection | Segmentation | Total FPS |
|-------|-----------|--------------|-----------|
| YOLO-World | 20-50ms | 50-100ms (FastSAM) | 10-25 FPS |
| OWL-V2 | 100-200ms | 200-500ms (SAM2) | 1-3 FPS |
| Grounding-DINO | 200-400ms | 200-500ms (SAM2) | 1-2 FPS |

For real-time applications, YOLO-World with FastSAM is recommended. For higher accuracy, use OWL-V2 or Grounding-DINO with SAM2.

## Requirements

- Python 3.8+
- Redis Server (local or remote)
- PyTorch with CUDA support (recommended)
- See `requirements.txt` for complete dependencies

## Related Documentation

- [Main README](../README.md) - Package overview and installation
- [API Documentation](vision_workflow_doc.md#component-details) - Detailed API reference
- [Examples](../examples/) - Code examples and tutorials

---

For questions or issues, please see the [main repository](https://github.com/dgaida/vision_detect_segment).
