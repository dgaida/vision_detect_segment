# Vision Detection Workflow Documentation

## Overview

This document describes the complete workflow of the `vision_detect_segment` package in conjunction with the `redis_robot_comm` package. The system provides a complete pipeline for real-time object detection, tracking, and segmentation in robotics applications using Redis as a communication backbone.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Workflow Steps](#workflow-steps)
- [Component Details](#component-details)
- [Code Examples](#code-examples)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

The vision detection system follows a producer-consumer architecture with Redis Streams as the communication layer:

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│                 │         │                 │         │                 │
│  Image Source   │────────▶│  Redis Server   │────────▶│  VisualCortex   │
│  (Camera/File)  │         │  (Stream Broker)│         │  (Detector)     │
│                 │         │                 │         │                 │
└─────────────────┘         └─────────────────┘         └─────────────────┘
        │                           ▲                            │
        │                           │                            │
        │                           │                            ▼
        │                           │                    ┌─────────────────┐
        │                           │                    │                 │
        │                           └────────────────────│  Detections     │
        │                                                │  (Results)      │
        │                                                │                 │
        └────────────────────────────────────────────────▶ └─────────────────┘
                    (Metadata: pose, workspace_id, etc.)
```

### Key Components

1. **RedisImageStreamer** (from `redis_robot_comm`)
   - Publishes images to Redis streams
   - Handles variable image sizes
   - Supports JPEG compression
   - Manages stream metadata

2. **VisualCortex** (from `vision_detect_segment`)
   - Main orchestration class
   - Consumes images from Redis
   - Coordinates detection, tracking, and segmentation
   - Publishes results back to Redis

3. **ObjectDetector**
   - Multi-model backend support (OWL-V2, YOLO-World, YOLOE, Grounding-DINO)
   - Real-time object detection
   - Confidence-based filtering

4. **ObjectTracker**
   - Persistent object tracking across frames
   - Progressive label stabilization
   - Track ID assignment
   - YOLO built-in tracking or ByteTrack for transformer models

5. **ObjectSegmenter**
   - Optional instance segmentation
   - SAM2, FastSAM, or YOLOE built-in segmentation
   - Mask generation for detected objects

6. **RedisMessageBroker** (from `redis_robot_comm`)
   - Publishes detection results
   - Stores object metadata
   - Enables downstream consumers

---

## Workflow Steps

### Step 1: Image Capture and Publishing

The workflow begins when an image source (camera, file, or simulation) captures a frame and publishes it to Redis.

```python
from redis_robot_comm import RedisImageStreamer
import cv2
import time

# Initialize streamer
streamer = RedisImageStreamer(
    host="localhost",
    port=6379,
    stream_name="robot_camera"
)

# Capture image
image = cv2.imread("workspace.jpg")  # or cv2.VideoCapture(0).read()

# Prepare metadata
metadata = {
    "robot": "robot_arm_1",
    "workspace": "assembly_station_A",
    "workspace_id": "ws_001",
    "robot_pose": {
        "x": 0.0, "y": 0.0, "z": 0.5,
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0
    },
    "frame_id": 1,
    "timestamp": time.time()
}

# Publish to Redis
stream_id = streamer.publish_image(
    image,
    metadata=metadata,
    compress_jpeg=True,
    quality=85,
    maxlen=5  # Keep only last 5 frames
)

print(f"Published image with ID: {stream_id}")
```

**Redis Stream Entry:**
```
Stream: "robot_camera"
Entry ID: "1699564231000-0"
Fields:
  - timestamp: "1699564231.123456"
  - image_data: "<base64_encoded_jpeg>"
  - format: "jpeg"
  - width: "1920"
  - height: "1080"
  - channels: "3"
  - dtype: "uint8"
  - compressed_size: "245678"
  - original_size: "6220800"
  - metadata: "{...}"
```

---

### Step 2: Image Retrieval by VisualCortex

The `VisualCortex` class monitors the Redis stream and retrieves images for processing.

```python
from vision_detect_segment import VisualCortex
from vision_detect_segment.config import create_test_config

# Initialize VisualCortex
config = create_test_config()
cortex = VisualCortex(
    objdetect_model_id="owlv2",
    device="auto",
    stream_name="robot_camera",
    verbose=True,
    config=config
)

# Manually trigger detection on latest image
success = cortex.detect_objects_from_redis()

if success:
    print("Detection completed successfully")
else:
    print("No image available or detection failed")
```

**Internal Process:**
1. `RedisImageStreamer.get_latest_image()` retrieves newest frame
2. Image is decoded from base64/JPEG
3. Metadata is extracted
4. Image validation is performed
5. Processing callback is triggered

---

### Step 3: Object Detection

Once an image is retrieved, the `ObjectDetector` performs detection using the configured model.

```python
# Inside VisualCortex.process_image_callback()
detected_objects = self._object_detector.detect_objects(
    image,
    confidence_threshold=0.3
)
```

**Detection Process:**

#### For YOLOE Models (Unified Detection & Segmentation):

YOLOE performs both detection and segmentation in a single forward pass, offering real-time performance with built-in instance segmentation.

```python
# YOLOE detection with optional tracking
if self._tracker and self._tracker._use_yolo_tracker:
    results = self._tracker.track(image, threshold)
else:
    results = self._model.predict(image, conf=threshold, max_det=20)

# Extract detections with built-in segmentation
detected_objects = []
boxes = results[0].boxes

for i, box in enumerate(boxes):
    cls = int(boxes.cls[i])
    class_name = results[0].names[cls]
    confidence = float(boxes.conf[i])
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    obj_dict = {
        "label": class_name,
        "confidence": confidence,
        "bbox": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2},
        "has_mask": False
    }

    # Add track ID if available
    if hasattr(results[0].boxes, "id") and results[0].boxes.id is not None:
        track_id = int(results[0].boxes.id[i])
        obj_dict["track_id"] = track_id

    detected_objects.append(obj_dict)

# Extract segmentation masks (YOLOE built-in)
if hasattr(results[0], "masks") and results[0].masks is not None:
    masks_data = results[0].masks.data

    for i, obj in enumerate(detected_objects):
        if i < len(masks_data):
            mask = masks_data[i].cpu().numpy()
            mask_8u = (mask * 255).astype(np.uint8)

            obj["mask_data"] = base64.b64encode(mask_8u.tobytes()).decode('utf-8')
            obj["has_mask"] = True
            obj["mask_shape"] = list(mask_8u.shape)
            obj["mask_dtype"] = str(mask_8u.dtype)
```

#### For Transformer-based Models (OWL-V2, Grounding-DINO):

```python
# Prepare inputs
inputs = processor(
    images=image,
    text=object_labels,
    return_tensors="pt"
).to(device)

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Post-process
results = processor.post_process_object_detection(
    outputs=outputs,
    target_sizes=[(height, width)],
    threshold=0.3
)
```

#### For YOLO-World:

```python
# Run detection/tracking
if tracking_enabled:
    results = model.track(
        image,
        persist=True,
        conf=0.25,
        max_det=20
    )
else:
    results = model.predict(
        image,
        conf=0.25,
        max_det=20
    )
```

**Detection Output:**
```python
detected_objects = [
    {
        "label": "red cube",
        "confidence": 0.95,
        "bbox": {
            "x_min": 100,
            "y_min": 150,
            "x_max": 200,
            "y_max": 250
        },
        "has_mask": False,
        "track_id": 1  # Only if tracking enabled
    },
    {
        "label": "blue circle",
        "confidence": 0.87,
        "bbox": {
            "x_min": 300,
            "y_min": 180,
            "x_max": 380,
            "y_max": 260
        },
        "has_mask": False,
        "track_id": 2
    }
]
```

---

### Step 4: Object Tracking (Optional)

If tracking is enabled, the `ObjectTracker` maintains persistent IDs across frames with progressive label stabilization.

```python
# For YOLO models (YOLO-World, YOLOE) - built-in tracking
if self._use_yolo_tracker and self.enable_tracking:
    results = self.model.track(
        image,
        persist=True,  # Maintain IDs across frames
        stream=False,
        conf=threshold,
        max_det=max_det
    )
    # Extract track IDs from results
    if hasattr(results[0].boxes, "id"):
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

# For transformer models - ByteTrack
if self._tracker and self.enable_tracking:
    detections = sv.Detections(
        xyxy=boxes.cpu().numpy(),
        confidence=scores.cpu().numpy(),
        class_id=class_ids
    )
    tracked_detections = self._tracker.update_with_detections(detections)
    track_ids = tracked_detections.tracker_id
```

**Progressive Label Stabilization:**

The tracker includes an advanced label stabilization system:

1. **Initial Frames (Frame 1+)**: Shows majority vote of label history
2. **Stabilization Phase (Frame 1-10)**: Continuously updates majority label
3. **Locked Phase (Frame 10+)**: Label locked to most common detection

```python
# Example label stabilization over 12 frames
# Track ID: 1
# Frame 1: "cat" → Display: "cat" (100% from 1 detection)
# Frame 2: "cat" → Display: "cat" (100% from 2 detections)
# Frame 3: "dog" → Display: "cat" (67% from 3 detections)
# Frame 4: "cat" → Display: "cat" (75% from 4 detections)
# ...
# Frame 10: "cat" → Display: "cat" (80% from 10 detections) - LOCKED
# Frame 11: "dog" → Display: "cat" (still locked)
# Frame 12: "dog" → Display: "cat" (still locked)
```

**Tracking Benefits:**
- Consistent object identity across frames
- Prevents label flickering between similar classes
- Motion trajectory analysis capability
- Object persistence through brief occlusions
- Improved downstream processing (counting, monitoring)

**Example with Track IDs and Stabilized Labels:**
```python
detected_objects = [
    {
        "label": "red cube",        # Stabilized label (locked after 10 frames)
        "confidence": 0.95,
        "bbox": {...},
        "track_id": 1              # Same ID as previous frames
    },
    {
        "label": "blue circle",     # New object, majority vote from 3 frames
        "confidence": 0.87,
        "bbox": {...},
        "track_id": 2              # Newly appeared object
    }
]
```

---

### Step 5: Instance Segmentation (Optional)

If segmentation is enabled, the system generates pixel-level masks for each detection.

#### Option 1: YOLOE Built-in Segmentation (Recommended)

YOLOE models have integrated segmentation that runs simultaneously with detection:

```python
# Segmentation happens automatically during detection
results = model.predict(image, conf=threshold)

# Extract masks directly from results
if hasattr(results[0], "masks") and results[0].masks is not None:
    masks_data = results[0].masks.data

    for i, obj in enumerate(detected_objects):
        if i < len(masks_data):
            mask = masks_data[i].cpu().numpy()
            mask_8u = (mask * 255).astype(np.uint8)

            obj["mask_data"] = base64.b64encode(mask_8u.tobytes()).decode('utf-8')
            obj["has_mask"] = True
            obj["mask_shape"] = list(mask_8u.shape)
            obj["mask_dtype"] = str(mask_8u.dtype)
```

**YOLOE Segmentation Features:**
- ✅ Unified pipeline (no separate segmentation step)
- ✅ Real-time performance (~100-160 FPS on GPU)
- ✅ High-quality masks comparable to SAM
- ✅ Works with both prompted and prompt-free variants

#### Option 2: External Segmentation (SAM2, FastSAM)

For models without built-in segmentation (OWL-V2, YOLO-World, Grounding-DINO):

```python
# For each detected object
for obj, box in zip(detected_objects, boxes):
    # Generate mask using external segmenter
    mask_8u, mask_binary = segmenter.segment_box_in_image(
        box,
        image
    )

    if mask_8u is not None:
        # Serialize mask for Redis
        obj["mask_data"] = base64.b64encode(mask_8u.tobytes()).decode('utf-8')
        obj["has_mask"] = True
        obj["mask_shape"] = list(mask_8u.shape)
        obj["mask_dtype"] = str(mask_8u.dtype)
```

**Segmentation Models Comparison:**

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| **YOLOE (Built-in)** | ⚡⚡⚡ Fast (~6-10ms) | ⭐⭐⭐ Excellent | Real-time unified detection & segmentation |
| **FastSAM** | ⚡⚡ Medium (~50-100ms) | ⭐⭐ Good | Fast external segmentation |
| **SAM2** | ⚡ Slow (~200-500ms) | ⭐⭐⭐ Excellent | High-quality masks when speed not critical |

**Example Segmented Object:**
```python
{
    "label": "red cube",
    "confidence": 0.95,
    "bbox": {...},
    "track_id": 1,
    "has_mask": True,
    "mask_data": "<base64_encoded_mask>",
    "mask_shape": [100, 100],  # [height, width]
    "mask_dtype": "uint8"
}
```

---

### Step 6: Publishing Results to Redis

After detection, tracking, and optional segmentation, results are published back to Redis.

```python
# Inside ObjectDetector._publish_detections()
metadata = {
    'timestamp': time.time(),
    'object_count': len(detected_objects),
    'detection_method': 'owlv2',
    'model_id': 'owlv2'
}

redis_broker.publish_objects(
    detected_objects,
    metadata
)
```

**Redis Stream Entry:**
```
Stream: "detected_objects"
Entry ID: "1699564231500-0"
Fields:
  - timestamp: "1699564231.234567"
  - objects: "[{...}, {...}]"  # JSON array of detected objects
  - camera_pose: "{...}"
```

---

### Step 7: Annotation and Visualization

The `VisualCortex` creates annotated images for visualization and debugging.

```python
# Get results
annotated_image = cortex.get_annotated_image()
detected_objects = cortex.get_detected_objects()

# Display
cv2.imshow("Detections", annotated_image)
cv2.waitKey(0)

# Print summary
for obj in detected_objects:
    print(f"Found {obj['label']} with confidence {obj['confidence']:.2f}")
    if 'track_id' in obj:
        print(f"  Track ID: {obj['track_id']}")
```

**Annotation Features:**
- Bounding boxes with corner markers
- Labels with confidence scores
- Track IDs (if tracking enabled)
- Segmentation masks with halo effect
- Configurable colors and styles

---

## Component Details

### VisualCortex

The main orchestrator that coordinates all components.

**Key Methods:**
- `detect_objects_from_redis()` - Manually trigger detection
- `get_detected_objects()` - Retrieve detection results
- `get_annotated_image()` - Get visualization
- `get_current_image()` - Get raw input image
- `add_detectable_object(label)` - Add new object type
- `get_stats()` - Get processing statistics

**Configuration:**
```python
from vision_detect_segment.config import VisionConfig

config = VisionConfig()

# Model settings
config.model.confidence_threshold = 0.3
config.model.max_detections = 20

# Redis settings
config.redis.host = "localhost"
config.redis.port = 6379
config.redis.stream_name = "robot_camera"

# Annotation settings
config.annotation.show_confidence = True
config.annotation.resize_scale_factor = 2.0

# Object labels
config.set_object_labels([
    "red cube", "blue sphere", "green cylinder",
    "robot gripper", "workpiece"
])
```

---

### ObjectDetector

Handles multi-model object detection.

**Supported Models:**

| Model | Backend | Speed | Segmentation | Use Case |
|-------|---------|-------|--------------|----------|
| **yoloe-11s/m/l** | Ultralytics | ⚡⚡⚡ Fast | Built-in ✅ | Real-time unified detection & segmentation |
| **yoloe-v8s/m/l** | Ultralytics | ⚡⚡⚡ Fast | Built-in ✅ | YOLOv8-based unified pipeline |
| **yoloe-*-pf** | Ultralytics | ⚡⚡⚡ Fast | Built-in ✅ | Prompt-free with 1200+ classes |
| **yolo-world** | Ultralytics | ⚡⚡⚡ Fast | External | Real-time open-vocabulary detection |
| **owlv2** | Transformers | ⚡⚡ Medium | External | Open-vocabulary with custom classes |
| **grounding_dino** | Transformers | ⚡ Slow | External | Complex text-guided queries |

**Key Methods:**
- `detect_objects(image, threshold)` - Run detection
- `add_label(label)` - Add detectable object
- `get_detections()` - Get supervision detections
- `get_label_texts()` - Get detection labels with track IDs

---

### ObjectTracker

Maintains object identity across frames.

**Tracking Strategies:**

1. **YOLO Built-in Tracking:**
   - Uses Ultralytics' native tracker
   - Optimized for YOLO models
   - `persist=True` maintains IDs

2. **ByteTrack (Transformer Models):**
   - Supervision's ByteTrack implementation
   - Works with OWL-V2 and Grounding-DINO
   - Robust to occlusions

**Configuration:**
```python
cortex = VisualCortex(
    objdetect_model_id="owlv2",
    config=config
)

# Tracking is enabled via ObjectDetector initialization
detector = ObjectDetector(
    device="cuda",
    model_id="owlv2",
    object_labels=labels,
    enable_tracking=True  # Enable tracking
)
```

---

### ObjectSegmenter

Generates instance segmentation masks.

**Key Methods:**
- `segment_objects(image, detections)` - Segment all detections
- `segment_box_in_image(box, image)` - Segment single object
- `is_available()` - Check if segmentation is available

**Mask Format:**
```python
# uint8 mask (0-255)
mask_8u = np.array([[0, 0, 255, 255],
                    [0, 255, 255, 255],
                    [255, 255, 255, 0],
                    [255, 255, 0, 0]], dtype=np.uint8)

# Binary mask (True/False)
mask_binary = mask_8u > 0
```

---

## Code Examples

### Complete End-to-End Example

```python
import cv2
import time
from vision_detect_segment import VisualCortex
from vision_detect_segment.config import create_test_config
from redis_robot_comm import RedisImageStreamer

# 1. Initialize components
streamer = RedisImageStreamer(stream_name="robot_camera")
config = create_test_config()
cortex = VisualCortex(
    objdetect_model_id="owlv2",
    device="auto",
    config=config
)

# 2. Publish image
image = cv2.imread("workspace.jpg")
metadata = {
    "robot": "arm1",
    "workspace": "station_A",
    "timestamp": time.time()
}
streamer.publish_image(image, metadata=metadata)

# 3. Wait for Redis
time.sleep(0.5)

# 4. Detect objects
success = cortex.detect_objects_from_redis()

if success:
    # 5. Get results
    detected_objects = cortex.get_detected_objects()
    annotated_image = cortex.get_annotated_image()

    # 6. Display
    print(f"Found {len(detected_objects)} objects:")
    for obj in detected_objects:
        print(f"  - {obj['label']}: {obj['confidence']:.2f}")
        if 'track_id' in obj:
            print(f"    Track ID: {obj['track_id']}")

    cv2.imshow("Detections", annotated_image)
    cv2.waitKey(0)
```

---

### Using YOLOE for Real-Time Performance

```python
import cv2
from vision_detect_segment import VisualCortex, get_default_config

# Initialize with YOLOE for real-time unified detection & segmentation
config = get_default_config("yoloe-11l")
cortex = VisualCortex(
    objdetect_model_id="yoloe-11l",
    device="cuda",
    config=config
)

# Open video stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Publish frame
    streamer.publish_image(frame)

    # Detect objects (includes segmentation automatically)
    if cortex.detect_objects_from_redis():
        detected_objects = cortex.get_detected_objects()

        # Check if segmentation masks are available
        for obj in detected_objects:
            if obj.get('has_mask', False):
                print(f"Object: {obj['label']}, Has mask: True")

        # Display annotated image with segmentation
        cv2.imshow("YOLOE Detection", cortex.get_annotated_image())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

### Continuous Processing Loop

```python
import cv2
import time
from vision_detect_segment import VisualCortex
from redis_robot_comm import RedisImageStreamer

# Initialize
streamer = RedisImageStreamer()
cortex = VisualCortex("yolo-world", device="cuda")

# Open camera
cap = cv2.VideoCapture(0)

try:
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            break

        # Publish to Redis
        streamer.publish_image(
            frame,
            metadata={"frame": cap.get(cv2.CAP_PROP_POS_FRAMES)}
        )

        # Detect objects
        if cortex.detect_objects_from_redis():
            # Show results
            annotated = cortex.get_annotated_image()
            cv2.imshow("Detections", annotated)

        # Break on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.033)  # ~30 FPS

finally:
    cap.release()
    cv2.destroyAllWindows()
```

---

### Processing Pre-recorded Video

```python
import cv2
from vision_detect_segment import VisualCortex
from redis_robot_comm import RedisImageStreamer

streamer = RedisImageStreamer()
cortex = VisualCortex("owlv2", device="cuda")

# Open video file
cap = cv2.VideoCapture("recording.mp4")
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Publish every 10th frame
    if frame_count % 10 == 0:
        streamer.publish_image(frame)

        if cortex.detect_objects_from_redis():
            objects = cortex.get_detected_objects()
            print(f"Frame {frame_count}: {len(objects)} objects")

cap.release()
```

---

## Performance Considerations

### Throughput Optimization

**1. Model Selection:**
```python
# Fastest - Real-time unified detection & segmentation (100-160 FPS on GPU)
cortex = VisualCortex("yoloe-11l", device="cuda")

# Fast - Real-time detection (30+ FPS on GPU)
cortex = VisualCortex("yolo-world", device="cuda")

# Medium - Open-vocabulary detection (10-15 FPS on GPU)
cortex = VisualCortex("owlv2", device="cuda")

# Slow - Complex queries (3-5 FPS on GPU)
cortex = VisualCortex("grounding_dino", device="cuda")
```

**2. Image Compression:**
```python
# High compression (faster, smaller)
streamer.publish_image(image, compress_jpeg=True, quality=70)

# Low compression (slower, better quality)
streamer.publish_image(image, compress_jpeg=True, quality=95)

# No compression (slowest, lossless)
streamer.publish_image(image, compress_jpeg=False)
```

**3. Reduce Object Labels:**
```python
# Fewer labels = faster detection
config = VisionConfig()
config.set_object_labels([
    "red cube", "blue sphere"  # Only 2 labels
])
```

**4. Disable Segmentation:**
```python
config = VisionConfig()
config.enable_segmentation = False  # Skip segmentation
```

**5. Adjust Confidence Threshold:**
```python
# Higher threshold = fewer detections = faster post-processing
config.model.confidence_threshold = 0.5  # Default: 0.3
```

---

### Memory Management

**Clear GPU Cache:**
```python
cortex.clear_cache()  # Clear PyTorch CUDA cache
```

**Monitor Memory:**
```python
memory_info = cortex.get_memory_usage()
print(f"Memory: {memory_info['rss_mb']:.1f} MB")
```

**Limit Redis Stream Size:**
```python
streamer.publish_image(image, maxlen=5)  # Keep only last 5 frames
```

---

### Latency Optimization

**End-to-end latency breakdown:**

| Component | Typical Latency | Notes |
|-----------|----------------|-------|
| Image encoding | 5-20 ms | Depends on JPEG quality |
| Redis publish | <1 ms | Local network |
| Redis retrieve | <1 ms | Local network |
| Image decoding | 5-20 ms | Depends on image size |
| Detection (YOLOE) | 6-10 ms | GPU-dependent |
| Detection (YOLO-World) | 20-50 ms | GPU-dependent |
| Detection (OWL-V2) | 100-200 ms | GPU-dependent |
| Segmentation (YOLOE built-in) | Included | Part of detection |
| Segmentation (FastSAM) | 50-100 ms | Per object |
| Segmentation (SAM2) | 200-500 ms | Per object |
| Annotation | 5-10 ms | CPU-bound |
| **Total (YOLOE)** | **20-50 ms** | **~20-50 FPS** |
| **Total (YOLO-World, no seg)** | **40-100 ms** | **~10-25 FPS** |
| **Total (OWL-V2 + SAM2)** | **500-1000 ms** | **~1-2 FPS** |

---

## Troubleshooting

### Common Issues

**1. Redis Connection Failed**
```
RedisConnectionError: Redis connection failed on localhost:6379
```
**Solution:**
```bash
# Start Redis server
docker run -p 6379:6379 redis:alpine

# Or check if Redis is running
redis-cli ping
```

---

**2. CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solution:**
```python
# Use CPU
cortex = VisualCortex("owlv2", device="cpu")

# Or clear cache
cortex.clear_cache()

# Or reduce image size
image = cv2.resize(image, (640, 480))

# Or use smaller model
cortex = VisualCortex("yoloe-11s", device="cuda")
```

---

**3. No Objects Detected**
```python
detected_objects = []  # Empty
```
**Solutions:**
```python
# 1. Lower confidence threshold
config.model.confidence_threshold = 0.1

# 2. Check object labels
config.set_object_labels(["red cube", "blue sphere"])

# 3. Verify image quality
cv2.imshow("Input", image)
```

---

**4. Model Loading Failed**
```
DependencyError: Missing dependency 'transformers'
```
**Solution:**
```bash
# Install dependencies
pip install transformers  # For OWL-V2, Grounding-DINO
pip install -U ultralytics>=8.3.0   # For YOLO-World, YOLOE
pip install segment-anything-2  # For SAM2 segmentation (optional)
```

---

**5. Tracking IDs Inconsistent**
```python
# Track IDs change every frame
```
**Solution:**
```python
# Ensure tracking is enabled
detector = ObjectDetector(
    ...
    enable_tracking=True
)

# For YOLO, verify persist=True
results = model.track(image, persist=True)
```

---

**6. Label Flickering Between Frames**
```python
# Labels keep changing: "cat" -> "dog" -> "cat"
```
**Solution:**
```python
# Use label stabilization (enabled by default)
tracker = ObjectTracker(
    model=model,
    model_id="owlv2",
    enable_tracking=True,
    stabilization_frames=10  # Stabilize after 10 frames
)

# Labels will be locked after stabilization period
```

---

**7. Segmentation Masks Not Generated**
```python
# has_mask: False for all objects
```
**Solution:**
```python
# Option 1: Use YOLOE with built-in segmentation
cortex = VisualCortex("yoloe-11l", device="cuda")

# Option 2: Check if segmentation is enabled
config.enable_segmentation = True

# Option 3: Install segmentation dependencies
pip install segment-anything-2  # For SAM2
# or use FastSAM (included with ultralytics)
```

---

## Best Practices

### 1. Configuration Management
```python
# Create custom config
config = VisionConfig()
config.model.confidence_threshold = 0.3
config.redis.host = "192.168.1.100"
config.verbose = True

# Save for later use
import json
config_dict = {
    "model": {
        "confidence_threshold": config.model.confidence_threshold
    }
}
with open("config.json", "w") as f:
    json.dump(config_dict, f)
```

---

### 2. Error Handling
```python
from vision_detect_segment.utils.exceptions import (
    VisionDetectionError,
    RedisConnectionError
)

try:
    cortex = VisualCortex("owlv2")
    cortex.detect_objects_from_redis()
except RedisConnectionError as e:
    print(f"Redis error: {e}")
except VisionDetectionError as e:
    print(f"Detection error: {e}")
```

---

### 3. Logging
```python
# Enable verbose logging
cortex = VisualCortex("owlv2", verbose=True)

# Logs will show:
# - Model loading time
# - Detection timing
# - Object counts
# - Memory usage
```

---

### 4. Testing
```python
# Use test config for faster iteration
from vision_detect_segment.config import create_test_config

config = create_test_config()  # Reduced object labels
cortex = VisualCortex("owlv2", config=config)
```

---

### 5. Production Deployment

**Optimize for throughput:**
```python
# Use YOLOE for best performance
config = get_default_config("yoloe-11m")
config.model.confidence_threshold = 0.5
config.enable_segmentation = True  # Built-in segmentation

cortex = VisualCortex(
    objdetect_model_id="yoloe-11m",
    device="cuda",
    config=config,
    publish_annotated=False  # Disable if not needed
)
```

**Monitor performance:**
```python
# Get processing statistics
stats = cortex.get_stats()
print(f"Processed frames: {stats['processed_frames']}")
print(f"Device: {stats['device']}")
print(f"Detection count: {stats['current_detections_count']}")

# Monitor memory
memory = cortex.get_memory_usage()
print(f"Memory usage: {memory['rss_mb']:.1f} MB")
```

**Handle errors gracefully:**
```python
import time

while True:
    try:
        success = cortex.detect_objects_from_redis()
        if not success:
            time.sleep(0.1)  # Wait for new images
            continue

        # Process results
        objects = cortex.get_detected_objects()

    except KeyboardInterrupt:
        print("Stopping...")
        cortex.cleanup()
        break

    except Exception as e:
        print(f"Error: {e}")
        time.sleep(1.0)  # Wait before retrying
```

---

## Summary

The `vision_detect_segment` workflow provides a complete pipeline for real-time object detection, tracking, and segmentation:

1. ✅ **Image Capture** - Redis-based streaming with compression
2. ✅ **Detection** - Multi-model support (YOLOE, YOLO-World, OWL-V2, Grounding-DINO)
3. ✅ **Tracking** - Persistent object IDs with progressive label stabilization
4. ✅ **Segmentation** - YOLOE built-in or external (SAM2, FastSAM)
5. ✅ **Publishing** - Results back to Redis for downstream processing
6. ✅ **Visualization** - Annotated images for debugging

**Key Features:**
- **Progressive Label Stabilization**: Labels shown from frame 1, stabilized after N frames
- **Unified Detection & Segmentation**: YOLOE provides both in one model
- **Real-Time Performance**: Up to 160 FPS with YOLOE on GPU
- **Open Vocabulary**: Detect custom objects without retraining
- **Flexible Configuration**: Easy to customize for different use cases

**Recommended Configurations:**
- **Speed-critical (>30 FPS)**: YOLOE-11s/m with built-in segmentation
- **Balanced (10-25 FPS)**: YOLO-World with FastSAM
- **High accuracy (<5 FPS)**: OWL-V2 or Grounding-DINO with SAM2
- **Custom classes**: YOLOE or OWL-V2 with text prompts

The system is designed for robotics applications requiring real-time vision processing with flexible configuration and robust error handling.

---

For questions or issues, please see the [main repository](https://github.com/dgaida/vision_detect_segment).
