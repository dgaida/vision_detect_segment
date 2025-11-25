# API Reference

## UML Class Diagram

![UML Class Diagram](vision_uml_diagram.png)

## VisualCortex

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

## Detection Results Format

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
