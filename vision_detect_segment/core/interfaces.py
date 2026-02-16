from typing import TypedDict, List, Optional, Tuple, Any, Protocol, Dict
import numpy as np
import torch

class BoundingBox(TypedDict):
    x_min: int
    y_min: int
    x_max: int
    y_max: int

class DetectedObject(TypedDict, total=False):
    """Detected object with optional segmentation and tracking."""
    label: str
    confidence: float
    bbox: BoundingBox
    track_id: Optional[int]
    has_mask: bool
    mask_data: Optional[str]  # Base64 encoded
    mask_shape: Optional[List[int]]
    mask_dtype: Optional[str]
    # Internal use
    results: Optional[Any]

class DetectionModel(Protocol):
    """Protocol for detection model backends."""
    def predict(self, image: np.ndarray, conf: float, max_det: int) -> Any: ...
    def get_names(self) -> Dict[int, str]: ...

class Tracker(Protocol):
    """Protocol for object tracking."""
    def update_with_detections(self, detections: Any) -> Any: ...
    def reset(self) -> None: ...
