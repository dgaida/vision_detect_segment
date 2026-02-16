import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO
from .base import DetectionBackend
from ...utils.config import MODEL_CONFIGS

class YOLOWorldBackend(DetectionBackend):
    """YOLO-World detection backend."""

    def __init__(self, model_id: str, device: str, object_labels: List[str]):
        self.model_id = model_id
        self.device = device
        self.object_labels = object_labels
        self.model = None

    def load_model(self) -> None:
        model_config = MODEL_CONFIGS["yolo-world"]
        model_path = model_config.model_params["model_path"]
        self.model = YOLO(model_path)
        self.model.set_classes(self.object_labels)

    def detect(self, image: np.ndarray, threshold: float) -> List[Dict[str, Any]]:
        results = self.model.predict(image, conf=threshold, max_det=20, verbose=False)

        detected_objects = []
        boxes = results[0].boxes

        if boxes is None:
            return detected_objects

        for i, box in enumerate(boxes):
            cls = int(boxes.cls[i])
            class_name = results[0].names[cls]
            confidence = float(boxes.conf[i])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            obj_dict = {
                "label": class_name,
                "confidence": confidence,
                "bbox": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2},
                "has_mask": False,
                "results": results # Keep results for supervision
            }

            if hasattr(boxes, "id") and boxes.id is not None:
                obj_dict["track_id"] = int(boxes.id[i])

            detected_objects.append(obj_dict)

        return detected_objects

    @property
    def supports_tracking(self) -> bool:
        return True

    @property
    def supports_segmentation(self) -> bool:
        return False

    def add_label(self, label: str) -> None:
        if label.lower() not in [lbl.lower() for lbl in self.object_labels]:
            self.object_labels.append(label.lower())
            if self.model:
                self.model.set_classes(self.object_labels)
