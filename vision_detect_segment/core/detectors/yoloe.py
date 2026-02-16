import numpy as np
import torch
from typing import List, Dict, Any, Optional
from ultralytics import YOLOE
from .base import DetectionBackend
from ...utils.config import MODEL_CONFIGS

class YOLOEBackend(DetectionBackend):
    """YOLOE detection backend."""

    def __init__(self, model_id: str, device: str, object_labels: List[str]):
        self.model_id = model_id
        self.device = device
        self.object_labels = object_labels
        self.model = None

    def load_model(self) -> None:
        model_config = MODEL_CONFIGS[self.model_id]
        model_path = model_config.model_params["model_path"]
        self.model = YOLOE(model_path)

        is_prompt_free = model_config.model_params.get("is_prompt_free", False)
        if not is_prompt_free and model_config.model_params.get("supports_prompts", True):
            self.model.set_classes(self.object_labels, self.model.get_text_pe(self.object_labels))

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
                "results": results
            }

            if hasattr(boxes, "id") and boxes.id is not None:
                obj_dict["track_id"] = int(boxes.id[i])

            detected_objects.append(obj_dict)

        # Handle built-in segmentation
        if hasattr(results[0], "masks") and results[0].masks is not None:
            masks_data = results[0].masks.data
            for i, obj in enumerate(detected_objects):
                if i < len(masks_data):
                    mask = masks_data[i].cpu().numpy()
                    mask_8u = (mask * 255).astype(np.uint8)
                    obj["mask_8u"] = mask_8u
                    obj["has_mask"] = True

        return detected_objects

    @property
    def supports_tracking(self) -> bool:
        return True

    @property
    def supports_segmentation(self) -> bool:
        return True

    def add_label(self, label: str) -> None:
        if label.lower() not in [l.lower() for l in self.object_labels]:
            self.object_labels.append(label.lower())
            model_config = MODEL_CONFIGS.get(self.model_id)
            if self.model and model_config and not model_config.model_params.get("is_prompt_free", False):
                self.model.set_classes(self.object_labels, self.model.get_text_pe(self.object_labels))
