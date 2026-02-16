import numpy as np
import torch
from typing import List, Dict, Any, Union
from transformers import Owlv2Processor, Owlv2ForObjectDetection, AutoProcessor, AutoModelForZeroShotObjectDetection
from .base import DetectionBackend
from ...utils.config import MODEL_CONFIGS

class TransformerBackend(DetectionBackend):
    """Base class for transformer-based detectors (OWL-V2, Grounding-DINO)."""

    def __init__(self, model_id: str, device: str, object_labels: List[str]):
        self.model_id = model_id
        self.device = device
        self.object_labels = object_labels
        self.model = None
        self.processor = None
        self.processed_labels = self._preprocess_labels(object_labels, model_id)

    def _preprocess_labels(self, labels: List[str], model_id: str) -> Union[List[str], str]:
        if model_id == "grounding_dino":
            return ". ".join([label.lower() for label in labels]) + "."
        return labels

    def load_model(self) -> None:
        model_config = MODEL_CONFIGS[self.model_id]
        model_path = model_config.model_params["model_path"]

        if self.model_id == "owlv2":
            self.processor = Owlv2Processor.from_pretrained(model_path)
            self.model = Owlv2ForObjectDetection.from_pretrained(model_path).to(self.device)
        else: # grounding_dino
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(self.device)

    def detect(self, image: np.ndarray, threshold: float) -> List[Dict[str, Any]]:
        h, w = image.shape[:2]
        inputs = self.processor(images=image, text=self.processed_labels, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        if self.model_id == "owlv2":
            results = self.processor.post_process_object_detection(
                outputs=outputs, target_sizes=[(h, w)], threshold=threshold
            )
            labels = self._extract_owlv2_labels(results)
        else: # grounding_dino
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=threshold, text_threshold=0.3, target_sizes=[(h, w)]
            )
            labels = results[0]["labels"]

        detected_objects = []
        for i, (box, score) in enumerate(zip(results[0]["boxes"], results[0]["scores"])):
            x_min, y_min, x_max, y_max = map(int, box)
            label = labels[i]

            obj_dict = {
                "label": str(label),
                "confidence": float(score),
                "bbox": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
                "has_mask": False,
                "results": results[0]
            }
            detected_objects.append(obj_dict)

        return detected_objects

    def _extract_owlv2_labels(self, results) -> np.ndarray:
        labels = []
        for label_idx in results[0]["labels"]:
            label_text = self.object_labels[label_idx.item()]
            labels.append(label_text)
        return np.array(labels)

    @property
    def supports_tracking(self) -> bool:
        return False

    @property
    def supports_segmentation(self) -> bool:
        return False

    def add_label(self, label: str) -> None:
        if label.lower() not in [lbl.lower() for lbl in self.object_labels]:
            self.object_labels.append(label.lower())
            self.processed_labels = self._preprocess_labels(self.object_labels, self.model_id)

class Owlv2Backend(TransformerBackend):
    """OWL-V2 specific backend."""
    pass

class GroundingDinoBackend(TransformerBackend):
    """Grounding-DINO specific backend."""
    pass
