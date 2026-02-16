import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class DetectionBackend(ABC):
    """Abstract base class for detection model backends."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the detection model and processor."""
        pass

    @abstractmethod
    def detect(self, image: np.ndarray, threshold: float) -> List[Dict[str, Any]]:
        """
        Run detection on the given image.

        Args:
            image: Input image in RGB format
            threshold: Confidence threshold

        Returns:
            List of detected objects as dictionaries
        """
        pass

    @property
    @abstractmethod
    def supports_tracking(self) -> bool:
        """Whether the backend has built-in tracking support."""
        pass

    @property
    @abstractmethod
    def supports_segmentation(self) -> bool:
        """Whether the backend has built-in segmentation support."""
        pass

    @abstractmethod
    def add_label(self, label: str) -> None:
        """Add a new label to the detector."""
        pass
