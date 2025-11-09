import numpy as np
import supervision as sv
from typing import Any

# --- Kompatibilitätsimport für ByteTrack ---
try:
    from supervision.tracker.byte_tracker.core import ByteTrack
except ImportError:
    from supervision.tracker.byte_tracker import ByteTrack

from ..utils.exceptions import DetectionError


class ObjectTracker:
    """
    Unified object tracking interface that wraps YOLO built-in tracking
    and ByteTrack tracking for transformer-based models.
    """

    def __init__(self, model: Any, model_id: str, enable_tracking: bool = False, verbose: bool = False):
        """
        Args:
            model: The underlying detection model (YOLO, OWL-V2, etc.)
            model_id: Identifier for the model type
            enable_tracking: Whether to enable persistent tracking
            verbose: Enable verbose logging
        """
        self.model = model
        self.model_id = model_id.lower()
        self.enable_tracking = enable_tracking
        self.verbose = verbose

        # Use Ultralytics tracker for YOLO models, ByteTrack for others
        self._use_yolo_tracker = "yolo" in self.model_id
        self._tracker = ByteTrack() if enable_tracking and not self._use_yolo_tracker else None

    def reset(self):
        """Reset internal tracking state."""
        if self._tracker:
            self._tracker.reset()
        if self._use_yolo_tracker and hasattr(self.model, "tracker"):
            try:
                self.model.tracker.reset()
            except Exception:
                pass

    # ----------------------------------------------------------------------
    # 🟢 YOLO tracking
    # ----------------------------------------------------------------------
    def track(self, image: np.ndarray, threshold: float = 0.25, max_det: int = 50):
        """
        Run YOLO built-in tracking if supported; otherwise fallback to detection.
        For transformer-based models, this is a no-op (tracking is handled elsewhere).

        Returns:
            The result object from the YOLO model (ultralytics.engine.results.Results)
        """
        print("track", self.enable_tracking)

        if not self._use_yolo_tracker:
            # For OWL-V2 / DINO this method is not used
            raise DetectionError("YOLO tracking is not applicable to transformer-based models")

        try:
            if self.enable_tracking:
                # Run Ultralytics tracking — persist ensures track IDs stay consistent
                results = self.model.track(
                    image,
                    persist=True,
                    stream=False,
                    verbose=False,
                    conf=threshold,
                    max_det=max_det
                )
            else:
                # Fall back to standard detection if tracking disabled
                results = self.model.predict(
                    image,
                    conf=threshold,
                    max_det=max_det,
                    verbose=False
                )

            return results

        except Exception as e:
            raise DetectionError(f"YOLO tracking failed: {e}")

    # ----------------------------------------------------------------------
    # 🟣 Transformer tracking helper
    # ----------------------------------------------------------------------
    def update_with_detections(self, detections: sv.Detections) -> sv.Detections:
        """
        Run ByteTrack tracking update for transformer-based models.

        Args:
            detections: supervision.Detections object

        Returns:
            Updated Detections with tracker IDs
        """
        if not self._tracker or not self.enable_tracking:
            return detections
        return self._tracker.update_with_detections(detections)
