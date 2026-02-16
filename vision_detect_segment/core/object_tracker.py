"""
Unified object tracking interface with progressive label stabilization.
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import supervision as sv

# --- Compatibility import for ByteTrack ---
try:
    from supervision.tracker.byte_tracker.core import ByteTrack
except ImportError:
    from supervision.tracker.byte_tracker import ByteTrack

from ..utils.exceptions import DetectionError


class ObjectTracker:
    """
    Unified object tracking interface with progressive label stabilization.

    Features:
    - Wraps YOLO built-in tracking and ByteTrack for transformer models
    - Tracks label history for each tracked object
    - Shows majority label from frame 1 onwards
    - Stabilizes labels after N frames by locking to most common label
    """

    def __init__(
        self,
        model: Any,
        model_id: str,
        enable_tracking: bool = False,
        verbose: bool = False,
        stabilization_frames: int = 10,
        min_frames_for_display: int = 1,
    ):
        """
        Initialize ObjectTracker.

        Args:
            model: The underlying detection model (YOLO, OWL-V2, etc.)
            model_id: Identifier for the model type
            enable_tracking: Whether to enable persistent tracking
            verbose: Enable verbose logging
            stabilization_frames: Number of frames before locking label
            min_frames_for_display: Minimum frames before showing any label
        """
        self.model = model
        self.model_id = model_id.lower()
        self.enable_tracking = enable_tracking
        self._verbose = verbose
        self.stabilization_frames = stabilization_frames
        self.min_frames_for_display = min_frames_for_display

        # Use Ultralytics tracker for YOLO models, ByteTrack for others
        self._use_yolo_tracker = "yolo" in self.model_id or "yoloe" in self.model_id
        self._tracker = ByteTrack() if enable_tracking and not self._use_yolo_tracker else None

        # Label tracking data structures
        self._label_history: Dict[int, List[str]] = defaultdict(list)
        self._stabilized_labels: Dict[int, str] = {}
        self._frame_counts: Dict[int, int] = defaultdict(int)

    def reset(self) -> None:
        """Reset internal tracking state and label history."""
        if self._tracker:
            self._tracker.reset()
        if self._use_yolo_tracker and hasattr(self.model, "tracker"):
            try:
                self.model.tracker.reset()
            except Exception as e:
                if self._verbose:
                    print(f"Warning: Could not reset YOLO tracker: {e}")

        # Reset label tracking
        self._label_history.clear()
        self._stabilized_labels.clear()
        self._frame_counts.clear()

    def update_label_history(self, track_ids: np.ndarray, labels: List[str]) -> List[str]:
        """
        Update label history for tracked objects and return progressive labels.

        Args:
            track_ids: Array of track IDs
            labels: List of detected labels corresponding to track IDs

        Returns:
            List of labels to display (majority vote or stabilized)
        """
        if not self.enable_tracking or track_ids is None or len(track_ids) == 0:
            return labels

        display_labels = []

        for track_id, current_label in zip(track_ids, labels):
            track_id = int(track_id)

            # New track handling
            if track_id not in self._label_history or len(self._label_history[track_id]) == 0:
                self._label_history[track_id] = []
                self._stabilized_labels.pop(track_id, None)
                self._frame_counts[track_id] = 0
                if self._verbose:
                    print(f"New track detected: ID {track_id}")

            self._frame_counts[track_id] += 1
            self._label_history[track_id].append(current_label)

            if self._frame_counts[track_id] < self.min_frames_for_display:
                display_label = current_label
            elif self._frame_counts[track_id] >= self.stabilization_frames:
                if track_id not in self._stabilized_labels:
                    label_counter = Counter(self._label_history[track_id])
                    most_common_label = label_counter.most_common(1)[0][0]
                    self._stabilized_labels[track_id] = most_common_label
                display_label = self._stabilized_labels[track_id]
            else:
                label_counter = Counter(self._label_history[track_id])
                display_label = label_counter.most_common(1)[0][0]

            display_labels.append(display_label)

        return display_labels

    def detect_lost_tracks(self, current_track_ids: np.ndarray) -> List[int]:
        """Detect tracks that have been lost."""
        if current_track_ids is None:
            return []

        current_ids = set(int(tid) for tid in current_track_ids)
        known_ids = set(self._label_history.keys())
        return list(known_ids - current_ids)

    def cleanup_lost_tracks(self, lost_track_ids: List[int]) -> None:
        """Clean up data for tracks that have been lost."""
        for track_id in lost_track_ids:
            self._label_history.pop(track_id, None)
            self._stabilized_labels.pop(track_id, None)
            self._frame_counts.pop(track_id, None)

    def get_all_track_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get tracking statistics for all active tracks."""
        return {track_id: self.get_track_info(track_id) for track_id in self._label_history.keys()}

    def get_track_info(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Get tracking information for a specific track ID."""
        if track_id not in self._label_history:
            return None

        label_counter = Counter(self._label_history[track_id])
        majority_label = label_counter.most_common(1)[0][0] if label_counter else None

        return {
            "track_id": track_id,
            "frame_count": self._frame_counts[track_id],
            "label_history": self._label_history[track_id],
            "label_distribution": dict(label_counter),
            "current_majority": majority_label,
            "stabilized_label": self._stabilized_labels.get(track_id),
            "is_stabilized": track_id in self._stabilized_labels,
        }

    def track(self, image: np.ndarray, threshold: float = 0.25, max_det: int = 50) -> Any:
        """Run YOLO built-in tracking."""
        if not self._use_yolo_tracker:
            raise DetectionError("YOLO tracking is not applicable to transformer-based models")

        try:
            if self.enable_tracking:
                return self.model.track(image, persist=True, stream=False, verbose=False, conf=threshold, max_det=max_det)
            return self.model.predict(image, conf=threshold, max_det=max_det, verbose=False)
        except Exception as e:
            raise DetectionError(f"YOLO tracking failed: {e}")

    def update_with_detections(self, detections: sv.Detections) -> sv.Detections:
        """Run ByteTrack tracking update."""
        if not self._tracker or not self.enable_tracking:
            return detections
        return self._tracker.update_with_detections(detections)

    @property
    def verbose(self) -> bool:
        """Legacy property for backward compatibility."""
        return self._verbose
