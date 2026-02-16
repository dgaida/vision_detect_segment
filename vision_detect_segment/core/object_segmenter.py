import numpy as np
import cv2
import torch
import supervision as sv
from typing import Optional, Tuple, List, Any
import logging

from ..utils.config import VisionConfig
from ..utils.exceptions import SegmentationError, DependencyError, ModelLoadError, handle_model_loading_error
from ..utils.utils import setup_logging, get_optimal_device, Timer, validate_image, clear_gpu_cache

# Handle optional dependencies gracefully
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    SAM2ImagePredictor = None
    SAM2_AVAILABLE = False

try:
    from ultralytics import FastSAM
    FASTSAM_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    FastSAM = None
    FASTSAM_AVAILABLE = False


class ObjectSegmenter:
    """
    A class for handling image segmentation tasks using models like SAM2 or FastSAM.
    """

    def __init__(
        self,
        segmentation_model: Optional[str] = "facebook/sam2.1-hiera-tiny",
        device: str = "cuda",
        verbose: bool = False,
        config: Optional[VisionConfig] = None,
    ):
        """
        Initialize the ObjectSegmenter.

        Args:
            segmentation_model: identifier for the segmentation model to use.
            device: Device to run the segmentation model on.
            verbose: Enable verbose logging
            config: Optional VisionConfig instance
        """
        self._device = get_optimal_device(prefer_gpu=(device != "cpu"))
        self._segmentation_model = segmentation_model
        self._model_id: Optional[str] = None
        self._segmenter: Optional[Any] = None
        self._verbose = verbose
        self._config = config or VisionConfig()
        self._logger = setup_logging(verbose)

        try:
            self._initialize_segmenter(segmentation_model)
            if verbose:
                self._logger.info(f"ObjectSegmenter initialized with {self._model_id} on {self._device}")
        except Exception as e:
            model_error = handle_model_loading_error(f"segmentation_{self._model_id}", e)
            self._logger.error(str(model_error))
            self._segmenter = None

    def _initialize_segmenter(self, segmentation_model: Optional[str]) -> None:
        """Initialize the segmentation model based on availability."""
        if SAM2_AVAILABLE and segmentation_model and "sam2" in segmentation_model.lower():
            self._model_id = "sam2.1-hiera-tiny"
            with Timer("Loading SAM2 model", self._logger if self._verbose else None):
                self._segmenter = SAM2ImagePredictor.from_pretrained(segmentation_model)
        elif FASTSAM_AVAILABLE:
            self._model_id = "fastsam"
            with Timer("Loading FastSAM model", self._logger if self._verbose else None):
                self._segmenter = FastSAM("FastSAM-x.pt")
        else:
            raise DependencyError("sam2 or ultralytics", "segmentation")

    def segment_objects(self, image: np.ndarray, detections: sv.Detections) -> sv.Detections:
        """
        Segments objects based on detections.

        Args:
            image: Input image
            detections: Object detections with bounding boxes

        Returns:
            Updated detections with masks
        """
        if not self._segmenter:
            raise SegmentationError("Segmentation model not loaded")

        validate_image(image)
        masks = []
        for box in detections.xyxy:
            try:
                box_tensor = torch.from_numpy(box) if isinstance(box, np.ndarray) else box
                _, mask = self.segment_box_in_image(box_tensor, image)
                masks.append(mask)
            except Exception as e:
                if self._verbose:
                    self._logger.warning(f"Segmentation failed for detection: {e}")
                masks.append(None)

        valid_masks = [m for m in masks if m is not None]
        if valid_masks:
            detections.mask = np.stack(valid_masks) if len(valid_masks) == len(detections.xyxy) else None
        return detections

    def segment_box_in_image(
        self, box: torch.Tensor, img_work: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Segment a single bounding box."""
        if self._segmenter is None:
            raise SegmentationError("Segmentation model not available")

        if self._model_id == "fastsam":
            return self._segment_box_with_fastsam(box, img_work)
        else:
            return self._segment_box_with_sam2(box, img_work)

    def _segment_box_with_fastsam(self, box: torch.Tensor, img_work: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Perform segmentation using FastSAM."""
        input_box = box.detach().cpu().numpy().tolist()
        imgsz = (img_work.shape[0] // 32) * 32 + 32

        try:
            results = self._segmenter(
                img_work, device=self._device, retina_masks=True, imgsz=imgsz,
                conf=0.4, iou=0.9, bboxes=input_box, verbose=False
            )
            masks = results[0].masks
            if masks is not None:
                mask_8u = self._create_mask8u(img_work, input_box, masks)
                return mask_8u, mask_8u > 0
            return None, None
        except Exception as e:
            if self._verbose: self._logger.error(f"FastSAM error: {e}")
            return None, None

    def _segment_box_with_sam2(self, box: torch.Tensor, img_work: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Perform segmentation using SAM2."""
        try:
            self._segmenter.set_image(img_work)
            input_box = box.detach().cpu().numpy()
            masks, scores, _ = self._segmenter.predict(
                point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=True
            )
            index = np.argmax(scores)
            mask = masks[index]
            mask_8u = (mask > 0).astype(np.uint8) * 255
            return mask_8u, mask > 0
        except Exception as e:
            if self._verbose: self._logger.error(f"SAM2 error: {e}")
            return None, None

    @staticmethod
    def _create_mask8u(img_work: np.ndarray, input_box: list, masks) -> np.ndarray:
        x_min, y_min, x_max, y_max = map(int, input_box)
        mask_data = masks.data[0].cpu().numpy()
        mask_8u = (mask_data > 0.5).astype(np.uint8) * 255
        full_mask = np.zeros(img_work.shape[:2], dtype=np.uint8)
        full_mask[y_min:y_max, x_min:x_max] = mask_8u[y_min:y_max, x_min:x_max]
        return full_mask

    def get_segmenter(self) -> Optional[Any]: return self._segmenter
    def get_model_id(self) -> Optional[str]: return self._model_id
    def is_available(self) -> bool: return self._segmenter is not None

    # Deprecated
    def segmenter(self): return self.get_segmenter()
    def verbose(self): return self._verbose
