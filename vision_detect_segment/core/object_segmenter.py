import numpy as np
import cv2
import torch
import supervision as sv
from typing import Optional, Tuple

from ..utils.config import VisionConfig
from ..utils.exceptions import SegmentationError, DependencyError, ModelLoadError, handle_model_loading_error
from ..utils.utils import setup_logging, get_optimal_device, Timer, validate_image, clear_gpu_cache

# Handle optional dependencies gracefully
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    SAM2_AVAILABLE = True
except (ModuleNotFoundError, ImportError) as e:
    print("SAM2 module not installed or cannot be found!", e)
    SAM2ImagePredictor = None
    SAM2_AVAILABLE = False

try:
    from ultralytics import FastSAM

    FASTSAM_AVAILABLE = True
except (ModuleNotFoundError, ImportError) as e:
    print("FastSAM module not installed or cannot be found!", e)
    FastSAM = None
    FASTSAM_AVAILABLE = False


class ObjectSegmenter:
    """
    A class for handling image segmentation tasks using a segmentation model like SAM2 or FastSAM.

    This class enables object segmentation in an image based on detections and provides
    tools to generate and attach segmentation masks to detected objects.
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
            segmentation_model: Path or identifier for the segmentation model to use.
            device: Device to run the segmentation model on. Can be 'cuda' or 'cpu'.
            verbose: Enable verbose logging
            config: Optional VisionConfig instance
        """
        # Private attributes
        self._device = get_optimal_device(prefer_gpu=(device != "cpu"))
        self._segmentation_model = segmentation_model
        self._model_id = None
        self._segmenter = None

        # Public configuration
        self.verbose = verbose
        self._config = config or VisionConfig()

        # Setup logging
        self._logger = setup_logging(verbose)

        try:
            self._initialize_segmenter(segmentation_model)

            if verbose:
                self._logger.info(f"ObjectSegmenter initialized with {self._model_id} on {self._device}")

        except Exception as e:
            model_error = handle_model_loading_error(f"segmentation_{self._model_id}", e)
            self._logger.error(str(model_error))
            # Don't raise here - allow system to work without segmentation
            self._segmenter = None
            if verbose:
                self._logger.warning("Segmentation will be disabled")

    def _initialize_segmenter(self, segmentation_model: Optional[str]):
        """Initialize the segmentation model."""
        if SAM2_AVAILABLE and segmentation_model and "sam2" in segmentation_model.lower():
            self._model_id = "sam2.1-hiera-tiny"
            with Timer("Loading SAM2 model", self._logger if self.verbose else None):
                self._segmenter = SAM2ImagePredictor.from_pretrained(segmentation_model)
        elif FASTSAM_AVAILABLE:
            self._model_id = "fastsam"
            with Timer("Loading FastSAM model", self._logger if self.verbose else None):
                self._segmenter = FastSAM("FastSAM-x.pt")
        else:
            available_models = []
            if SAM2_AVAILABLE:
                available_models.append("SAM2")
            if FASTSAM_AVAILABLE:
                available_models.append("FastSAM")

            if not available_models:
                raise DependencyError(
                    "sam2 or ultralytics",
                    "segmentation",
                    "Install with: pip install segment-anything-2 or pip install ultralytics",
                )

            # Fallback to FastSAM if available
            if FASTSAM_AVAILABLE:
                self._model_id = "fastsam"
                self._segmenter = FastSAM("FastSAM-x.pt")
            else:
                raise ModelLoadError("segmentation", "No compatible segmentation model available")

    def segment_objects(self, image: np.ndarray, detections: sv.Detections) -> sv.Detections:
        """
        Segments objects in the provided image based on detections.

        Args:
            image: Input image containing the objects to be segmented.
            detections: Object detections including bounding boxes.

        Returns:
            sv.Detections: Updated detections with segmentation masks added.

        Raises:
            SegmentationError: If the segmentation model is not loaded.
        """
        if not self._segmenter:
            raise SegmentationError("Segmentation model not loaded")

        try:
            validate_image(image)
        except Exception as e:
            raise SegmentationError(f"Invalid input image: {e}")

        masks = []
        for i, box in enumerate(detections.xyxy):
            try:
                # Convert numpy array to torch tensor if needed
                if isinstance(box, np.ndarray):
                    box_tensor = torch.from_numpy(box)
                else:
                    box_tensor = box

                mask_8u, mask = self.segment_box_in_image(box_tensor, image)
                if mask is not None:
                    masks.append(mask)
                else:
                    if self.verbose:
                        self._logger.warning(f"No mask generated for detection {i}")

            except Exception as e:
                if self.verbose:
                    self._logger.warning(f"Segmentation failed for detection {i}: {e}")
                masks.append(None)

        # Filter out None masks
        valid_masks = [m for m in masks if m is not None]
        if valid_masks:
            detections.mask = valid_masks

        return detections

    def segment_box_in_image(
        self, box: torch.Tensor, img_work: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Segments a detected object's bounding box in the image.

        Args:
            box: Bounding box coordinates of the object (format: [x_min, y_min, x_max, y_max]).
            img_work: Current workspace image where the object resides.

        Returns:
            tuple:
                - Optional[np.ndarray]: uint8 mask of the segmented object, normalized to 255,
                or None if segmentation fails.
                - Optional[np.ndarray]: Binary mask (boolean array) of the segmented object,
                or None if segmentation fails.

        Raises:
            SegmentationError: If segmentation fails due to model or input errors.
        """
        if self._segmenter is None:
            raise SegmentationError("Segmentation model not available")

        try:
            validate_image(img_work)
        except Exception as e:
            raise SegmentationError(f"Invalid input image: {e}")

        try:
            if self._model_id == "fastsam":
                return self._segment_box_with_fastsam(box, img_work)
            else:  # SAM2
                return self._segment_box_with_sam2(box, img_work)
        except Exception as e:
            raise SegmentationError(
                f"Segmentation failed: {e}", bbox=box.detach().cpu().numpy().tolist(), segmentation_model=self._model_id
            )

    def _segment_box_with_fastsam(
        self, box: torch.Tensor, img_work: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Segment using FastSAM model."""
        input_box = box.detach().cpu().numpy()
        x_min, y_min, x_max, y_max = map(int, input_box)
        input_box = [x_min, y_min, x_max, y_max]

        # Ensure image size is multiple of 32 for FastSAM
        imgsz = (img_work.shape[0] // 32) * 32 + 32

        try:
            # Run FastSAM inference
            everything_results = self._segmenter(
                img_work,
                device=self._device,
                retina_masks=True,
                imgsz=imgsz,
                conf=0.4,
                iou=0.9,
                bboxes=input_box,
                verbose=False,
            )

            masks = everything_results[0].masks

            if masks is not None:
                mask_8u = self._create_mask8u(img_work, input_box, masks)
                mask_binary = mask_8u > 0

                if not mask_binary.any():
                    if self.verbose:
                        self._logger.warning(f"Empty mask generated for bbox {input_box}")
                    return None, None

                return mask_8u, mask_binary
            else:
                return None, None

        except Exception as e:
            if self.verbose:
                self._logger.error(f"FastSAM segmentation failed: {e}")
            return None, None
        finally:
            # Clear GPU cache to prevent memory issues
            if self._device == "cuda":
                clear_gpu_cache()

    def _segment_box_with_sam2(
        self, box: torch.Tensor, img_work: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Segment using SAM2 model."""
        try:
            with torch.inference_mode():
                if self._device == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        return self._run_sam2_inference(box, img_work)
                else:
                    return self._run_sam2_inference(box, img_work)
        finally:
            # Clear GPU cache to prevent memory issues
            if self._device == "cuda":
                clear_gpu_cache()

    def _run_sam2_inference(
        self, box: torch.Tensor, img_work: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Run SAM2 model inference."""
        self._segmenter.set_image(img_work)
        input_box = box.detach().cpu().numpy()

        try:
            masks, scores, _ = self._segmenter.predict(
                point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=True
            )

            # Choose mask with the highest score
            index = np.argmax(scores)
            mask = masks[index]
            mask_normalized = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
            mask_8u = mask_normalized.astype(np.uint8)

            if mask_8u is not None:
                mask_binary = mask_8u > 0
                return mask_8u, mask_binary
            else:
                return None, None

        except Exception as e:
            if self.verbose:
                self._logger.error(f"SAM2 inference failed: {e}")
            return None, None

    @staticmethod
    def _create_mask8u(img_work: np.ndarray, input_box: list, masks) -> np.ndarray:
        """Extract and process the segmentation mask for FastSAM."""
        x_min, y_min, x_max, y_max = input_box
        mask_data = masks.data[0].cpu().numpy()

        # Convert the mask to binary (255 for mask, 0 for background)
        mask_8u = (mask_data > 0.5).astype(np.uint8) * 255

        # Ensure that the mask respects the input box region
        mask_binary_cropped = mask_8u[y_min:y_max, x_min:x_max]

        full_mask = np.zeros(img_work.shape[:2], dtype=np.uint8)

        # Place the cropped mask back into the full-sized mask to avoid overflow
        full_mask[y_min:y_max, x_min:x_max] = mask_binary_cropped

        return full_mask

    # Public API methods
    def get_segmenter(self) -> Optional[any]:
        """
        Returns the loaded segmentation model.

        Returns:
            Optional segmentation model instance, or None if the model failed to load.
        """
        return self._segmenter

    def get_model_id(self) -> Optional[str]:
        """Get the current segmentation model identifier."""
        return self._model_id

    def get_device(self) -> str:
        """Get the current computation device."""
        return self._device

    def is_available(self) -> bool:
        """Check if segmentation is available."""
        return self._segmenter is not None

    # Static utility methods
    @staticmethod
    def add_masks2detections(detections: sv.Detections) -> sv.Detections:
        """
        Adds segmentation masks to detections. Converts masks into a compatible format
        and integrates them with the detections object.

        Args:
            detections: Detections object to which masks will be added.

        Returns:
            sv.Detections: Updated detections with masks included.
        """
        if hasattr(detections, "mask") and detections.mask is not None:
            try:
                # Ensure masks are in the correct format
                masks = detections.mask
                if isinstance(masks, list) and len(masks) > 0:
                    # Stack masks if they're a list
                    if isinstance(masks[0], np.ndarray):
                        masks_array = np.array(masks)
                        if masks_array.ndim == 3:
                            detections = sv.Detections(
                                xyxy=detections.xyxy,
                                confidence=detections.confidence,
                                class_id=detections.class_id,
                                mask=masks_array,
                            )
            except Exception as e:
                # If mask processing fails, return original detections
                print(f"Warning: Could not process masks: {e}")

        return detections

    # Backward compatibility methods (deprecated)
    def segmenter(self) -> Optional[any]:
        """Deprecated: use get_segmenter() instead."""
        return self.get_segmenter()

    def verbose(self) -> bool:
        """Deprecated: access verbose attribute directly."""
        return self.verbose
