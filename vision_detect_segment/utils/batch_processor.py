"""
vision_detect_segment/utils/batch_processor.py

Batch processing utilities for high-throughput detection scenarios.
Optimizes GPU utilization through batched inference.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import logging


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    batch_size: int = 4
    max_wait_time: float = 0.1  # seconds
    dynamic_batching: bool = True
    pad_to_max_size: bool = False
    max_batch_dimension: Tuple[int, int] = (1024, 1024)


class BatchAccumulator:
    """
    Accumulates items for batch processing with timeout.

    Features:
    - Dynamic batch sizing based on load
    - Timeout-based flushing
    - Image dimension compatibility checking
    """

    def __init__(self, config: BatchConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        self.batch: List[Tuple[np.ndarray, Dict[str, Any]]] = []
        self.batch_start_time: Optional[float] = None

    def add(self, image: np.ndarray, metadata: Dict[str, Any]) -> Optional[List[Tuple[np.ndarray, Dict[str, Any]]]]:
        """
        Add item to batch. Returns batch if ready, None otherwise.

        Args:
            image: Input image
            metadata: Image metadata

        Returns:
            Completed batch or None
        """
        # Start timer on first item
        if not self.batch:
            self.batch_start_time = time.time()

        # Check if image is compatible with current batch
        if self.batch and not self._is_compatible(image):
            # Return current batch and start new one
            ready_batch = self.batch
            self.batch = [(image, metadata)]
            self.batch_start_time = time.time()
            return ready_batch

        # Add to batch
        self.batch.append((image, metadata))

        # Check if batch is ready
        if self._is_ready():
            return self.flush()

        return None

    def flush(self) -> Optional[List[Tuple[np.ndarray, Dict[str, Any]]]]:
        """Force flush current batch."""
        if not self.batch:
            return None

        ready_batch = self.batch
        self.batch = []
        self.batch_start_time = None
        return ready_batch

    def _is_compatible(self, image: np.ndarray) -> bool:
        """Check if image is compatible with current batch."""
        if not self.batch:
            return True

        # For simplicity, check if dimensions are similar
        # In production, you might want more sophisticated logic
        first_img = self.batch[0][0]

        # Allow some size variation (within 10%)
        h1, w1 = first_img.shape[:2]
        h2, w2 = image.shape[:2]

        if self.config.pad_to_max_size:
            # All images will be padded, so always compatible
            return True

        # Check if dimensions are reasonably close
        h_diff = abs(h1 - h2) / max(h1, h2)
        w_diff = abs(w1 - w2) / max(w1, w2)

        return h_diff < 0.1 and w_diff < 0.1

    def _is_ready(self) -> bool:
        """Check if batch should be processed."""
        # Size-based trigger
        if len(self.batch) >= self.config.batch_size:
            return True

        # Time-based trigger
        if self.batch_start_time:
            elapsed = time.time() - self.batch_start_time
            if elapsed >= self.config.max_wait_time:
                return True

        return False


class BatchImageProcessor:
    """
    Processes images in batches for improved GPU utilization.

    Handles:
    - Batch accumulation
    - Image preprocessing
    - Batched inference
    - Result unpacking
    """

    def __init__(
        self,
        model,
        config: Optional[BatchConfig] = None,
        device: str = "cuda",
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.config = config or BatchConfig()
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        self.accumulator = BatchAccumulator(config, logger)

        # Statistics
        self.stats = {
            "batches_processed": 0,
            "total_images": 0,
            "total_batch_time": 0.0,
            "avg_batch_size": 0.0,
        }

    def process_single(
        self, image: np.ndarray, metadata: Dict[str, Any]
    ) -> Tuple[List[Dict], Optional[List[Tuple[np.ndarray, Dict[str, Any]]]]]:
        """
        Process single image (may trigger batch processing).

        Args:
            image: Input image
            metadata: Image metadata

        Returns:
            Tuple of (individual_result, remaining_batch_items)
        """
        # Add to accumulator
        batch = self.accumulator.add(image, metadata)

        if batch:
            # Process batch
            results = self._process_batch(batch)
            return results, None

        return [], None

    def process_batch(self, images: List[np.ndarray], metadata_list: List[Dict[str, Any]]) -> List[Dict]:
        """
        Process a complete batch of images.

        Args:
            images: List of input images
            metadata_list: List of metadata dicts

        Returns:
            List of detection results
        """
        batch = list(zip(images, metadata_list))
        return self._process_batch(batch)

    def _process_batch(self, batch: List[Tuple[np.ndarray, Dict[str, Any]]]) -> List[Dict]:
        """Internal batch processing."""
        start_time = time.time()

        try:
            # Preprocess batch
            images_tensor, scales = self._preprocess_batch([item[0] for item in batch])

            # Run inference
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        batch_results = self._run_inference(images_tensor)
                else:
                    batch_results = self._run_inference(images_tensor)

            # Postprocess results
            results = self._postprocess_batch(batch_results, batch, scales)

            # Update statistics
            batch_time = time.time() - start_time
            self._update_stats(len(batch), batch_time)

            if self.logger:
                self.logger.debug(
                    f"Processed batch of {len(batch)} images in "
                    f"{batch_time*1000:.1f}ms "
                    f"({len(batch)/batch_time:.1f} FPS)"
                )

            return results

        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            # Return empty results for failed batch
            return [{"objects": [], "error": str(e)} for _ in batch]

    def _preprocess_batch(self, images: List[np.ndarray]) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
        """
        Preprocess images for batched inference.

        Returns:
            Tuple of (batched_tensor, scale_factors)
        """
        if self.config.pad_to_max_size:
            # Pad all images to same size
            return self._preprocess_with_padding(images)
        else:
            # Simple stacking (assumes similar sizes)
            return self._preprocess_simple(images)

    def _preprocess_simple(self, images: List[np.ndarray]) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
        """Simple preprocessing without padding."""
        # Find max dimensions in batch
        max_h = max(img.shape[0] for img in images)
        max_w = max(img.shape[1] for img in images)

        batch_size = len(images)
        channels = images[0].shape[2] if len(images[0].shape) == 3 else 1

        # Create batch tensor
        batch_tensor = torch.zeros((batch_size, channels, max_h, max_w), dtype=torch.float32, device=self.device)

        scales = []

        for i, img in enumerate(images):
            h, w = img.shape[:2]

            # Convert to tensor and normalize
            img_tensor = torch.from_numpy(img).float() / 255.0

            if len(img.shape) == 3:
                img_tensor = img_tensor.permute(2, 0, 1)

            # Place in batch tensor
            batch_tensor[i, :, :h, :w] = img_tensor.to(self.device)

            # Store original dimensions for scaling back
            scales.append((1.0, 1.0))

        return batch_tensor, scales

    def _preprocess_with_padding(self, images: List[np.ndarray]) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
        """Preprocessing with padding to target size."""
        target_h, target_w = self.config.max_batch_dimension
        batch_size = len(images)
        channels = images[0].shape[2] if len(images[0].shape) == 3 else 1

        batch_tensor = torch.zeros((batch_size, channels, target_h, target_w), dtype=torch.float32, device=self.device)

        scales = []

        for i, img in enumerate(images):
            h, w = img.shape[:2]

            # Calculate scale to fit in target
            scale = min(target_h / h, target_w / w)
            new_h, new_w = int(h * scale), int(w * scale)

            # Resize if needed
            if scale != 1.0:
                import cv2

                img = cv2.resize(img, (new_w, new_h))

            # Convert and normalize
            img_tensor = torch.from_numpy(img).float() / 255.0
            if len(img.shape) == 3:
                img_tensor = img_tensor.permute(2, 0, 1)

            # Place in batch tensor (top-left)
            batch_tensor[i, :, :new_h, :new_w] = img_tensor.to(self.device)

            scales.append((scale, scale))

        return batch_tensor, scales

    def _run_inference(self, images_tensor: torch.Tensor) -> Any:
        """
        Run model inference on batch.
        Override this for specific model types.
        """
        # This is a placeholder - override in subclass
        return self.model(images_tensor)

    def _postprocess_batch(
        self, batch_results: Any, batch: List[Tuple[np.ndarray, Dict[str, Any]]], scales: List[Tuple[float, float]]
    ) -> List[Dict]:
        """
        Postprocess batch results.
        Override this for specific model types.
        """
        # Placeholder - override in subclass
        results = []
        for i, (image, metadata) in enumerate(batch):
            results.append(
                {
                    "objects": [],
                    "metadata": metadata,
                    "image_shape": image.shape,
                }
            )
        return results

    def _update_stats(self, batch_size: int, batch_time: float):
        """Update processing statistics."""
        self.stats["batches_processed"] += 1
        self.stats["total_images"] += batch_size
        self.stats["total_batch_time"] += batch_time

        # Calculate running average batch size
        total_batches = self.stats["batches_processed"]
        self.stats["avg_batch_size"] = self.stats["total_images"] / total_batches

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()

        if stats["batches_processed"] > 0:
            stats["avg_batch_time"] = stats["total_batch_time"] / stats["batches_processed"]
            stats["throughput_fps"] = stats["total_images"] / stats["total_batch_time"]

        return stats

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "batches_processed": 0,
            "total_images": 0,
            "total_batch_time": 0.0,
            "avg_batch_size": 0.0,
        }

    def flush(self) -> List[Dict]:
        """Flush any pending items in accumulator."""
        batch = self.accumulator.flush()
        if batch:
            return self._process_batch(batch)
        return []
