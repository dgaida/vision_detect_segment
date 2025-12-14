"""
Unit tests for batch_processor.py
Tests BatchConfig, BatchAccumulator, and BatchImageProcessor.
"""

import pytest
import numpy as np
import time
import torch
from unittest.mock import Mock, patch
from vision_detect_segment.utils.batch_processor import (
    BatchConfig,
    BatchAccumulator,
    BatchImageProcessor,
)


class TestBatchConfig:
    """Tests for BatchConfig dataclass."""

    def test_default_config(self):
        """Test default batch configuration."""
        config = BatchConfig()

        assert config.batch_size == 4
        assert config.max_wait_time == 0.1
        assert config.dynamic_batching is True
        assert config.pad_to_max_size is False
        assert config.max_batch_dimension == (1024, 1024)

    def test_custom_config(self):
        """Test custom batch configuration."""
        config = BatchConfig(
            batch_size=8, max_wait_time=0.2, dynamic_batching=False, pad_to_max_size=True, max_batch_dimension=(512, 512)
        )

        assert config.batch_size == 8
        assert config.max_wait_time == 0.2
        assert config.dynamic_batching is False
        assert config.pad_to_max_size is True
        assert config.max_batch_dimension == (512, 512)


class TestBatchAccumulator:
    """Tests for BatchAccumulator."""

    @pytest.fixture
    def accumulator(self):
        """Create BatchAccumulator with default config."""
        config = BatchConfig(batch_size=3, max_wait_time=0.5)
        return BatchAccumulator(config)

    def test_initialization(self, accumulator):
        """Test accumulator initialization."""
        assert len(accumulator.batch) == 0
        assert accumulator.batch_start_time is None

    def test_add_first_item(self, accumulator):
        """Test adding first item starts timer."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        metadata = {"frame_id": 1}

        result = accumulator.add(image, metadata)

        assert result is None  # Not ready yet
        assert len(accumulator.batch) == 1
        assert accumulator.batch_start_time is not None

    def test_batch_ready_by_size(self, accumulator):
        """Test batch ready when size threshold reached."""
        # Add items until batch is full
        for i in range(3):
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            metadata = {"frame_id": i}
            result = accumulator.add(image, metadata)

        # Last add should return the batch
        assert result is not None
        assert len(result) == 3

    def test_batch_ready_by_timeout(self):
        """Test batch ready when timeout reached."""
        config = BatchConfig(batch_size=10, max_wait_time=0.1)
        accumulator = BatchAccumulator(config)

        # Add one item
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = accumulator.add(image, {})

        assert result is None

        # Wait for timeout
        time.sleep(0.15)

        # Add another - should trigger timeout
        image2 = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = accumulator.add(image2, {})

        assert result is not None
        assert len(result) == 1  # First batch

    def test_incompatible_image_splits_batch(self, accumulator):
        """Test that incompatible images trigger batch split."""
        # Add compatible images
        image1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        accumulator.add(image1, {"id": 1})

        # Add very different size image
        image2 = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        result = accumulator.add(image2, {"id": 2})

        # Should return previous batch
        assert result is not None
        assert len(result) == 1
        # New batch started with incompatible image
        assert len(accumulator.batch) == 1

    def test_compatible_images_same_batch(self):
        """Test that similar-sized images stay in same batch."""
        config = BatchConfig(batch_size=5, pad_to_max_size=False)
        accumulator = BatchAccumulator(config)

        # Add similar-sized images
        for i in range(3):
            # Vary size slightly (within 10%)
            size = 100 + i * 5
            image = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            result = accumulator.add(image, {"id": i})

            if i < 2:
                assert result is None

        assert len(accumulator.batch) == 3

    def test_pad_to_max_size_always_compatible(self):
        """Test that padding makes all images compatible."""
        config = BatchConfig(batch_size=5, pad_to_max_size=True)
        accumulator = BatchAccumulator(config)

        # Add different sized images
        sizes = [50, 100, 150, 200]
        for i, size in enumerate(sizes):
            image = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            result = accumulator.add(image, {"id": i})

            assert result is None  # All compatible

        assert len(accumulator.batch) == 4

    def test_flush(self, accumulator):
        """Test manual flush."""
        # Add items
        for i in range(2):
            image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            accumulator.add(image, {"id": i})

        # Flush manually
        result = accumulator.flush()

        assert result is not None
        assert len(result) == 2
        assert len(accumulator.batch) == 0

    def test_flush_empty_batch(self, accumulator):
        """Test flushing empty batch."""
        result = accumulator.flush()
        assert result is None


class TestBatchImageProcessor:
    """Tests for BatchImageProcessor."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.return_value = Mock()
        return model

    @pytest.fixture
    def processor(self, mock_model):
        """Create BatchImageProcessor instance."""
        config = BatchConfig(batch_size=4, max_wait_time=0.1)
        return BatchImageProcessor(model=mock_model, config=config, device="cpu")

    def test_initialization(self, mock_model):
        """Test processor initialization."""
        config = BatchConfig(batch_size=8)
        processor = BatchImageProcessor(model=mock_model, config=config, device="cuda")

        assert processor.config.batch_size == 8
        assert processor.device == "cuda"
        assert processor.stats["batches_processed"] == 0

    def test_preprocess_simple(self, processor):
        """Test simple preprocessing without padding."""
        images = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
        ]

        batch_tensor, scales = processor._preprocess_simple(images)

        assert isinstance(batch_tensor, torch.Tensor)
        assert batch_tensor.shape[0] == 2  # Batch size
        assert len(scales) == 2
        assert scales[0] == (1.0, 1.0)

    def test_preprocess_with_padding(self, processor):
        """Test preprocessing with padding."""
        processor.config.pad_to_max_size = True
        processor.config.max_batch_dimension = (256, 256)

        images = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
        ]

        batch_tensor, scales = processor._preprocess_with_padding(images)

        assert batch_tensor.shape == (2, 3, 256, 256)
        assert len(scales) == 2
        # First image needs scaling
        assert scales[0][0] > 1.0

    def test_process_batch(self, processor):
        """Test processing a complete batch."""
        images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)]
        metadata_list = [{"id": i} for i in range(3)]

        results = processor.process_batch(images, metadata_list)

        assert len(results) == 3
        assert all("metadata" in r for r in results)
        assert processor.stats["batches_processed"] == 1
        assert processor.stats["total_images"] == 3

    def test_get_stats(self, processor):
        """Test getting statistics."""
        # Process some batches
        for _ in range(2):
            images = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)]
            processor.process_batch(images, [{}])

        stats = processor.get_stats()

        assert stats["batches_processed"] == 2
        assert stats["total_images"] == 2
        assert "avg_batch_time" in stats
        assert "throughput_fps" in stats

    def test_reset_stats(self, processor):
        """Test resetting statistics."""
        # Process a batch
        images = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)]
        processor.process_batch(images, [{}])

        assert processor.stats["batches_processed"] == 1

        # Reset
        processor.reset_stats()

        assert processor.stats["batches_processed"] == 0
        assert processor.stats["total_images"] == 0

    def test_flush_pending(self, processor):
        """Test flushing pending items."""
        # Add items to accumulator
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        processor.accumulator.add(image, {"id": 1})

        assert len(processor.accumulator.batch) == 1

        # Flush
        results = processor.flush()

        assert len(results) == 1
        assert len(processor.accumulator.batch) == 0

    def test_update_stats(self, processor):
        """Test statistics update."""
        initial_count = processor.stats["batches_processed"]

        processor._update_stats(batch_size=5, batch_time=0.5)

        assert processor.stats["batches_processed"] == initial_count + 1
        assert processor.stats["total_images"] == 5
        assert processor.stats["avg_batch_size"] == 5.0

    def test_process_with_gpu(self, mock_model):
        """Test processing with GPU device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = BatchConfig(batch_size=2)
        processor = BatchImageProcessor(model=mock_model, config=config, device="cuda")

        images = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)]
        results = processor.process_batch(images, [{}])

        assert len(results) == 1

    def test_batch_with_different_sizes(self, processor):
        """Test batching images of different sizes."""
        images = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8),
            np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8),
        ]

        # Should handle different sizes
        batch_tensor, scales = processor._preprocess_simple(images)

        assert batch_tensor.shape[0] == 3
        # Max height/width should be from largest image
        assert batch_tensor.shape[2] >= 120
        assert batch_tensor.shape[3] >= 120


class TestBatchProcessingWorkflow:
    """Integration tests for complete batch processing workflow."""

    def test_accumulate_and_process(self):
        """Test complete accumulate -> process workflow."""
        config = BatchConfig(batch_size=3, max_wait_time=1.0)

        mock_model = Mock()
        processor = BatchImageProcessor(model=mock_model, config=config, device="cpu")

        # Add items one by one
        results_list = []

        for i in range(5):
            image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            metadata = {"frame": i}

            result, batch = processor.process_single(image, metadata)

            if result:
                results_list.extend(result)

        # Flush remaining
        remaining = processor.flush()
        if remaining:
            results_list.extend(remaining)

        # Should have processed all 5 images
        assert processor.stats["total_images"] >= 3  # At least one full batch

    def test_priority_batching(self):
        """Test that batches are formed correctly with priorities."""
        config = BatchConfig(batch_size=4)
        accumulator = BatchAccumulator(config)

        # Add items with similar characteristics
        for i in range(4):
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            metadata = {"priority": i}
            result = accumulator.add(image, metadata)

        # Should get batch when full
        assert result is not None
        assert len(result) == 4

    def test_error_handling_in_batch(self):
        """Test error handling when processing batch fails."""
        mock_model = Mock()
        config = BatchConfig(batch_size=2)

        processor = BatchImageProcessor(model=mock_model, config=config, device="cpu")

        # Make preprocessing fail
        with patch.object(processor, "_preprocess_batch", side_effect=Exception("Preprocessing failed")):
            images = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)]
            results = processor.process_batch(images, [{}])

            # Should return error results
            assert len(results) == 1
            assert "error" in results[0]


class TestBatchAccumulatorEdgeCases:
    """Test edge cases for BatchAccumulator."""

    def test_empty_batch_operations(self):
        """Test operations on empty batch."""
        config = BatchConfig()
        accumulator = BatchAccumulator(config)

        # Flush empty batch
        result = accumulator.flush()
        assert result is None

        # Check if ready when empty
        assert accumulator._is_ready() is False

    def test_grayscale_image_compatibility(self):
        """Test compatibility check with grayscale images."""
        config = BatchConfig()
        accumulator = BatchAccumulator(config)

        # Add color image
        color_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        accumulator.add(color_img, {})

        # Try to add grayscale
        gray_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = accumulator.add(gray_img, {})

        # Should split batch
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
