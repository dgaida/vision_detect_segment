"""
Unit tests for async_processor.py
Tests AsyncProcessor, BackpressureHandler, and ProcessingTask/ProcessingResult.
"""

import pytest
import numpy as np
import time
from vision_detect_segment.core.async_processor import (
    AsyncProcessor,
    BackpressureHandler,
    ProcessingTask,
    ProcessingResult,
    ProcessingState,
)


class TestProcessingTask:
    """Tests for ProcessingTask dataclass."""

    def test_task_creation(self):
        """Test creating a processing task."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        metadata = {"frame_id": 1}

        task = ProcessingTask(image=image, metadata=metadata, task_id="test_task", priority=5)

        assert task.task_id == "test_task"
        assert task.priority == 5
        assert np.array_equal(task.image, image)
        assert task.metadata == metadata
        assert task.timestamp > 0

    def test_task_priority_ordering(self):
        """Test that tasks are ordered by priority (higher first)."""
        task1 = ProcessingTask(image=np.zeros((10, 10, 3)), metadata={}, task_id="task1", priority=5)
        task2 = ProcessingTask(image=np.zeros((10, 10, 3)), metadata={}, task_id="task2", priority=10)

        # Higher priority should be "less than" (comes first)
        assert task2 < task1


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def test_result_creation_success(self):
        """Test creating a successful result."""
        result = ProcessingResult(
            task_id="test_task",
            detections=[{"label": "test"}],
            annotated_image=np.zeros((100, 100, 3)),
            processing_time=0.5,
            success=True,
            metadata={"frame_id": 1},
        )

        assert result.task_id == "test_task"
        assert result.success is True
        assert len(result.detections) == 1
        assert result.processing_time == 0.5
        assert result.error is None

    def test_result_creation_failure(self):
        """Test creating a failed result."""
        result = ProcessingResult(
            task_id="failed_task",
            detections=[],
            annotated_image=None,
            processing_time=0.1,
            success=False,
            error="Processing failed",
        )

        assert result.success is False
        assert result.error == "Processing failed"
        assert len(result.detections) == 0


class TestBackpressureHandler:
    """Tests for BackpressureHandler."""

    def test_initialization(self):
        """Test backpressure handler initialization."""
        handler = BackpressureHandler(max_queue_size=100, drop_threshold=0.8, frame_skip_threshold=0.9)

        assert handler.max_queue_size == 100
        assert handler.drop_threshold == 80
        assert handler.frame_skip_threshold == 90
        assert handler.dropped_frames == 0
        assert handler.skipped_frames == 0

    def test_accept_frame_low_pressure(self):
        """Test accepting frames under low pressure."""
        handler = BackpressureHandler(max_queue_size=100)

        # Low queue size - should accept
        assert handler.should_accept_frame(queue_size=50, priority=0) is True
        assert handler.dropped_frames == 0
        assert handler.skipped_frames == 0

    def test_reject_frame_high_pressure_low_priority(self):
        """Test rejecting low priority frames under high pressure."""
        handler = BackpressureHandler(max_queue_size=100)

        # High queue size with low priority - should skip
        assert handler.should_accept_frame(queue_size=95, priority=0) is False
        assert handler.skipped_frames == 1

    def test_accept_high_priority_under_pressure(self):
        """Test accepting high priority frames even under pressure."""
        handler = BackpressureHandler(max_queue_size=100)

        # High queue size but high priority - should accept
        assert handler.should_accept_frame(queue_size=95, priority=10) is True
        assert handler.skipped_frames == 0

    def test_drop_medium_pressure_low_priority(self):
        """Test dropping frames at medium pressure."""
        handler = BackpressureHandler(max_queue_size=100)

        # Medium pressure, low priority - should drop
        assert handler.should_accept_frame(queue_size=85, priority=0) is False
        assert handler.dropped_frames == 1

    def test_get_stats(self):
        """Test getting backpressure statistics."""
        handler = BackpressureHandler(max_queue_size=100)

        handler.dropped_frames = 5
        handler.skipped_frames = 10
        handler.processed_frames = 100

        stats = handler.get_stats()

        assert stats["dropped_frames"] == 5
        assert stats["skipped_frames"] == 10
        assert stats["processed_frames"] == 100
        assert "drop_rate" in stats
        assert "uptime_seconds" in stats

    def test_reset_stats(self):
        """Test resetting statistics."""
        handler = BackpressureHandler(max_queue_size=100)

        handler.dropped_frames = 10
        handler.skipped_frames = 20
        handler.processed_frames = 50

        handler.reset_stats()

        assert handler.dropped_frames == 0
        assert handler.skipped_frames == 0
        assert handler.processed_frames == 0


class TestAsyncProcessor:
    """Tests for AsyncProcessor."""

    @pytest.fixture
    def mock_process_func(self):
        """Create a mock processing function."""

        def process(image, metadata):
            time.sleep(0.01)  # Simulate processing
            return {"objects": [{"label": "test"}]}

        return process

    @pytest.fixture
    def processor(self, mock_process_func):
        """Create AsyncProcessor instance."""
        proc = AsyncProcessor(process_func=mock_process_func, num_workers=2, max_queue_size=10, enable_backpressure=True)
        yield proc
        # Cleanup
        if proc.state == ProcessingState.RUNNING:
            proc.stop(timeout=1.0)

    def test_initialization(self, mock_process_func):
        """Test processor initialization."""
        processor = AsyncProcessor(
            process_func=mock_process_func, num_workers=2, max_queue_size=50, batch_size=1, enable_backpressure=True
        )

        assert processor.num_workers == 2
        assert processor.batch_size == 1
        assert processor.backpressure is not None
        assert processor.state == ProcessingState.IDLE

    def test_start_stop(self, processor):
        """Test starting and stopping the processor."""
        assert processor.state == ProcessingState.IDLE

        processor.start()
        assert processor.state == ProcessingState.RUNNING
        assert len(processor.workers) == 2

        processor.stop(timeout=2.0)
        assert processor.state == ProcessingState.STOPPED

    def test_submit_task_when_running(self, processor):
        """Test submitting a task when processor is running."""
        processor.start()

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        metadata = {"frame_id": 1}

        accepted = processor.submit_task(image=image, metadata=metadata, task_id="test_1", priority=5)

        assert accepted is True

        processor.stop(timeout=2.0)

    def test_submit_task_when_stopped(self, processor):
        """Test that submitting when stopped returns False."""
        # Processor is not started
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        accepted = processor.submit_task(image=image, metadata={}, task_id="test_2")

        assert accepted is False

    def test_get_result(self, processor):
        """Test getting processing results."""
        processor.start()

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        processor.submit_task(image=image, metadata={}, task_id="test_3")

        # Wait a bit for processing
        time.sleep(0.5)

        result = processor.get_result(timeout=1.0)

        if result:
            assert isinstance(result, ProcessingResult)
            assert result.task_id == "test_3"

        processor.stop(timeout=2.0)

    def test_backpressure_drops_frames(self, mock_process_func):
        """Test that backpressure drops frames when queue is full."""
        processor = AsyncProcessor(process_func=mock_process_func, num_workers=1, max_queue_size=5, enable_backpressure=True)

        processor.start()

        # Submit many tasks rapidly
        accepted_count = 0
        for i in range(20):
            image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
            if processor.submit_task(image=image, metadata={}, task_id=f"task_{i}"):
                accepted_count += 1

        # Some should be dropped due to backpressure
        assert accepted_count < 20

        processor.stop(timeout=2.0)

    def test_process_batch(self, processor):
        """Test batch processing."""
        processor.start()

        # Submit multiple tasks
        for i in range(3):
            image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            processor.submit_task(image=image, metadata={"id": i}, task_id=f"batch_{i}")

        # Wait for processing
        time.sleep(0.5)

        # Get results
        results = []
        for _ in range(3):
            result = processor.get_result(timeout=0.5)
            if result:
                results.append(result)

        assert len(results) > 0

        processor.stop(timeout=2.0)

    def test_get_stats(self, processor):
        """Test getting processing statistics."""
        processor.start()

        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        processor.submit_task(image=image, metadata={}, task_id="stats_test")

        time.sleep(0.5)

        stats = processor.get_stats()

        assert "tasks_submitted" in stats
        assert "tasks_completed" in stats
        assert "state" in stats
        assert stats["state"] == ProcessingState.RUNNING.value
        assert stats["tasks_submitted"] >= 1

        processor.stop(timeout=2.0)

    def test_worker_handles_errors(self, processor):
        """Test that workers handle processing errors gracefully."""

        # Create processor with failing function
        def failing_func(image, metadata):
            raise ValueError("Simulated processing error")

        error_processor = AsyncProcessor(process_func=failing_func, num_workers=1, max_queue_size=5, enable_backpressure=False)

        error_processor.start()

        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        error_processor.submit_task(image=image, metadata={}, task_id="error_test")

        time.sleep(0.5)

        result = error_processor.get_result(timeout=1.0)

        if result:
            assert result.success is False
            assert result.error is not None

        error_processor.stop(timeout=2.0)

    def test_clear_queues(self, processor):
        """Test clearing input and result queues."""
        processor.start()

        # Submit some tasks
        for i in range(3):
            image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
            processor.submit_task(image=image, metadata={}, task_id=f"clear_{i}")

        # Clear queues
        processor.clear_queues()

        assert processor.input_queue.qsize() == 0
        assert processor.result_queue.qsize() == 0

        processor.stop(timeout=2.0)

    def test_graceful_shutdown_with_pending_tasks(self, mock_process_func):
        """Test graceful shutdown with tasks still in queue."""
        processor = AsyncProcessor(process_func=mock_process_func, num_workers=1, max_queue_size=50)

        processor.start()

        # Submit many tasks
        for i in range(10):
            image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
            processor.submit_task(image=image, metadata={}, task_id=f"shutdown_{i}")

        # Stop immediately (some tasks still pending)
        processor.stop(timeout=3.0)

        assert processor.state == ProcessingState.STOPPED


class TestAsyncProcessorStatistics:
    """Tests for statistics tracking."""

    def test_throughput_calculation(self):
        """Test throughput FPS calculation."""

        def fast_func(image, metadata):
            return {"objects": []}

        processor = AsyncProcessor(process_func=fast_func, num_workers=2, max_queue_size=50)

        processor.start()

        # Submit and process tasks
        for i in range(10):
            image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
            processor.submit_task(image=image, metadata={}, task_id=f"fps_{i}")

        time.sleep(1.0)

        stats = processor.get_stats()

        if stats["tasks_completed"] > 0:
            assert "throughput_fps" in stats
            assert stats["throughput_fps"] > 0

        processor.stop(timeout=2.0)

    def test_average_processing_time(self):
        """Test average processing time calculation."""

        def timed_func(image, metadata):
            time.sleep(0.05)
            return {"objects": []}

        processor = AsyncProcessor(process_func=timed_func, num_workers=1, max_queue_size=10)

        processor.start()

        # Submit tasks
        for i in range(5):
            image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
            processor.submit_task(image=image, metadata={}, task_id=f"time_{i}")

        time.sleep(1.0)

        stats = processor.get_stats()

        if stats["tasks_completed"] > 0:
            assert "avg_processing_time" in stats
            assert stats["avg_processing_time"] > 0

        processor.stop(timeout=2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
