"""
vision_detect_segment/core/async_processor.py

Asynchronous processing module for vision detection with proper queue management,
backpressure handling, and worker thread coordination.
"""

import threading
import queue
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor


class ProcessingState(Enum):
    """Processing pipeline states."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class ProcessingTask:
    """Container for a processing task."""

    image: np.ndarray
    metadata: Dict[str, Any]
    task_id: str
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # Higher = more important

    def __lt__(self, other):
        """Support priority queue ordering."""
        return self.priority > other.priority


@dataclass
class ProcessingResult:
    """Container for processing results."""

    task_id: str
    detections: List[Dict]
    annotated_image: Optional[np.ndarray]
    processing_time: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackpressureHandler:
    """
    Handles backpressure when processing falls behind.

    Implements adaptive strategies:
    - Drop oldest frames when queue is full
    - Reduce frame rate dynamically
    - Skip frames based on priority
    """

    def __init__(
        self,
        max_queue_size: int = 100,
        drop_threshold: float = 0.8,  # Drop frames when 80% full
        frame_skip_threshold: float = 0.9,  # Skip frames when 90% full
    ):
        self.max_queue_size = max_queue_size
        self.drop_threshold = int(max_queue_size * drop_threshold)
        self.frame_skip_threshold = int(max_queue_size * frame_skip_threshold)

        # Statistics
        self.dropped_frames = 0
        self.skipped_frames = 0
        self.processed_frames = 0
        self.last_reset = time.time()

    def should_accept_frame(self, queue_size: int, priority: int = 0) -> bool:
        """Determine if a new frame should be accepted."""
        if queue_size < self.drop_threshold:
            return True

        if queue_size >= self.frame_skip_threshold:
            # Only accept high priority frames
            if priority >= 5:
                return True
            self.skipped_frames += 1
            return False

        # Medium backpressure - drop low priority frames
        if priority < 2:
            self.dropped_frames += 1
            return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get backpressure statistics."""
        uptime = time.time() - self.last_reset
        return {
            "dropped_frames": self.dropped_frames,
            "skipped_frames": self.skipped_frames,
            "processed_frames": self.processed_frames,
            "drop_rate": self.dropped_frames / max(1, self.processed_frames),
            "skip_rate": self.skipped_frames / max(1, self.processed_frames),
            "uptime_seconds": uptime,
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self.dropped_frames = 0
        self.skipped_frames = 0
        self.processed_frames = 0
        self.last_reset = time.time()


class AsyncProcessor:
    """
    Asynchronous processor with worker threads and queue management.

    Features:
    - Non-blocking image processing
    - Configurable worker threads
    - Backpressure handling
    - Batch processing support
    - Graceful shutdown
    """

    def __init__(
        self,
        process_func: Callable,
        num_workers: int = 2,
        max_queue_size: int = 100,
        batch_size: int = 1,
        enable_backpressure: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize async processor.

        Args:
            process_func: Function to process tasks (image, metadata) -> result
            num_workers: Number of worker threads
            max_queue_size: Maximum queue size
            batch_size: Number of items to process in batch (1 = no batching)
            enable_backpressure: Enable adaptive backpressure handling
            logger: Optional logger instance
        """
        self.process_func = process_func
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(__name__)

        # Task queues
        self.input_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)
        self.result_queue: queue.Queue = queue.Queue()

        # Backpressure handling
        self.backpressure = BackpressureHandler(max_queue_size) if enable_backpressure else None

        # Worker management
        self.workers: List[threading.Thread] = []
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.state = ProcessingState.IDLE
        self.state_lock = threading.Lock()

        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_processing_time": 0.0,
            "start_time": None,
        }
        self.stats_lock = threading.Lock()

    def start(self):
        """Start the processing workers."""
        with self.state_lock:
            if self.state == ProcessingState.RUNNING:
                self.logger.warning("Processor already running")
                return

            self.state = ProcessingState.RUNNING
            self.stats["start_time"] = time.time()

            # Start worker threads
            for i in range(self.num_workers):
                worker = threading.Thread(target=self._worker_loop, name=f"AsyncWorker-{i}", daemon=True)
                worker.start()
                self.workers.append(worker)

            self.logger.info(f"Started {self.num_workers} worker threads")

    def stop(self, timeout: float = 5.0):
        """
        Stop the processor gracefully.

        Args:
            timeout: Maximum time to wait for workers to finish
        """
        with self.state_lock:
            if self.state in [ProcessingState.STOPPED, ProcessingState.STOPPING]:
                return

            self.logger.info("Stopping processor...")
            self.state = ProcessingState.STOPPING

        # Wait for queue to drain or timeout
        deadline = time.time() + timeout
        while not self.input_queue.empty() and time.time() < deadline:
            time.sleep(0.1)

        # Signal workers to stop by putting None sentinels
        for _ in range(self.num_workers):
            try:
                self.input_queue.put(None, timeout=1.0)
            except queue.Full:
                pass

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=2.0)

        with self.state_lock:
            self.state = ProcessingState.STOPPED

        self.logger.info("Processor stopped")

    def submit_task(
        self,
        image: np.ndarray,
        metadata: Dict[str, Any],
        task_id: Optional[str] = None,
        priority: int = 0,
        timeout: float = 0.1,
    ) -> bool:
        """
        Submit a task for processing.

        Args:
            image: Input image
            metadata: Task metadata
            task_id: Optional task identifier
            priority: Task priority (higher = more important)
            timeout: Max time to wait if queue is full

        Returns:
            bool: True if task was accepted, False if dropped
        """
        if self.state != ProcessingState.RUNNING:
            self.logger.warning(f"Cannot submit task - processor state: {self.state}")
            return False

        # Check backpressure
        queue_size = self.input_queue.qsize()
        if self.backpressure and not self.backpressure.should_accept_frame(queue_size, priority):
            return False

        # Create task
        if task_id is None:
            task_id = f"task_{time.time()}_{id(image)}"

        task = ProcessingTask(
            image=image,
            metadata=metadata,
            task_id=task_id,
            priority=priority,
        )

        # Submit to queue
        try:
            self.input_queue.put(task, timeout=timeout)
            with self.stats_lock:
                self.stats["tasks_submitted"] += 1
            return True
        except queue.Full:
            self.logger.warning(f"Queue full, dropped task {task_id}")
            if self.backpressure:
                self.backpressure.dropped_frames += 1
            return False

    def get_result(self, timeout: float = 0.1) -> Optional[ProcessingResult]:
        """
        Get a processing result if available.

        Args:
            timeout: Max time to wait for result

        Returns:
            ProcessingResult or None if no result available
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _worker_loop(self):
        """Main worker loop - runs in separate thread."""
        worker_name = threading.current_thread().name
        self.logger.debug(f"{worker_name} started")

        batch: List[ProcessingTask] = []

        while self.state == ProcessingState.RUNNING:
            try:
                # Get task from queue
                task = self.input_queue.get(timeout=0.5)

                # None is sentinel for shutdown
                if task is None:
                    break

                batch.append(task)

                # Process batch when full or queue is empty
                if len(batch) >= self.batch_size or self.input_queue.empty():
                    self._process_batch(batch)
                    batch.clear()

            except queue.Empty:
                # Process any remaining batch items
                if batch:
                    self._process_batch(batch)
                    batch.clear()
                continue
            except Exception as e:
                self.logger.error(f"{worker_name} error: {e}", exc_info=True)

        # Process final batch
        if batch:
            self._process_batch(batch)

        self.logger.debug(f"{worker_name} stopped")

    def _process_batch(self, batch: List[ProcessingTask]):
        """Process a batch of tasks."""
        for task in batch:
            start_time = time.time()

            try:
                # Call the processing function
                detections = self.process_func(task.image, task.metadata)

                # Create result
                result = ProcessingResult(
                    task_id=task.task_id,
                    detections=detections.get("objects", []) if isinstance(detections, dict) else detections,
                    annotated_image=detections.get("annotated_image") if isinstance(detections, dict) else None,
                    processing_time=time.time() - start_time,
                    success=True,
                    metadata=task.metadata,
                )

                # Update stats
                with self.stats_lock:
                    self.stats["tasks_completed"] += 1
                    self.stats["total_processing_time"] += result.processing_time

                if self.backpressure:
                    self.backpressure.processed_frames += 1

            except Exception as e:
                # Create error result
                result = ProcessingResult(
                    task_id=task.task_id,
                    detections=[],
                    annotated_image=None,
                    processing_time=time.time() - start_time,
                    success=False,
                    error=str(e),
                    metadata=task.metadata,
                )

                with self.stats_lock:
                    self.stats["tasks_failed"] += 1

                self.logger.error(f"Task {task.task_id} failed: {e}")

            # Put result in output queue
            try:
                self.result_queue.put(result, timeout=1.0)
            except queue.Full:
                self.logger.warning(f"Result queue full, dropped result {task.task_id}")

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        with self.stats_lock:
            stats = self.stats.copy()

        # Calculate derived metrics
        if stats["start_time"]:
            uptime = time.time() - stats["start_time"]
            stats["uptime_seconds"] = uptime
            stats["throughput_fps"] = stats["tasks_completed"] / max(1, uptime)

            if stats["tasks_completed"] > 0:
                stats["avg_processing_time"] = stats["total_processing_time"] / stats["tasks_completed"]

        stats["queue_size"] = self.input_queue.qsize()
        stats["result_queue_size"] = self.result_queue.qsize()
        stats["state"] = self.state.value

        if self.backpressure:
            stats["backpressure"] = self.backpressure.get_stats()

        return stats

    def clear_queues(self):
        """Clear all queues (useful for testing or reset)."""
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break

        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
