#!/usr/bin/env python3
"""
scripts/detect_objects_async.py

Async detection script with background processing, proper error handling,
and real-time statistics display.
"""

import signal
import sys
import time
from typing import Optional

from vision_detect_segment import get_default_config
from vision_detect_segment.core.visualcortex_async import VisualCortexAsync
from vision_detect_segment.utils.exceptions import RedisConnectionError


class AsyncDetectionService:
    """
    Service wrapper for async object detection.

    Handles:
    - Graceful startup and shutdown
    - Signal handling (Ctrl+C)
    - Statistics reporting
    - Error recovery
    """

    def __init__(
        self,
        model_id: str = "owlv2",
        num_workers: int = 2,
        verbose: bool = False,
    ):
        self.model_id = model_id
        self.num_workers = num_workers
        self.verbose = verbose
        self.cortex: Optional[VisualCortexAsync] = None
        self.running = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n🛑 Shutdown signal received...")
        self.stop()
        sys.exit(0)

    def start(self) -> bool:
        """Start the detection service."""
        print("=" * 70)
        print("Async Object Detection Service - Starting...")
        print("=" * 70)

        # Load configuration
        try:
            config = get_default_config(self.model_id)
            config.enable_label_monitoring = True
            config.verbose = self.verbose
        except Exception as e:
            print(f"\n❌ Configuration error: {e}")
            return False

        # Initialize VisualCortex
        try:
            print(f"\n📦 Initializing {self.model_id} with {self.num_workers} workers...")

            self.cortex = VisualCortexAsync(
                objdetect_model_id=self.model_id,
                device="auto",
                stream_name="robot_camera",
                annotated_stream_name="annotated_camera",
                publish_annotated=True,
                verbose=self.verbose,
                config=config,
                num_workers=self.num_workers,
                max_queue_size=100,
                enable_backpressure=True,
                redis_poll_interval=0.01,
            )

            print("✅ VisualCortex initialized")

        except RedisConnectionError as e:
            print(f"\n❌ Redis connection failed: {e}")
            print("\n💡 Make sure Redis is running:")
            print("   docker run -p 6379:6379 redis:alpine")
            return False
        except Exception as e:
            print(f"\n❌ Initialization failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        # Start async processing
        try:
            print("\n🚀 Starting async processing pipeline...")
            self.cortex.start()
            self.running = True
            print("✅ Pipeline started successfully")
        except Exception as e:
            print(f"\n❌ Failed to start pipeline: {e}")
            return False

        # Print service info
        print("\n" + "=" * 70)
        print("Async Object Detection Service - Running")
        print("=" * 70)
        print(f"Model:           {self.model_id}")
        print(f"Workers:         {self.num_workers}")
        print("Input stream:    robot_camera")
        print("Output stream:   annotated_camera")
        print("Detection feed:  detected_objects")
        print("\nPress Ctrl+C to stop")
        print("=" * 70 + "\n")

        return True

    def stop(self):
        """Stop the detection service."""
        if not self.running:
            return

        print("\n🛑 Stopping detection service...")
        self.running = False

        if self.cortex:
            try:
                # Get final stats
                final_stats = self.cortex.get_stats()

                # Stop processing
                self.cortex.stop(timeout=5.0)

                # Cleanup
                self.cortex.cleanup()

                # Print summary
                print("\n" + "=" * 70)
                print("Service Summary")
                print("=" * 70)
                self._print_stats(final_stats)
                print("=" * 70)

            except Exception as e:
                print(f"⚠️  Error during shutdown: {e}")

        print("✅ Service stopped")

    def run(self, report_interval: float = 5.0):
        """
        Run the service with periodic statistics reporting.

        Args:
            report_interval: Seconds between stat reports
        """
        if not self.start():
            return

        last_report = time.time()

        try:
            while self.running:
                # Check if it's time to report stats
                now = time.time()
                if now - last_report >= report_interval:
                    self._report_stats()
                    last_report = now

                # Sleep briefly
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n⚠️  Keyboard interrupt")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.stop()

    def _report_stats(self):
        """Print current statistics."""
        if not self.cortex:
            return

        try:
            stats = self.cortex.get_stats()

            print("\n" + "─" * 70)
            print("📊 Service Statistics")
            print("─" * 70)
            self._print_stats(stats)
            print("─" * 70 + "\n")

        except Exception as e:
            print(f"⚠️  Failed to get stats: {e}")

    def _print_stats(self, stats: dict):
        """Pretty print statistics."""
        # Processing stats
        print(f"Uptime:          {stats.get('uptime_seconds', 0):.1f}s")
        print(f"Tasks submitted: {stats.get('tasks_submitted', 0)}")
        print(f"Tasks completed: {stats.get('tasks_completed', 0)}")
        print(f"Tasks failed:    {stats.get('tasks_failed', 0)}")

        # Performance
        throughput = stats.get("throughput_fps", 0)
        avg_time = stats.get("avg_processing_time", 0)
        print(f"Throughput:      {throughput:.2f} FPS")
        print(f"Avg proc time:   {avg_time*1000:.1f}ms")

        # Queue status
        queue_size = stats.get("queue_size", 0)
        print(f"Queue size:      {queue_size}")

        # Current state
        detections = stats.get("current_detections", 0)
        print(f"Current dets:    {detections}")

        # Backpressure stats
        if "backpressure" in stats:
            bp = stats["backpressure"]
            print("\nBackpressure:")
            print(f"  Dropped:       {bp.get('dropped_frames', 0)}")
            print(f"  Skipped:       {bp.get('skipped_frames', 0)}")
            print(f"  Drop rate:     {bp.get('drop_rate', 0)*100:.1f}%")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Async Object Detection Service")
    parser.add_argument(
        "--model",
        type=str,
        default="owlv2",
        choices=["owlv2", "yolo-world", "yoloe-11l", "grounding_dino"],
        help="Detection model to use",
    )
    parser.add_argument("--workers", type=int, default=2, help="Number of worker threads")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--report-interval", type=float, default=5.0, help="Statistics report interval (seconds)")

    args = parser.parse_args()

    # Create and run service
    service = AsyncDetectionService(
        model_id=args.model,
        num_workers=args.workers,
        verbose=args.verbose,
    )

    service.run(report_interval=args.report_interval)


if __name__ == "__main__":
    main()
