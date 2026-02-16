#!/usr/bin/env python3
"""
scripts/detect_objects_publish_annotated_frames_async.py

Async version of object detection with annotated frame publishing.
Uses background processing for improved throughput.
"""

import sys
import time

from vision_detect_segment import get_default_config
from vision_detect_segment.core.visualcortex_async import VisualCortexAsync
from vision_detect_segment.utils.exceptions import RedisConnectionError


def check_redis_connection(visual_cortex: VisualCortexAsync) -> bool:
    """
    Check if Redis connection is available and working.

    Args:
        visual_cortex: VisualCortexAsync instance to check

    Returns:
        bool: True if Redis is available, False otherwise
    """
    if visual_cortex._streamer is None:
        return False

    try:
        visual_cortex._streamer.client.ping()
        return True
    except Exception as e:
        print(f"Redis connection test failed: {e}")
        return False


def main():
    """
    Main async detection loop with proper Redis error handling.
    """
    # Configuration
    det_mdl = "owlv2"  # or "yoloe-11l"
    config = get_default_config(det_mdl)
    config.enable_label_monitoring = True

    print("=" * 70)
    print("Async Object Detection Service - Starting...")
    print("=" * 70)

    # Initialize VisualCortexAsync
    try:
        visual_cortex = VisualCortexAsync(
            objdetect_model_id=det_mdl,
            device="auto",
            stream_name="robot_camera",
            annotated_stream_name="annotated_camera",
            publish_annotated=True,
            verbose=False,
            config=config,
            # Async-specific parameters
            num_workers=2,
            max_queue_size=100,
            enable_backpressure=True,
            redis_poll_interval=0.01,  # 100 Hz polling
        )
    except Exception as e:
        print(f"\n❌ ERROR: Failed to initialize VisualCortexAsync: {e}")
        print("\nPlease check:")
        print("  - Model dependencies are installed")
        print("  - GPU/CUDA is available (if using GPU)")
        sys.exit(1)

    # Check Redis connection
    print("\n🔍 Checking Redis connection...")

    if visual_cortex._streamer is None:
        print("\n❌ ERROR: Redis streamer could not be initialized!")
        print("\nPossible causes:")
        print("  - Redis server is not running")
        print("  - Redis connection settings are incorrect")
        print(f"  - Cannot connect to {config.redis.host}:{config.redis.port}")
        print("\nTo fix this:")
        print("  1. Start Redis server:")
        print("     docker run -p 6379:6379 redis:alpine")
        print("  2. Or if Redis is installed locally:")
        print("     redis-server")
        print("\nExiting...")
        sys.exit(1)

    # Verify connection works
    if not check_redis_connection(visual_cortex):
        print("\n❌ ERROR: Redis connection test failed!")
        print(f"\nCannot connect to Redis at {config.redis.host}:{config.redis.port}")
        print("\nPlease ensure Redis server is running:")
        print("  docker run -p 6379:6379 redis:alpine")
        print("\nExiting...")
        sys.exit(1)

    print("✅ Redis connection successful!")

    # Start async processing pipeline
    try:
        print("\n🚀 Starting async processing pipeline...")
        visual_cortex.start()
        print("✅ Pipeline started successfully")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to start pipeline: {e}")
        sys.exit(1)

    # Connection successful - show service info
    print("\n" + "=" * 70)
    print("Async Object Detection Service - Running")
    print("=" * 70)
    print(f"Model: {det_mdl}")
    print("Input stream: robot_camera")
    print("Detection stream: detected_objects")
    print("Annotated stream: annotated_camera")
    print(f"Redis: {config.redis.host}:{config.redis.port}")
    print(f"Workers: {visual_cortex._num_workers}")
    print(f"Queue size: {visual_cortex._max_queue_size}")
    print(f"Backpressure: {'enabled' if visual_cortex._enable_backpressure else 'disabled'}")
    print("\nPress Ctrl+C to stop")
    print("=" * 70 + "\n")

    last_stats_time = time.time()
    stats_interval = 5.0  # Report stats every 5 seconds

    try:
        while True:
            # Check if it's time to report stats
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                stats = visual_cortex.get_stats()

                print("\n" + "─" * 70)
                print("📊 Service Statistics")
                print("─" * 70)
                print(f"Uptime:          {stats.get('uptime_seconds', 0):.1f}s")
                print(f"Tasks submitted: {stats.get('tasks_submitted', 0)}")
                print(f"Tasks completed: {stats.get('tasks_completed', 0)}")
                print(f"Tasks failed:    {stats.get('tasks_failed', 0)}")
                print(f"Throughput:      {stats.get('throughput_fps', 0):.2f} FPS")
                print(f"Avg proc time:   {stats.get('avg_processing_time', 0)*1000:.1f}ms")
                print(f"Queue size:      {stats.get('queue_size', 0)}")
                print(f"Current dets:    {stats.get('current_detections', 0)}")

                # Backpressure stats
                if "backpressure" in stats:
                    bp = stats["backpressure"]
                    print("\nBackpressure:")
                    print(f"  Dropped:       {bp.get('dropped_frames', 0)}")
                    print(f"  Skipped:       {bp.get('skipped_frames', 0)}")
                    print(f"  Drop rate:     {bp.get('drop_rate', 0)*100:.1f}%")

                print("─" * 70 + "\n")

                last_stats_time = current_time

            # Sleep briefly to avoid busy waiting
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("Stopping async detection service...")
        print("=" * 70)
    except RedisConnectionError as e:
        print(f"\n\n❌ Redis Connection Error: {e}")
        print("\nThe connection to Redis was lost.")
        print("Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        print("\n🛑 Stopping pipeline...")
        visual_cortex.stop(timeout=5.0)

        # Get final stats
        final_stats = visual_cortex.get_stats()
        print("\n" + "=" * 70)
        print("Service Summary")
        print("=" * 70)
        print(f"Total tasks:     {final_stats.get('tasks_completed', 0)}")
        print(f"Avg throughput:  {final_stats.get('throughput_fps', 0):.2f} FPS")
        print(f"Total runtime:   {final_stats.get('uptime_seconds', 0):.1f}s")
        print("=" * 70)

        print("\n🧹 Cleaning up...")
        visual_cortex.cleanup()
        print("✅ Service stopped")


if __name__ == "__main__":
    main()
