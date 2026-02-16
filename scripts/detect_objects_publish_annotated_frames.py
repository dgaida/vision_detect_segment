#!/usr/bin/env python3
"""
Improved detect_objects_publish_annotated_frames.py with proper Redis error handling.

This script now properly exits when Redis is not available, providing clear error messages.
"""

import sys
import time

from vision_detect_segment import VisualCortex, get_default_config
from vision_detect_segment.core.visualcortex_metrics import VisionMetrics
from vision_detect_segment.utils.exceptions import RedisConnectionError
from vision_detect_segment.utils.metrics_exporter import MetricsExporter


def check_redis_connection(visual_cortex: VisualCortex) -> bool:
    """
    Check if Redis connection is available and working.

    Args:
        visual_cortex: VisualCortex instance to check

    Returns:
        bool: True if Redis is available, False otherwise
    """
    if visual_cortex._streamer is None:
        return False

    try:
        # Try to get latest image to verify connection
        # This will fail if Redis is not reachable
        visual_cortex._streamer.get_latest_image()
        return True  # Connection works, even if no image available yet
    except Exception as e:
        print(f"Redis connection test failed: {e}")
        return False


def main():
    """
    Main detection loop with proper Redis error handling.
    """
    # Configuration
    det_mdl = "owlv2"  # or "yoloe-11l"
    config = get_default_config(det_mdl)
    config.enable_label_monitoring = True

    print("=" * 60)
    print("Object Detection Service - Starting...")
    print("=" * 60)

    # Initialize VisualCortex
    try:
        visual_cortex = VisualCortex(
            objdetect_model_id=det_mdl,
            device="auto",
            stream_name="robot_camera",
            annotated_stream_name="annotated_camera",
            publish_annotated=True,
            verbose=False,
            config=config,
        )

        # Initialize metrics
        metrics = VisionMetrics(model_id=det_mdl)
        visual_cortex._metrics = metrics

        # Start metrics HTTP server
        exporter = MetricsExporter(port=9090)
        exporter.start()
    except Exception as e:
        print(f"\n❌ ERROR: Failed to initialize VisualCortex: {e}")
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

    # Connection successful - start processing
    print("\n" + "=" * 60)
    print("Object Detection Service - Running")
    print("=" * 60)
    print(f"Model: {det_mdl}")
    print("Input stream: robot_camera")
    print("Detection stream: detected_objects")
    print("Annotated stream: annotated_camera")
    print(f"Redis: {config.redis.host}:{config.redis.port}")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")

    frame_count = 0
    consecutive_failures = 0
    max_consecutive_failures = 5

    try:
        while True:
            # Trigger detection from latest Redis image
            success = visual_cortex.detect_objects_from_redis()

            if success:
                frame_count += 1
                consecutive_failures = 0  # Reset failure counter

                metrics.record_frame_processed(success=True)

                if frame_count % 20 == 0:
                    detected = visual_cortex.get_detected_objects()
                    print(f"✓ Processed {frame_count} frames, " f"current detections: {len(detected)}")

                    for det in detected:
                        metrics.record_detection(class_name=det["label"], confidence=det["confidence"])
            else:
                consecutive_failures += 1

                # After several failures, check if Redis is still available
                if consecutive_failures >= max_consecutive_failures:
                    print(f"\n⚠️  Warning: {consecutive_failures} consecutive failures")
                    print("   Checking Redis connection...")

                    if not check_redis_connection(visual_cortex):
                        print("\n❌ ERROR: Lost connection to Redis!")
                        print("   Redis server may have stopped.")
                        print("\nExiting...")
                        sys.exit(1)

                    # Redis is still alive, just no images
                    print("   Redis is available, waiting for images...")
                    consecutive_failures = 0  # Reset counter

            # Small delay to avoid busy waiting
            time.sleep(0.05)  # 20 FPS max

    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Stopping detection service...")
        print("=" * 60)
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
        print("\nCleaning up...")
        visual_cortex.cleanup()
        print(f"✓ Processed {frame_count} total frames")
        print("✓ Detection service stopped")


if __name__ == "__main__":
    main()
