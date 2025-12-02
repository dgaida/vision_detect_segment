#!/usr/bin/env python3
"""
Gets current frames from robot_camera redis stream, detects and segments objects and publishes the detected objects to
the redis  and publishes annotated frames to the redis streamer annotated_camera

have to run the script after mcp server is started. or before

But this runs independently of robot_environment package.
"""

import time
from vision_detect_segment import VisualCortex, get_default_config


def main():
    """
    Main detection loop that processes camera frames and publishes results.
    """
    # Configuration
    det_mdl = "owlv2"  # or "yoloe-11l"
    config = get_default_config(det_mdl)
    config.enable_label_monitoring = True  # Enable Redis label monitoring

    # Initialize VisualCortex
    visual_cortex = VisualCortex(
        objdetect_model_id=det_mdl,
        device="auto",
        stream_name="robot_camera",
        annotated_stream_name="annotated_camera",
        publish_annotated=True,
        verbose=True,
        config=config,
    )

    print("Starting object detection service...")
    print("Listening for images on Redis stream: robot_camera")
    print("Publishing detections to Redis stream: detected_objects")
    print("Publishing annotated frames to: annotated_camera")
    print("Press Ctrl+C to stop")

    frame_count = 0

    try:
        while True:
            # Trigger detection from latest Redis image
            success = visual_cortex.detect_objects_from_redis()

            if success:
                frame_count += 1
                if frame_count % 10 == 0:
                    detected = visual_cortex.get_detected_objects()
                    print(f"Processed {frame_count} frames, " f"current detections: {len(detected)}")

            # Small delay to avoid busy waiting
            time.sleep(0.05)  # 20 FPS max

    except KeyboardInterrupt:
        print("\nStopping detection service...")
    finally:
        visual_cortex.cleanup()
        print("Detection service stopped")


if __name__ == "__main__":
    main()
