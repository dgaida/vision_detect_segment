#!/usr/bin/env python3
"""
Gets current frames from robot_camera redis stream, detects and segments objects and publishes the detected objects to
the redis  and publishes annotated frames to the redis streamer annotated_camera

have to run the script after mcp server is started. or before
"""

import time
from vision_detect_segment import VisualCortex
from vision_detect_segment.config import get_default_config


def main():
    # 1. Initialize VisualCortex with annotated publishing enabled
    config = get_default_config("owlv2")
    cortex = VisualCortex(
        "owlv2",
        device="cuda",
        stream_name="robot_camera",
        annotated_stream_name="annotated_camera",  # Custom stream name
        publish_annotated=True,  # Enable publishing
        verbose=True,
        config=config,
    )

    while True:
        # Process and auto-publish annotated frame
        cortex.detect_objects_from_redis()

        time.sleep(0.1)


if __name__ == "__main__":
    exit(main())
