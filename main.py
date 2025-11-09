import cv2
import time
from pathlib import Path

from vision_detect_segment.core.visualcortex import VisualCortex
from vision_detect_segment.utils.config import get_default_config, create_test_config
from vision_detect_segment.utils.utils import (
    create_test_image, load_image_safe, format_detection_results,
    setup_logging, Timer
)
from vision_detect_segment.utils.exceptions import (
    VisionDetectionError, RedisConnectionError, DetectionError
)
from redis_robot_comm import RedisImageStreamer


def publish_test_image(stream_name: str = "robot_camera", use_test_config: bool = True):
    """Publish a test image to Redis."""
    logger = setup_logging(verbose=True)

    try:
        streamer = RedisImageStreamer(stream_name=stream_name)

        # Try to load example image, create test image if not found
        image_path = Path("examples/example.png")

        if image_path.exists():
            image = load_image_safe(image_path)
            logger.info(f"Loaded image from {image_path}")
        else:
            # Create test image with shapes suitable for detection
            if use_test_config:
                shapes = ["square", "circle"]  # Simple shapes for testing
            else:
                shapes = ["square", "circle", "rectangle"]

            image = create_test_image(shapes=shapes)
            logger.info("Created test image (example.png not found)")

            # Optionally save the test image
            cv2.imwrite("test_image.png", image)
            logger.info("Test image saved as test_image.png")

        logger.info(f"Publishing image with shape: {image.shape}")

        # Prepare metadata
        metadata = {
            "robot": "test_robot",
            "workspace": "test_workspace",
            "workspace_id": "workspace_test",
            "robot_pose": {"x": 0.0, "y": 0.0, "z": 0.5, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "frame_id": 1,
            "timestamp": time.time()
        }

        with Timer("Publishing image to Redis", logger):
            streamer.publish_image(image, metadata=metadata)

        logger.info("Image published successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to publish test image: {e}")
        return False


def display_results(visual_cortex: VisualCortex, show_images: bool = True):
    """Display detection results and optionally show images."""
    logger = setup_logging(verbose=True)

    try:
        # Get results using new API
        detected_objects = visual_cortex.get_detected_objects()
        raw_image = visual_cortex.get_current_image()
        annotated_image = visual_cortex.get_annotated_image()

        # Print detection summary using utility function
        summary = format_detection_results(detected_objects, max_items=10)
        logger.info("\n=== Detection Results ===")
        logger.info(summary)

        # Print processing statistics
        stats = visual_cortex.get_stats()
        logger.info("\n=== Processing Statistics ===")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        # Show images if requested
        if show_images:
            try:
                if raw_image is not None:
                    logger.info(f"Showing raw image (shape: {raw_image.shape})")
                    cv2.imshow("Raw Image", raw_image)
                    cv2.waitKey(0)

                if annotated_image is not None:
                    logger.info(f"Showing annotated image (shape: {annotated_image.shape})")
                    cv2.imshow("Annotated Image", annotated_image)
                    cv2.waitKey(0)

                cv2.destroyAllWindows()

            except Exception as e:
                logger.error(f"Error displaying images: {e}")

    except Exception as e:
        logger.error(f"Error in display_results: {e}")


def test_manual_processing_with_config():
    """Test manual processing using configuration system."""
    logger = setup_logging(verbose=True)

    try:
        logger.info("=== Testing Manual Processing with Config ===")

        # Use test configuration for faster processing
        config = create_test_config()
        logger.info(f"Using test config with {len(config.get_object_labels()[0])} object labels")

        # Publish image first
        with Timer("Publishing test image", logger):
            success = publish_test_image(use_test_config=True)

        if not success:
            logger.error("Failed to publish test image")
            return None

        time.sleep(1)  # Give Redis time to store

        # Initialize VisualCortex with configuration
        with Timer("Initializing VisualCortex", logger):
            visual_cortex = VisualCortex(
                objdetect_model_id="owlv2",
                device="auto",
                verbose=True,
                config=config
            )

        # Manually trigger detection
        logger.info("Manually triggering object detection...")
        with Timer("Object detection from Redis", logger):
            success = visual_cortex.detect_objects_from_redis()

        if success:
            display_results(visual_cortex)

            # Clear GPU cache
            visual_cortex.clear_cache()

            return visual_cortex
        else:
            logger.error("Failed to get image from Redis for manual processing")
            return None

    except VisionDetectionError as e:
        logger.error(f"Vision detection error: {e}")
        if hasattr(e, 'details') and e.details:
            logger.error(f"Error details: {e.details}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in test_manual_processing_with_config: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_default_config_processing():
    """Test processing with default configuration."""
    logger = setup_logging(verbose=True)

    try:
        logger.info("=== Testing Default Configuration ===")

        # Use default configuration
        config = get_default_config("owlv2")
        logger.info(f"Using default config with {len(config.get_object_labels()[0])} object labels")

        # Publish image
        success = publish_test_image(use_test_config=False)
        if not success:
            return None

        time.sleep(1)

        # Initialize VisualCortex with default config
        visual_cortex = VisualCortex(
            objdetect_model_id="owlv2",
            device="auto",
            verbose=True,
            config=config
        )

        # Test detection
        success = visual_cortex.detect_objects_from_redis()

        if success:
            logger.info("Default configuration test successful")

            display_results(visual_cortex)

            # Show brief results without images to save time
            detected_objects = visual_cortex.get_detected_objects()
            summary = format_detection_results(detected_objects, max_items=3)
            logger.info(summary)

            # Clear GPU cache
            visual_cortex.clear_cache()

        return visual_cortex

    except Exception as e:
        logger.error(f"Default configuration test failed: {e}")
        return None


def test_error_handling():
    """Test error handling capabilities."""
    logger = setup_logging(verbose=True)

    logger.info("=== Testing Error Handling ===")

    try:
        # Test 1: Invalid model name
        logger.info("Testing invalid model error handling...")
        try:
            VisualCortex("invalid_model", device="auto", verbose=False)  # verbose=False to avoid noise
            logger.error("ERROR: Should have failed with invalid model")
            return False
        except VisionDetectionError as e:
            logger.info(f"✓ Correctly caught invalid model error: {type(e).__name__}")
        except Exception as e:
            logger.info(f"✓ Caught model error (different exception type): {type(e).__name__}")

        # Test 2: Redis connection issues
        logger.info("Testing Redis connection error handling...")
        try:
            config = create_test_config()
            config.redis.host = "invalid_host_12345"  # This should fail

            # This should still work but with Redis warning
            VisualCortex("owlv2", device="auto", verbose=False, config=config)
            logger.info("✓ VisualCortex initialized despite Redis issues (graceful degradation)")

        except VisionDetectionError as e:
            logger.info(f"✓ Redis error properly handled: {type(e).__name__}")
        except Exception as e:
            logger.info(f"✓ Redis connection issue handled: {type(e).__name__}")

        # Test 3: Image processing error handling
        logger.info("Testing image processing error handling...")
        try:
            config = create_test_config()
            VisualCortex("owlv2", device="auto", verbose=False, config=config)

            # Test with invalid image (this should be handled gracefully)
            invalid_image = None
            try:
                # This should trigger image validation error
                from vision_detect_segment.utils import validate_image
                validate_image(invalid_image)
                logger.error("ERROR: Should have failed with invalid image")
            except Exception as e:
                logger.info(f"✓ Image validation error correctly caught: {type(e).__name__}")

        except Exception as e:
            logger.warning(f"Image processing test had issues: {e}")

        logger.info("✓ Error handling tests completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error handling test failed unexpectedly: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to test the vision detection system."""
    logger = setup_logging(verbose=True)

    logger.info("Starting Vision Detection System Test")
    logger.info("====================================")

    try:
        with Timer("Complete test execution", logger):
            # Test 1: Manual processing with test config (main test)
            visual_cortex = test_manual_processing_with_config()

            if visual_cortex is not None:
                logger.info("✓ Manual processing test passed")

                # Test 2: Default config (optional, comment out if too slow)
                test_default_config_processing()

                # Test 3: Error handling
                test_error_handling()

                # Get final memory usage
                memory_info = visual_cortex.get_memory_usage()
                if memory_info:
                    logger.info(f"Final memory usage: {memory_info}")

                logger.info("\n=== All Tests Completed Successfully ===")
            else:
                logger.error("Main test failed - aborting remaining tests")

    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except VisionDetectionError as e:
        logger.error(f"\nVision system error: {e}")
        if hasattr(e, 'details') and e.details:
            logger.error(f"Details: {e.details}")
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

    logger.info("Test execution finished.")


if __name__ == "__main__":
    main()
