"""
Unit tests for configuration module.
"""

import pytest

from vision_detect_segment.utils.config import (
    MODEL_CONFIGS,
    AnnotationConfig,
    ModelConfig,
    RedisConfig,
    VisionConfig,
    create_test_config,
    get_default_config,
    get_model_config,
)


class TestModelConfig:
    """Tests for ModelConfig class."""

    def test_default_values(self):
        """Test default ModelConfig values."""
        config = ModelConfig(name="test_model")
        assert config.name == "test_model"
        assert config.confidence_threshold == 0.3
        assert config.max_detections == 20
        assert config.device_preference == "auto"
        assert isinstance(config.model_params, dict)

    def test_get_device_auto(self):
        """Test automatic device selection."""
        config = ModelConfig(name="test", device_preference="auto")
        device = config.get_device()
        assert device in ["cuda", "cpu"]

    def test_get_device_specific(self):
        """Test specific device selection."""
        config = ModelConfig(name="test", device_preference="cpu")
        assert config.get_device() == "cpu"


class TestRedisConfig:
    """Tests for RedisConfig class."""

    def test_default_values(self):
        """Test default RedisConfig values."""
        config = RedisConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.stream_name == "robot_camera"
        assert config.detection_stream == "detected_objects"
        assert config.connection_timeout == 5
        assert config.retry_attempts == 3

    def test_custom_values(self):
        """Test custom RedisConfig values."""
        config = RedisConfig(host="192.168.1.100", port=6380, stream_name="custom_stream")
        assert config.host == "192.168.1.100"
        assert config.port == 6380
        assert config.stream_name == "custom_stream"


class TestAnnotationConfig:
    """Tests for AnnotationConfig class."""

    def test_default_values(self):
        """Test default AnnotationConfig values."""
        config = AnnotationConfig()
        assert config.text_scale == 0.5
        assert config.text_padding == 3
        assert config.box_thickness == 2
        assert config.resize_scale_factor == 2.0
        assert config.show_confidence is True
        assert config.show_labels is True


class TestVisionConfig:
    """Tests for VisionConfig class."""

    def test_initialization(self):
        """Test VisionConfig initialization."""
        config = VisionConfig()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.redis, RedisConfig)
        assert isinstance(config.annotation, AnnotationConfig)
        assert config.verbose is False
        assert config.enable_segmentation is True

    def test_default_labels(self):
        """Test that default labels are set."""
        config = VisionConfig()
        labels = config.get_object_labels()
        assert len(labels) == 1
        assert len(labels[0]) > 0
        assert "blue circle" in labels[0]
        assert "red cube" in labels[0]

    def test_add_object_label(self):
        """Test adding object labels."""
        config = VisionConfig()
        initial_count = len(config.get_object_labels()[0])

        config.add_object_label("test object")
        new_count = len(config.get_object_labels()[0])

        assert new_count == initial_count + 1
        assert "test object" in config.get_object_labels()[0]

    def test_add_duplicate_label(self):
        """Test that duplicate labels are not added."""
        config = VisionConfig()
        config.set_object_labels(["object1", "object2"])

        config.add_object_label("object1")  # Try to add duplicate
        labels = config.get_object_labels()[0]

        assert labels.count("object1") == 1

    def test_set_object_labels(self):
        """Test setting custom object labels."""
        config = VisionConfig()
        custom_labels = ["label1", "label2", "label3"]

        config.set_object_labels(custom_labels)
        result_labels = config.get_object_labels()[0]

        assert len(result_labels) == 3
        assert all(label in result_labels for label in custom_labels)

    def test_labels_lowercase(self):
        """Test that labels are converted to lowercase."""
        config = VisionConfig()
        config.set_object_labels(["TEST", "Label", "OBJECT"])

        labels = config.get_object_labels()[0]
        assert all(label.islower() for label in labels)


class TestModelConfigs:
    """Tests for predefined model configurations."""

    def test_all_models_available(self):
        """Test that all expected models are configured."""
        expected_models = ["owlv2", "yolo-world", "grounding_dino"]

        for model_name in expected_models:
            assert model_name in MODEL_CONFIGS

    # TODO: if I only call pytest tests\test_config.py the test passes, but if all tests are executed this test fails
    def test_owlv2_config(self):
        """Test OWL-V2 model configuration."""
        config = get_model_config("owlv2")
        assert config.name == "owlv2"
        # Note: The actual confidence_threshold in config.py is 0.3, but create_test_config()
        # sets it to 0.2. Here we test the MODEL_CONFIGS directly.
        # After reviewing config.py, the owlv2 config has confidence_threshold=0.3
        # But the test uses create_test_config() which overrides it to 0.2
        # We should test the actual MODEL_CONFIGS value here
        assert config.confidence_threshold == 0.3  # This is correct from MODEL_CONFIGS
        assert "model_path" in config.model_params
        assert config.model_params["requires_transformers"] is True

    def test_yolo_world_config(self):
        """Test YOLO-World model configuration."""
        config = get_model_config("yolo-world")
        assert config.name == "yolo-world"
        assert config.confidence_threshold == 0.25
        assert "model_path" in config.model_params
        assert config.model_params["requires_ultralytics"] is True

    def test_grounding_dino_config(self):
        """Test Grounding-DINO model configuration."""
        config = get_model_config("grounding_dino")
        assert config.name == "grounding_dino"
        assert "model_path" in config.model_params
        assert config.model_params["requires_transformers"] is True


class TestGetDefaultConfig:
    """Tests for get_default_config function."""

    def test_owlv2_default_config(self):
        """Test getting default config for OWL-V2."""
        config = get_default_config("owlv2")
        assert isinstance(config, VisionConfig)
        assert config.model.name == "owlv2"
        assert len(config.get_object_labels()[0]) > 0

    def test_yolo_world_default_config(self):
        """Test getting default config for YOLO-World."""
        config = get_default_config("yolo-world")
        assert isinstance(config, VisionConfig)
        assert config.model.name == "yolo-world"

    def test_invalid_model_name(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError, match="Unsupported model"):
            get_default_config("invalid_model")


class TestCreateTestConfig:
    """Tests for create_test_config function."""

    def test_test_config_creation(self):
        """Test creating test configuration."""
        config = create_test_config()
        assert isinstance(config, VisionConfig)
        assert config.verbose is True
        assert config.model.confidence_threshold == 0.2

    def test_test_config_reduced_labels(self):
        """Test that test config has reduced labels."""
        default_config = get_default_config("owlv2")
        test_config = create_test_config()

        default_label_count = len(default_config.get_object_labels()[0])
        test_label_count = len(test_config.get_object_labels()[0])

        assert test_label_count < default_label_count
        assert test_label_count == 7  # Expected test label count

    def test_test_config_has_expected_labels(self):
        """Test that test config has expected labels."""
        config = create_test_config()
        labels = config.get_object_labels()[0]

        expected_labels = ["blue square", "chocolate bar", "mars", "snickers", "red circle", "pen", "book"]

        for expected_label in expected_labels:
            assert expected_label in labels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
