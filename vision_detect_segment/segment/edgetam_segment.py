"""
EdgeTAM Segmentation Model Wrapper for vision_detect_segment.
"""

import torch
import numpy as np

# Assume EdgeTAM codebase is installed and available in PYTHONPATH.
from sam2.modeling.sam2_base import SAM2Base
import yaml
import os


class EdgeTAMSegmenter:
    """
    Wrapper for EdgeTAM Track-Anything Model.
    """

    def __init__(self, config_path="sam2/edgetam.yaml", weights_path=None, device="cuda"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        # EdgeTAM uses hydra config; here we adapt the config extraction.
        model_config = config["model"]
        # Instantiate model from config. Normally you'd use hydra or omegaconf, but direct for clarity:
        self.model = SAM2Base(**model_config)
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
        self.model.eval()
        self.device = device
        self.model.to(device)

    def segment(self, image: np.ndarray):
        """
        Runs EdgeTAM segmentation on the provided image.
        Args:
            image: np.ndarray, shape [H,W,3], RGB format.
        Returns:
            mask: np.ndarray, binary segmentation mask or multi-class mask.
        """
        # Preprocessing here should match EdgeTAM requirements.
        input_tensor = self.preprocess(image)
        with torch.no_grad():
            pred_mask = self.model(input_tensor)
        # Postprocess output to match mask format.
        mask = self.postprocess(pred_mask)
        return mask

    def preprocess(self, image: np.ndarray):
        """
        Converts input image to EdgeTAM input tensor.
        """
        image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        return tensor

    def postprocess(self, output):
        """
        Converts model output to usable mask.
        """
        mask = output.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8)
        return mask


# Convenience function for integration by factory/discovery mechanism in your package
def create_edgetam_segmenter(config_path=None, weights_path=None, device="cpu"):
    config = config_path or os.path.join(os.path.dirname(__file__), "edgetam.yaml")
    return EdgeTAMSegmenter(config_path=config, weights_path=weights_path, device=device)
