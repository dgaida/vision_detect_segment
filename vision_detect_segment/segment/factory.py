"""
Factory for segmentation models.
"""

from .edgetam_segment import create_edgetam_segmenter


def get_segmenter(model_type="default", **kwargs):
    if model_type == "edgetam":
        return create_edgetam_segmenter(**kwargs)
    # ... handle other models as usual
    raise ValueError(f"Unknown segmentation model type: {model_type}")
