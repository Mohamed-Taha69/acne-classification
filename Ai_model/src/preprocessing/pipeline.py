"""
Complete preprocessing pipeline for acne classification.

This pipeline combines multiple preprocessing steps to optimize
images for acne lesion classification, improving model accuracy.
"""

import numpy as np
from PIL import Image

from .median_filter import apply_median_filter
from .segmentation import otsu_segment, kmeans_segment
from .enhancement import clahe_enhance, sharpen


DEFAULT_PIPELINE_CONFIG = {
    "median_ksize": 3,  # Optimized: preserves fine lesion details while removing noise
    "use_segmentation": False,  # Disabled by default: slow, removes context, ConvNeXt learns focus
    "segmentation_method": "otsu",  # or "kmeans"
    "clahe_clip_limit": 2.0,  # Optimized: effective enhancement without over-amplification
    "clahe_tile_grid_size": (8, 8),  # Optimized: faster than (6,6), still effective
    "sharpen_strength": 0.0,  # Disabled by default: redundant with augmentation sharpen
}


def _ensure_pil(image):
    """Convert numpy arrays to PIL Images for consistent processing."""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            return Image.fromarray(image.astype(np.uint8))
        return Image.fromarray(image.astype(np.uint8), mode="L")
    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")
    return image


def _apply_segmentation(image, method: str):
    """Apply the requested segmentation method."""
    method = (method or "otsu").lower()
    if method == "kmeans":
        _, segmented = kmeans_segment(image, K=2)
        return segmented
    # default to otsu
    _, segmented = otsu_segment(image)
    return segmented


def preprocess_image(image, config: dict | None = None):
    """
    Apply preprocessing pipeline to an image using an optional configuration.

    Args:
        image: PIL Image or numpy array (RGB assumed for numpy)
        config: Optional dict to override DEFAULT_PIPELINE_CONFIG keys.

    Returns:
        PIL Image processed according to the pipeline
    """
    cfg = {**DEFAULT_PIPELINE_CONFIG, **(config or {})}
    image = _ensure_pil(image)

    processed = image
    if cfg.get("median_ksize", 0):
        processed = apply_median_filter(processed, ksize=int(cfg["median_ksize"]))

    if cfg.get("use_segmentation", True):
        processed = _apply_segmentation(processed, cfg.get("segmentation_method", "otsu"))

    processed = clahe_enhance(
        processed,
        clip_limit=float(cfg.get("clahe_clip_limit", 2.0)),
        tile_grid_size=tuple(cfg.get("clahe_tile_grid_size", (8, 8))),
    )

    strength = float(cfg.get("sharpen_strength", 1.5))
    if strength > 0:
        processed = sharpen(processed, strength=strength)

    return processed


