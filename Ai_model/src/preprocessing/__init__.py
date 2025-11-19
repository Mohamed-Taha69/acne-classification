"""
Preprocessing module for acne classification.

This module provides image preprocessing functions including:
- Noise reduction (median filtering)
- Segmentation (Otsu, K-means)
- Enhancement (CLAHE, sharpening)
- Complete preprocessing pipeline
"""

from .median_filter import apply_median_filter
from .segmentation import otsu_segment, kmeans_segment
from .enhancement import clahe_enhance, sharpen
from .pipeline import preprocess_image, DEFAULT_PIPELINE_CONFIG

__all__ = [
    'apply_median_filter',
    'otsu_segment',
    'kmeans_segment',
    'clahe_enhance',
    'sharpen',
    'preprocess_image',
    'DEFAULT_PIPELINE_CONFIG',
]

