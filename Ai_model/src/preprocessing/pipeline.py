"""
Complete preprocessing pipeline for acne classification.

This pipeline combines multiple preprocessing steps to optimize
images for acne lesion classification, improving model accuracy.
"""

import numpy as np
from PIL import Image

from .median_filter import apply_median_filter
from .segmentation import otsu_segment
from .enhancement import clahe_enhance, sharpen


def preprocess_image(image):
    """
    Apply complete preprocessing pipeline to an image.
    
    Pipeline steps:
    1. Median filter - Remove noise while preserving edges
    2. Otsu segmentation - Isolate acne lesions from background
    3. CLAHE enhancement - Improve local contrast
    4. Sharpening - Enhance edge details
    
    Args:
        image: Input image (PIL Image or numpy array)
            - If numpy array, assumes RGB format (not BGR)
            - If PIL Image, will be handled correctly
    
    Returns:
        PIL Image: Preprocessed image ready for model input
    
    Note:
        The function handles both PIL Images and numpy arrays.
        For numpy arrays, RGB format is assumed (not OpenCV's BGR).
    """
    # Ensure we have a PIL Image for consistent processing
    if isinstance(image, np.ndarray):
        # Assume RGB format (not BGR) for numpy arrays
        # Convert to PIL Image
        if len(image.shape) == 3:
            # RGB image
            image = Image.fromarray(image.astype(np.uint8))
        else:
            # Grayscale
            image = Image.fromarray(image.astype(np.uint8), mode='L')
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    # Step 1: Apply median filter to reduce noise
    # This preserves edges while removing salt-and-pepper noise
    filtered = apply_median_filter(image, ksize=5)
    
    # Step 2: Apply Otsu segmentation to isolate lesions
    # We use the mask to focus on relevant regions
    mask, segmented = otsu_segment(filtered)
    
    # Step 3: Apply CLAHE for contrast enhancement
    # This improves visibility of subtle acne lesions
    enhanced = clahe_enhance(segmented, clip_limit=2.0, tile_grid_size=(8, 8))
    
    # Step 4: Apply sharpening to enhance edge details
    # This makes lesion boundaries more distinct
    sharpened = sharpen(enhanced, strength=1.5)
    
    return sharpened

