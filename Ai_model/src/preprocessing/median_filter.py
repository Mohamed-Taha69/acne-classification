"""
Median filter for noise reduction in acne images.

Median filtering is effective for removing salt-and-pepper noise
while preserving edges, which is crucial for acne lesion detection.
"""

import cv2
import numpy as np
from PIL import Image


def apply_median_filter(image, ksize=5):
    """
    Apply median filter to reduce noise in the image.
    
    Args:
        image: Input image (PIL Image or numpy array)
        ksize: Kernel size for median filter (must be odd, default: 5)
    
    Returns:
        PIL Image: Filtered image
    
    Raises:
        ValueError: If ksize is even
    """
    if ksize % 2 == 0:
        raise ValueError(f"ksize must be odd, got {ksize}")
    
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
        is_pil = True
    else:
        img_array = image
        is_pil = False
    
    # Handle RGB vs BGR
    if len(img_array.shape) == 3:
        # Apply median filter to each channel
        filtered = np.zeros_like(img_array)
        for i in range(img_array.shape[2]):
            filtered[:, :, i] = cv2.medianBlur(img_array[:, :, i], ksize)
    else:
        # Grayscale
        filtered = cv2.medianBlur(img_array, ksize)
    
    # Convert back to PIL Image if input was PIL
    if is_pil:
        return Image.fromarray(filtered)
    
    return filtered

