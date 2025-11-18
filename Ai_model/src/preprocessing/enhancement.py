"""
Image enhancement for acne classification.

Enhancement techniques improve image quality and contrast,
making acne lesions more visible and easier to classify.
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter


def clahe_enhance(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    CLAHE enhances local contrast while preventing over-amplification,
    which is ideal for medical images with varying lighting conditions.
    
    Args:
        image: Input image (PIL Image or numpy array)
        clip_limit: Threshold for contrast limiting (default: 2.0)
        tile_grid_size: Size of grid for histogram equalization (default: (8, 8))
    
    Returns:
        PIL Image: Enhanced image
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
        is_pil = True
    else:
        img_array = image
        is_pil = False
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE to each channel if RGB
    if len(img_array.shape) == 3:
        enhanced = np.zeros_like(img_array)
        # Convert RGB to LAB color space for better enhancement
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        # Apply CLAHE only to L channel (lightness)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        # Grayscale
        enhanced = clahe.apply(img_array)
    
    # Convert back to PIL Image if input was PIL
    if is_pil:
        return Image.fromarray(enhanced)
    
    return enhanced


def sharpen(image, strength=1.5):
    """
    Apply unsharp masking for image sharpening.
    
    Unsharp masking enhances edges and fine details in the image,
    making acne lesions more distinct and easier to detect.
    
    Args:
        image: Input image (PIL Image)
        strength: Sharpening strength factor (default: 1.5)
    
    Returns:
        PIL Image: Sharpened image
    """
    # Ensure input is PIL Image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Apply unsharp mask filter
    # The unsharp mask filter uses a Gaussian blur and subtracts it from the original
    sharpened = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    # Blend with original based on strength
    if strength != 1.0:
        # Create a blend between original and sharpened
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Sharpness(sharpened)
        sharpened = enhancer.enhance(strength)
    
    return sharpened

