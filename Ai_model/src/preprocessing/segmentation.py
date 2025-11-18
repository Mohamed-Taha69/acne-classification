"""
Image segmentation for acne lesion detection.

Segmentation helps isolate acne lesions from background skin,
improving the model's ability to focus on relevant features.
"""

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def otsu_segment(image):
    """
    Apply Otsu's thresholding for binary segmentation.
    
    Otsu's method automatically determines the optimal threshold
    to separate foreground (acne lesions) from background (skin).
    
    Args:
        image: Input image (PIL Image or numpy array)
    
    Returns:
        tuple: (mask, segmented_image) where:
            - mask: Binary mask (PIL Image)
            - segmented_image: Original image with mask applied (PIL Image)
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
        is_pil = True
    else:
        img_array = image
        is_pil = False
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply Otsu's thresholding
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply mask to original image
    if len(img_array.shape) == 3:
        mask_3d = np.stack([mask] * 3, axis=-1)
        segmented = np.where(mask_3d > 0, img_array, 0)
    else:
        segmented = np.where(mask > 0, img_array, 0)
    
    # Convert back to PIL if input was PIL
    if is_pil:
        mask_pil = Image.fromarray(mask)
        segmented_pil = Image.fromarray(segmented.astype(np.uint8))
        return mask_pil, segmented_pil
    
    return mask, segmented


def kmeans_segment(image, K=2):
    """
    Apply K-means clustering for segmentation.
    
    K-means groups pixels into K clusters based on color similarity,
    useful for separating acne lesions from healthy skin.
    
    Args:
        image: Input image (PIL Image or numpy array)
        K: Number of clusters (default: 2 for binary segmentation)
    
    Returns:
        tuple: (mask, segmented_image) where:
            - mask: Cluster labels as mask (PIL Image)
            - segmented_image: Image colored by cluster labels (PIL Image)
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
        is_pil = True
    else:
        img_array = image
        is_pil = False
    
    # Reshape image to 2D array of pixels
    if len(img_array.shape) == 3:
        h, w, c = img_array.shape
        pixels = img_array.reshape(-1, c)
    else:
        h, w = img_array.shape
        pixels = img_array.reshape(-1, 1)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    
    # Reshape labels back to image shape
    mask = labels.reshape(h, w)
    
    # Create segmented image using cluster centers
    segmented_pixels = kmeans.cluster_centers_[labels]
    segmented = segmented_pixels.reshape(img_array.shape).astype(np.uint8)
    
    # Normalize mask to 0-255 range
    mask_normalized = ((mask - mask.min()) / (mask.max() - mask.min() + 1e-8) * 255).astype(np.uint8)
    
    # Convert back to PIL if input was PIL
    if is_pil:
        mask_pil = Image.fromarray(mask_normalized)
        segmented_pil = Image.fromarray(segmented)
        return mask_pil, segmented_pil
    
    return mask_normalized, segmented

