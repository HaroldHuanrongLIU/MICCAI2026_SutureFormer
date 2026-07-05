"""Local crop extraction centered on the needle tip."""

import numpy as np


def extract_local_crop(image: np.ndarray, x: int, y: int, crop_size: int = 128) -> np.ndarray:
    """Extract a local crop centered at (x, y) from the full surgical image.

    Args:
        image: [H_img, W_img, 3] uint8 RGB image (902 x 1264).
        x: Needle tip x-coordinate (0-1263).
        y: Needle tip y-coordinate (0-901).
        crop_size: Size of the square crop (default 128).

    Returns:
        crop: [crop_size, crop_size, 3] uint8 RGB.
    """
    half = crop_size // 2
    H_img, W_img = image.shape[:2]

    # Compute crop boundaries with clamping
    y1 = max(0, y - half)
    y2 = min(H_img, y + half)
    x1 = max(0, x - half)
    x2 = min(W_img, x + half)

    # Extract region
    crop = image[y1:y2, x1:x2]

    # Zero-pad if crop is smaller than crop_size (near boundaries)
    pad_top = half - (y - y1)
    pad_bottom = half - (y2 - y)
    pad_left = half - (x - x1)
    pad_right = half - (x2 - x)

    if any(p > 0 for p in [pad_top, pad_bottom, pad_left, pad_right]):
        crop = np.pad(
            crop,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    return crop  # [crop_size, crop_size, 3]
