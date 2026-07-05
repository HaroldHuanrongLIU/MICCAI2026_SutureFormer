"""Guidance channel construction for local crop regions."""

from typing import List, Tuple

import numpy as np
from scipy.ndimage import maximum_filter


def construct_guidance_channel(
    x: int,
    y: int,
    trajectory_coords: List[Tuple[int, int]],
    trajectory_confidences: List[float],
    crop_size: int = 128,
) -> np.ndarray:
    """Create guidance channel showing the trajectory path in local crop coordinates.

    Encodes trajectory points within the crop region with intensity proportional
    to their annotation confidence.

    Args:
        x: Current needle tip x-coordinate (crop center).
        y: Current needle tip y-coordinate (crop center).
        trajectory_coords: List of (x, y) for all frames in the trajectory.
        trajectory_confidences: List of confidence values for each frame.
        crop_size: Size of the square crop.

    Returns:
        guidance: [crop_size, crop_size, 1] float32 in [0, 1].
    """
    half = crop_size // 2
    guidance = np.zeros((crop_size, crop_size), dtype=np.float32)

    for (tx, ty), conf in zip(trajectory_coords, trajectory_confidences):
        # Convert to local crop coordinates
        local_x = tx - x + half
        local_y = ty - y + half

        # Check if within crop bounds
        if 0 <= local_x < crop_size and 0 <= local_y < crop_size:
            # Intensity encodes confidence (keyframes brighter)
            guidance[int(local_y), int(local_x)] = conf

    # Dilate the path for visibility (radius ~2 pixels)
    guidance = maximum_filter(guidance, size=3)

    return guidance[..., np.newaxis]  # [crop_size, crop_size, 1]
