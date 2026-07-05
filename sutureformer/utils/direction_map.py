"""Direction-to-displacement mapping for the 9 discrete action directions."""

import torch

# Direction unit vectors (normalized)
# 0=up, 1=NE, 2=right, 3=SE, 4=down, 5=SW, 6=left, 7=NW, 8=idle
DIRECTION_VECTORS = torch.tensor(
    [
        [0.0, -1.0],  # 0: up (y decreases)
        [1.0, -1.0],  # 1: NE
        [1.0, 0.0],  # 2: right
        [1.0, 1.0],  # 3: SE
        [0.0, 1.0],  # 4: down (y increases)
        [-1.0, 1.0],  # 5: SW
        [-1.0, 0.0],  # 6: left
        [-1.0, -1.0],  # 7: NW
        [0.0, 0.0],  # 8: idle (no movement)
    ],
    dtype=torch.float32,
)

# Normalize diagonal directions to unit length
for i in [1, 3, 5, 7]:
    DIRECTION_VECTORS[i] = DIRECTION_VECTORS[i] / DIRECTION_VECTORS[i].norm()


def compute_displacement(
    direction_idx: torch.LongTensor, magnitude: torch.FloatTensor
) -> torch.FloatTensor:
    """Compute (dx, dy) from direction index and magnitude.

    Args:
        direction_idx: [B] LongTensor, values in {0, ..., 8}.
        magnitude: [B, 1] FloatTensor.

    Returns:
        displacement: [B, 2] (dx, dy) in normalized coordinate space.
    """
    unit_vec = DIRECTION_VECTORS.to(direction_idx.device)[direction_idx]  # [B, 2]
    displacement = unit_vec * magnitude  # [B, 2]
    return displacement
