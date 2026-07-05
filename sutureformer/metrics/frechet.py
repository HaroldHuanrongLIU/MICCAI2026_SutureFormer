"""Discrete Frechet Distance metric."""

import numpy as np
from numpy.linalg import norm


def compute_frechet_distance(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Discrete Frechet distance between predicted and GT trajectories.

    Measures shape similarity regardless of timing. Uses vectorized DP
    with pre-computed pairwise distance matrix.

    Args:
        predicted: [F, 2] single trajectory (numpy).
        ground_truth: [F, 2] single trajectory (numpy).

    Returns:
        Discrete Frechet distance.
    """
    n, m = len(predicted), len(ground_truth)

    # Pre-compute full pairwise distance matrix [n, m]
    diff = predicted[:, np.newaxis, :] - ground_truth[np.newaxis, :, :]  # [n, m, 2]
    dist = norm(diff, axis=-1)  # [n, m]

    # DP: row-by-row with vectorized operations where possible
    dp = np.empty((n, m))
    dp[0, 0] = dist[0, 0]

    # First row: dp[0, j] = max(dp[0, j-1], dist[0, j])
    for j in range(1, m):
        dp[0, j] = max(dp[0, j - 1], dist[0, j])

    # First column: dp[i, 0] = max(dp[i-1, 0], dist[i, 0])
    for i in range(1, n):
        dp[i, 0] = max(dp[i - 1, 0], dist[i, 0])

    # Interior: dp[i,j] = max(min(dp[i-1,j], dp[i-1,j-1], dp[i,j-1]), dist[i,j])
    for i in range(1, n):
        for j in range(1, m):
            dp[i, j] = max(min(dp[i - 1, j], dp[i - 1, j - 1], dp[i, j - 1]), dist[i, j])

    return float(dp[n - 1, m - 1])
