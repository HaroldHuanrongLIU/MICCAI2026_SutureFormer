"""Keyframe-based observation/prediction split for variable-length trajectories."""

from typing import Tuple


def compute_obs_pred_split(labels: list, obs_keyframe_end: int = 6) -> Tuple[int, int]:
    """Compute observation/prediction split based on keyframe boundaries.

    Splits a trajectory at the obs_keyframe_end-th keyframe so that the
    observation phase contains exactly that many keyframes, and the prediction
    phase contains the remaining frames (which must include at least one keyframe).

    Args:
        labels: List of per-frame label dicts, each with an ``is_keyframe`` bool.
        obs_keyframe_end: Number of keyframes to include in the observation phase.

    Returns:
        (n_obs, n_pred) tuple. Returns (0, 0) if the trajectory has insufficient
        keyframes or no prediction frames.
    """
    keyframe_indices = [i for i, lbl in enumerate(labels) if lbl["is_keyframe"]]

    # Need at least obs_keyframe_end + 1 keyframes (obs_keyframe_end for obs, 1+ for pred)
    if len(keyframe_indices) < obs_keyframe_end + 1:
        return (0, 0)

    # Split at the obs_keyframe_end-th keyframe (0-indexed list -> index obs_keyframe_end-1)
    split_frame_idx = keyframe_indices[obs_keyframe_end - 1]
    n_obs = split_frame_idx + 1
    n_pred = len(labels) - n_obs

    if n_pred <= 0:
        return (0, 0)

    return (n_obs, n_pred)
