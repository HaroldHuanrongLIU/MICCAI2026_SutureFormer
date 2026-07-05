"""Variable-length collate function with padding and masking."""

from typing import Any, Dict, List

import torch


def collate_variable_length(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-length trajectory samples with padding and masks.

    Pads observation and prediction tensors to the maximum length in the batch,
    creates boolean masks (True = valid, False = pad), and collects metadata.

    Args:
        batch: List of sample dicts from TrajectoryDataset.

    Returns:
        Dict with padded tensors, masks, and metadata lists.
    """
    B = len(batch)
    max_obs = max(b["n_obs"] for b in batch)
    max_pred = max(b["n_pred"] for b in batch)

    # Infer crop size from first sample
    _, C, crop_h, crop_w = batch[0]["obs_images"].shape

    # Observation tensors
    obs_images = torch.zeros(B, max_obs, C, crop_h, crop_w)
    obs_coords = torch.zeros(B, max_obs, 2)
    obs_mask = torch.zeros(B, max_obs, dtype=torch.bool)

    # Prediction tensors
    pred_coords = torch.zeros(B, max_pred, 2)
    pred_conf = torch.zeros(B, max_pred)
    pred_is_kf = torch.zeros(B, max_pred)
    pred_mask = torch.zeros(B, max_pred, dtype=torch.bool)

    n_obs = torch.zeros(B, dtype=torch.long)
    n_pred = torch.zeros(B, dtype=torch.long)

    patient_ids = []
    trajectory_ids = []

    for i, sample in enumerate(batch):
        no = sample["n_obs"]
        np_ = sample["n_pred"]

        obs_images[i, :no] = sample["obs_images"]
        obs_coords[i, :no] = sample["obs_coords"]
        obs_mask[i, :no] = True

        pred_coords[i, :np_] = sample["pred_coords"]
        pred_conf[i, :np_] = sample["pred_conf"]
        pred_is_kf[i, :np_] = sample["pred_is_kf"]
        pred_mask[i, :np_] = True

        n_obs[i] = no
        n_pred[i] = np_

        patient_ids.append(sample["patient_id"])
        trajectory_ids.append(sample["trajectory_id"])

    return {
        "obs_images": obs_images,
        "obs_coords": obs_coords,
        "obs_mask": obs_mask,
        "pred_coords": pred_coords,
        "pred_conf": pred_conf,
        "pred_is_kf": pred_is_kf,
        "pred_mask": pred_mask,
        "n_obs": n_obs,
        "n_pred": n_pred,
        "patient_id": patient_ids,
        "trajectory_id": trajectory_ids,
    }
