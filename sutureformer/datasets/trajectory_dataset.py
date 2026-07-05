"""PyTorch Dataset for surgical trajectory prediction with keyframe-based splits."""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from sutureformer.datasets.crop import extract_local_crop
from sutureformer.datasets.guidance import construct_guidance_channel
from sutureformer.datasets.keyframe_split import compute_obs_pred_split

# Image coordinate ranges for normalization
X_MAX = 1263.0
Y_MAX = 901.0


class TrajectoryDataset(Dataset):
    """Dataset for surgical trajectory prediction using keyframe-based splits.

    Each trajectory is split at the obs_keyframe_end-th keyframe boundary,
    producing variable-length observation and prediction phases.
    """

    def __init__(
        self,
        patient_ids: List[str],
        data_root: str = "/path/to/dataset/"
        "MICCAI2026_Surgical_Trajectory_Prediction_Dataset",
        crop_size: int = 128,
        obs_keyframe_end: int = 6,
        min_keyframes: int = 7,
        cache_dir: Optional[str] = None,
        image_size: Optional[List[int]] = None,
        coord_range: Optional[List[int]] = None,
        num_buckets: Optional[int] = None,
        _target_: Optional[str] = None,
    ) -> None:
        """Initialize the trajectory dataset.

        Args:
            patient_ids: List of patient ID strings to include.
            data_root: Path to the dataset root directory.
            crop_size: Size of the local crop around needle tip.
            obs_keyframe_end: Number of keyframes in the observation phase.
            min_keyframes: Minimum keyframes required (obs_keyframe_end + 1).
            cache_dir: Directory for per-sample .pt cache files. When set,
                processed samples are saved on first access and loaded from
                cache on subsequent accesses (~360x faster).
            image_size: Ignored (kept for Hydra config compatibility).
            coord_range: Ignored (kept for Hydra config compatibility).
            num_buckets: Ignored (kept for Hydra config compatibility).
            _target_: Ignored (Hydra instantiation field).
        """
        super().__init__()
        self.data_root = data_root
        self.crop_size = crop_size
        self.obs_keyframe_end = obs_keyframe_end
        self.cache_dir = cache_dir
        if cache_dir is not None:
            self._cache_subdir = os.path.join(cache_dir, f"kf{obs_keyframe_end}")
            os.makedirs(self._cache_subdir, exist_ok=True)
        else:
            self._cache_subdir = None

        # Build sample index: list of (patient_id, traj_id, n_obs, n_pred) tuples
        self.samples: List[Dict[str, Any]] = []
        for patient_id in patient_ids:
            patient_dir = os.path.join(data_root, patient_id)
            if not os.path.isdir(patient_dir):
                continue

            for traj_id in sorted(os.listdir(patient_dir)):
                traj_dir = os.path.join(patient_dir, traj_id)
                if not os.path.isdir(traj_dir):
                    continue

                # Find the labels JSON file
                label_file = os.path.join(traj_dir, f"{patient_id}_{traj_id}_labels.json")
                if not os.path.exists(label_file):
                    continue

                with open(label_file, "r") as f:
                    label_data = json.load(f)

                labels = label_data["labels"]

                # Count keyframes for quick filter
                num_kf = sum(1 for lbl in labels if lbl["is_keyframe"])
                if num_kf < min_keyframes:
                    continue

                n_obs, n_pred = compute_obs_pred_split(labels, obs_keyframe_end)
                if n_obs == 0:
                    continue

                self.samples.append(
                    {
                        "patient_id": patient_id,
                        "trajectory_id": traj_id,
                        "traj_dir": traj_dir,
                        "n_obs": n_obs,
                        "n_pred": n_pred,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _cache_path(self, patient_id: str, trajectory_id: str) -> Optional[str]:
        """Return the cache file path for a sample, or None if caching is disabled."""
        if self._cache_subdir is None:
            return None
        return os.path.join(self._cache_subdir, f"{patient_id}_{trajectory_id}.pt")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a single trajectory sample split at the keyframe boundary.

        Returns:
            Dict with keys:
                obs_images: [n_obs, 4, crop_size, crop_size] float32 (RGB+guidance) in [0, 1]
                obs_coords: [n_obs, 2] float32 normalized (x, y)
                pred_coords: [n_pred, 2] float32 normalized (x, y)
                pred_conf: [n_pred] float32
                pred_is_kf: [n_pred] float32 (0 or 1)
                n_obs: int
                n_pred: int
                patient_id: str
                trajectory_id: str
        """
        sample = self.samples[idx]
        n_obs = sample["n_obs"]
        n_pred = sample["n_pred"]
        patient_id = sample["patient_id"]
        trajectory_id = sample["trajectory_id"]

        # Fast path: load from cache
        cache_path = self._cache_path(patient_id, trajectory_id)
        if cache_path is not None and os.path.exists(cache_path):
            cached = torch.load(cache_path, weights_only=True)
            return {
                "obs_images": cached["obs_images"].float(),
                "obs_coords": cached["obs_coords"],
                "pred_coords": cached["pred_coords"],
                "pred_conf": cached["pred_conf"],
                "pred_is_kf": cached["pred_is_kf"],
                "n_obs": n_obs,
                "n_pred": n_pred,
                "patient_id": patient_id,
                "trajectory_id": trajectory_id,
            }

        # Slow path: load from raw PNGs
        result = self._load_from_raw(sample)

        # Save to cache
        if cache_path is not None:
            tmp_path = cache_path + ".tmp"
            torch.save(
                {
                    "obs_images": result["obs_images"].half(),
                    "obs_coords": result["obs_coords"],
                    "pred_coords": result["pred_coords"],
                    "pred_conf": result["pred_conf"],
                    "pred_is_kf": result["pred_is_kf"],
                },
                tmp_path,
            )
            os.replace(tmp_path, cache_path)

        return result

    def _load_from_raw(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Load a sample from raw PNG files (slow path)."""
        n_obs = sample["n_obs"]
        n_pred = sample["n_pred"]
        patient_id = sample["patient_id"]
        trajectory_id = sample["trajectory_id"]
        traj_dir = sample["traj_dir"]

        # Load labels
        label_file = os.path.join(traj_dir, f"{patient_id}_{trajectory_id}_labels.json")
        with open(label_file, "r") as f:
            label_data = json.load(f)
        labels = label_data["labels"]

        # Only use observation-phase coordinates for guidance (no future leakage)
        obs_guide_coords = [(lbl["pos_x"], lbl["pos_y"]) for lbl in labels[:n_obs]]
        obs_guide_confidences = [lbl["confidence"] for lbl in labels[:n_obs]]

        # Observation frames (0 to n_obs)
        obs_labels = labels[:n_obs]
        obs_images = []
        obs_coords = []

        for lbl in obs_labels:
            x, y = lbl["pos_x"], lbl["pos_y"]

            # Load image and extract crop
            img_path = os.path.join(traj_dir, lbl["frame_file"])
            img = np.array(Image.open(img_path).convert("RGB"))  # [902, 1264, 3]
            crop = extract_local_crop(img, x, y, self.crop_size)  # [128, 128, 3]

            # Convert to float32 [0, 1] and to CHW format
            crop_tensor = torch.from_numpy(crop).float().permute(2, 0, 1) / 255.0  # [3, 128, 128]

            # Guidance channel (obs-only, no prediction-phase leakage)
            guid = construct_guidance_channel(
                x, y, obs_guide_coords, obs_guide_confidences, self.crop_size
            )  # [128, 128, 1]
            guid_tensor = torch.from_numpy(guid).float().permute(2, 0, 1)  # [1, 128, 128]

            # Concatenate RGB + guidance -> 4 channels
            combined = torch.cat([crop_tensor, guid_tensor], dim=0)  # [4, 128, 128]
            obs_images.append(combined)

            # Normalized coordinates
            obs_coords.append(torch.tensor([x / X_MAX, y / Y_MAX], dtype=torch.float32))

        # Prediction frames (n_obs to end)
        pred_labels = labels[n_obs : n_obs + n_pred]
        pred_coords = []
        pred_conf = []
        pred_is_kf = []

        for lbl in pred_labels:
            pred_coords.append(
                torch.tensor([lbl["pos_x"] / X_MAX, lbl["pos_y"] / Y_MAX], dtype=torch.float32)
            )
            pred_conf.append(lbl["confidence"])
            pred_is_kf.append(float(lbl["is_keyframe"]))

        return {
            "obs_images": torch.stack(obs_images),  # [n_obs, 4, 128, 128]
            "obs_coords": torch.stack(obs_coords),  # [n_obs, 2]
            "pred_coords": torch.stack(pred_coords),  # [n_pred, 2]
            "pred_conf": torch.tensor(pred_conf, dtype=torch.float32),  # [n_pred]
            "pred_is_kf": torch.tensor(pred_is_kf, dtype=torch.float32),  # [n_pred]
            "n_obs": n_obs,
            "n_pred": n_pred,
            "patient_id": patient_id,
            "trajectory_id": trajectory_id,
        }
