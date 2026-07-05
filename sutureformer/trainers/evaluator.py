"""Evaluation protocol for surgical trajectory prediction."""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from sutureformer.metrics.frechet import compute_frechet_distance
from sutureformer.models.sutureformer import SutureFormer
from sutureformer.utils.direction_map import DIRECTION_VECTORS
from sutureformer.utils.guidance_extrapolation import generate_test_guidance

logger = logging.getLogger("sutureformer.evaluator")


class Evaluator:
    """Evaluates trajectory prediction using greedy rollout and standard metrics."""

    def __init__(
        self,
        model: SutureFormer,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = model
        self.device = device

    def _rollout_batch(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Run softmax-weighted rollout on a single batch.

        Returns:
            pred_px: [B, max_pred, 2] predicted positions in pixel space.
            gt_px: [B, max_pred, 2] ground truth positions in pixel space.
            meta: dict with pred_mask, pred_is_kf, n_pred tensors.
        """
        obs_images = batch["obs_images"].to(self.device)
        obs_coords = batch["obs_coords"].to(self.device)
        obs_mask = batch["obs_mask"].to(self.device)
        pred_coords = batch["pred_coords"].to(self.device)
        pred_mask = batch["pred_mask"].to(self.device)
        pred_is_kf = batch["pred_is_kf"].to(self.device)
        n_pred = batch["n_pred"].to(self.device)
        n_obs = batch["n_obs"].to(self.device)

        B = obs_images.shape[0]
        max_pred = pred_coords.shape[1]

        # Encode observations
        z_ctx = self.model.encode_observation(obs_images, obs_coords, obs_mask)

        # Generate pseudo-guidance for testing
        guidance = generate_test_guidance(obs_coords, obs_mask, n_pred)

        # Start from last valid observed position
        last_obs_idx = (n_obs - 1).clamp(min=0)
        current_pos = obs_coords[torch.arange(B, device=self.device), last_obs_idx, :]

        predicted = []
        for k in range(max_pred):
            active_mask = k < n_pred
            step_ratio = torch.zeros(B, 1, device=self.device)
            if active_mask.any():
                step_ratio[active_mask, 0] = k / n_pred[active_mask].float()

            guidance_pos = guidance[:, min(k, max_pred - 1), :]
            direction_logits, magnitude = self.model.predict_step(
                z_ctx, current_pos, guidance_pos, step_ratio
            )

            # Softmax-weighted displacement: probability-weighted average
            # over all direction vectors instead of argmax
            probs = torch.softmax(direction_logits, dim=-1)  # [B, 9]
            dirs = DIRECTION_VECTORS.to(self.device)  # [9, 2]
            weighted_dir = (probs @ dirs)  # [B, 2]
            displacement = weighted_dir * magnitude  # [B, 2]
            next_pos = (current_pos + displacement).clamp(0.0, 1.0)

            # Only update active samples
            current_pos = torch.where(active_mask.unsqueeze(1), next_pos, current_pos)
            predicted.append(next_pos)

        predicted = torch.stack(predicted, dim=1)  # [B, max_pred, 2]

        # Convert to pixel space for metric computation
        pred_px = predicted.clone()
        pred_px[:, :, 0] *= 1263
        pred_px[:, :, 1] *= 901
        gt_px = pred_coords.clone()
        gt_px[:, :, 0] *= 1263
        gt_px[:, :, 1] *= 901

        meta = {
            "pred_mask": pred_mask,
            "pred_is_kf": pred_is_kf,
            "n_pred": n_pred,
        }
        return pred_px, gt_px, meta

    def evaluate_keyframe(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on keyframe positions only.

        Computes ADE, FDE, FD using only the prediction-phase keyframe
        positions (where pred_is_kf == 1), ignoring interpolated frames.

        Args:
            data_loader: DataLoader for val or test split (collate_variable_length).

        Returns:
            Dict with ADE, FDE, FD metrics (in pixel space, keyframe-only).
        """
        self.model.eval()
        all_ade, all_fde, all_fd = [], [], []

        with torch.no_grad():
            for batch in data_loader:
                pred_px, gt_px, meta = self._rollout_batch(batch)
                pred_is_kf = meta["pred_is_kf"]
                n_pred = meta["n_pred"]
                B = pred_px.shape[0]

                errors = torch.norm(pred_px - gt_px, dim=-1)  # [B, max_pred]

                for i in range(B):
                    np_i = n_pred[i].item()
                    kf_mask = pred_is_kf[i, :np_i].bool()
                    if not kf_mask.any():
                        continue

                    kf_indices = kf_mask.nonzero(as_tuple=True)[0]

                    # ADE: mean error at keyframe positions
                    all_ade.append(errors[i, :np_i][kf_mask].mean().item())

                    # FDE: error at last keyframe
                    all_fde.append(errors[i, kf_indices[-1].item()].item())

                    # FD: Frechet distance on keyframe-only subsequence
                    pred_kf = pred_px[i, :np_i][kf_mask].cpu().numpy()
                    gt_kf = gt_px[i, :np_i][kf_mask].cpu().numpy()
                    if len(pred_kf) >= 2:
                        all_fd.append(compute_frechet_distance(pred_kf, gt_kf))

        results = {
            "ADE": float(np.mean(all_ade)),
            "FDE": float(np.mean(all_fde)),
            "FD": float(np.mean(all_fd)) if all_fd else float("nan"),
        }
        logger.info(f"Evaluation results (keyframe-only): {results}")
        return results

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Default evaluation using keyframe-only metrics."""
        return self.evaluate_keyframe(data_loader)
