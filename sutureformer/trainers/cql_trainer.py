"""Offline CQL-Discrete trainer for surgical trajectory prediction."""

import copy
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from sutureformer.losses.cql_loss import compute_cql_q_loss, compute_nstep_returns
from sutureformer.losses.rl_losses import compute_bc_loss, compute_policy_loss
from sutureformer.metrics.frechet import compute_frechet_distance
from sutureformer.models.sutureformer import SutureFormer
from sutureformer.models.q_network import QNetwork
from sutureformer.utils.direction_map import DIRECTION_VECTORS
from sutureformer.datasets.obs_augmentation import augment_observations
from sutureformer.utils.expert_actions import extract_expert_transitions
from sutureformer.utils.guidance_extrapolation import generate_test_guidance

logger = logging.getLogger("sutureformer.trainer")


class CQLTrainer:
    """Offline CQL-Discrete trainer for surgical trajectory prediction.

    Orchestrates:
    - Expert transition extraction from GT trajectories (offline)
    - Twin Q-network training with CQL conservative penalty
    - Policy (direction head) training via entropy-regularized objective
    - Magnitude head training via supervised loss
    """

    def __init__(
        self,
        model: SutureFormer,
        cfg_trainer: Any,
        cfg_model: Any,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg_trainer

        # Q-networks (twin)
        state_dim = cfg_model.get("state_dim", 480)
        q_hidden_dim = cfg_model.get("q_hidden_dim", 256)
        num_directions = cfg_model.get("num_directions", 9)

        self.q1 = QNetwork(state_dim, q_hidden_dim, num_directions).to(device)
        self.q2 = QNetwork(state_dim, q_hidden_dim, num_directions).to(device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        # Freeze target networks
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        # Fixed entropy temperature (no auto-tuning for offline CQL)
        self.alpha = cfg_trainer.get("alpha", 0.2)

        self.n_critic_updates = getattr(cfg_trainer, "n_critic_updates", 1)

        # Optimizers (4 groups — no alpha optimizer for offline CQL)
        self.optimizer_encoder = Adam(model.obs_encoder.parameters(), lr=cfg_trainer.lr_encoder)
        self.optimizer_actor = Adam(
            list(model.state_encoder.parameters()) + list(model.direction_head.parameters()),
            lr=cfg_trainer.lr_actor,
        )
        self.optimizer_critic = Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=cfg_trainer.lr_critic,
        )
        self.optimizer_magnitude = Adam(
            model.magnitude_head.parameters(), lr=cfg_trainer.lr_magnitude
        )

        # Cosine annealing LR schedulers (decay to 1% of initial LR)
        total_epochs = cfg_trainer.total_epochs
        self.scheduler_encoder = CosineAnnealingLR(
            self.optimizer_encoder, T_max=total_epochs, eta_min=cfg_trainer.lr_encoder * 0.01
        )
        self.scheduler_actor = CosineAnnealingLR(
            self.optimizer_actor, T_max=total_epochs, eta_min=cfg_trainer.lr_actor * 0.01
        )
        self.scheduler_critic = CosineAnnealingLR(
            self.optimizer_critic, T_max=total_epochs, eta_min=cfg_trainer.lr_critic * 0.01
        )
        self.scheduler_magnitude = CosineAnnealingLR(
            self.optimizer_magnitude, T_max=total_epochs, eta_min=cfg_trainer.lr_magnitude * 0.01
        )

        # Training state
        self.global_epoch = 0
        self.best_val_ade = float("inf")

    def soft_update(self, target: nn.Module, source: nn.Module, tau: float) -> None:
        """Polyak averaging for target network update."""
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        save_dir: str = "logs/hydra",
        tb_writer=None,
        wandb_run=None,
    ) -> None:
        """Main training loop (epoch-based, offline CQL).

        Per-batch steps:
        1. Encode observations (with gradients)
        2. Extract expert transitions from GT trajectories
        3. Q-update with CQL penalty (states detached, no encoder grads)
        4. Policy/BC + Magnitude update (states with grad, backprops through encoder)
        5. Soft target update

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader (optional).
            save_dir: Directory for checkpoints.
            tb_writer: TensorBoard SummaryWriter (optional).
            wandb_run: WandB run object (optional).
        """
        os.makedirs(save_dir, exist_ok=True)
        total_epochs = self.cfg.total_epochs
        alpha_cql = getattr(self.cfg, "alpha_cql", 1.0)
        lambda_bc = getattr(self.cfg, "lambda_bc", 0.0)
        n_step = getattr(self.cfg, "n_step", 1)
        max_grad_transitions = getattr(self.cfg, "max_grad_transitions", 2048)
        batch_count = 0

        for epoch in range(total_epochs):
            self.model.train()
            self.global_epoch = epoch
            epoch_logs = {
                "mag_loss": 0.0,
                "q1_loss": 0.0,
                "q2_loss": 0.0,
                "cql_penalty": 0.0,
                "policy_loss": 0.0,
                "bc_loss": 0.0,
            }
            epoch_batches = 0

            for batch in train_loader:
                obs_images = batch["obs_images"].to(self.device)
                obs_coords = batch["obs_coords"].to(self.device)
                obs_mask = batch["obs_mask"].to(self.device)
                n_obs_batch = batch["n_obs"].to(self.device)

                # Observation augmentation (training only)
                obs_images, obs_coords, obs_mask, n_obs_batch = augment_observations(
                    obs_images, obs_coords, obs_mask, n_obs_batch,
                    coord_jitter_std=getattr(self.cfg, "obs_coord_jitter_std", 0.0),
                    color_jitter_brightness=getattr(self.cfg, "obs_color_jitter_brightness", 0.0),
                    temporal_dropout_prob=getattr(self.cfg, "obs_temporal_dropout_prob", 0.0),
                )
                batch["n_obs"] = n_obs_batch.cpu()

                # Step 1: Encode observations (with gradients)
                z_ctx = self.model.encode_observation(obs_images, obs_coords, obs_mask)

                # Step 2: Extract expert transitions
                transitions = extract_expert_transitions(self.model, z_ctx, batch, self.cfg)

                states = transitions["states"]  # [N, 480] with grad
                actions = transitions["actions"]  # [N]
                rewards = transitions["rewards"]  # [N]
                next_states = transitions["next_states"]  # [N, 480]
                dones = transitions["dones"]  # [N]
                mag_gt = transitions["mag_gt"]  # [N, 1]
                sample_ids = transitions["sample_ids"]  # [N]
                step_ids = transitions["step_ids"]  # [N]

                N_total = states.shape[0]
                alpha_val = self.alpha

                # Compute N-step returns
                if n_step > 1:
                    nstep_rewards, nstep_next_states, nstep_dones, _ = compute_nstep_returns(
                        rewards, dones, next_states.detach(), sample_ids, step_ids,
                        self.cfg.gamma, n_step,
                    )
                    q_rewards = nstep_rewards
                    q_next_states = nstep_next_states
                    q_dones = nstep_dones
                else:
                    q_rewards = rewards
                    q_next_states = next_states.detach()
                    q_dones = dones

                # Step 3: Q-update (detached states, no encoder gradients)
                # Run n_critic_updates times per batch to let Q converge faster.
                q1_bellman = q2_bellman = 0.0
                q1_cql = q2_cql = 0.0
                states_det = states.detach()

                for _ in range(self.n_critic_updates):
                    q1_values = self.q1(states_det)  # [N, 9]
                    q2_values = self.q2(states_det)  # [N, 9]

                    with torch.no_grad():
                        next_logits = self.model.direction_head(q_next_states)  # [N, 9]
                        q1_target_next = self.q1_target(q_next_states)  # [N, 9]
                        q2_target_next = self.q2_target(q_next_states)  # [N, 9]

                    q1_total, q1_bellman, q1_cql = compute_cql_q_loss(
                        q1_values, actions, q_rewards, q1_target_next, q2_target_next,
                        next_logits, alpha_val, alpha_cql, self.cfg.gamma, q_dones, n_step,
                    )
                    q2_total, q2_bellman, q2_cql = compute_cql_q_loss(
                        q2_values, actions, q_rewards, q1_target_next, q2_target_next,
                        next_logits, alpha_val, alpha_cql, self.cfg.gamma, q_dones, n_step,
                    )

                    self.optimizer_critic.zero_grad()
                    (q1_total + q2_total).backward()
                    nn.utils.clip_grad_norm_(
                        list(self.q1.parameters()) + list(self.q2.parameters()),
                        self.cfg.gradient_clip,
                    )
                    self.optimizer_critic.step()
                    self.soft_update(self.q1_target, self.q1, self.cfg.tau)
                    self.soft_update(self.q2_target, self.q2, self.cfg.tau)

                # Subsample for gradient-heavy steps if needed
                if N_total > max_grad_transitions:
                    idx = torch.randperm(N_total, device=self.device)[:max_grad_transitions]
                    states_sub = states[idx]
                    mag_gt_sub = mag_gt[idx]
                    actions_sub = actions[idx]
                else:
                    states_sub = states
                    mag_gt_sub = mag_gt
                    actions_sub = actions

                # Steps 4-5: Policy + Magnitude (both need encoder gradients)
                # Compute both losses on the same graph, backward both before
                # any optimizer step to avoid in-place param update invalidation.
                direction_logits = self.model.direction_head(states_sub)
                q1_vals = self.q1(states_sub.detach()).detach()
                q2_vals = self.q2(states_sub.detach()).detach()
                policy_loss = compute_policy_loss(direction_logits, q1_vals, q2_vals, alpha_val)
                bc_loss = (
                    compute_bc_loss(direction_logits, actions_sub)
                    if lambda_bc > 0
                    else torch.tensor(0.0, device=self.device)
                )

                mag_pred = self.model.magnitude_head(states_sub)  # [N_sub, 1]
                mag_loss = self.cfg.lambda_magnitude * F.mse_loss(mag_pred, mag_gt_sub)

                # Zero all relevant optimizers, then backward both losses
                self.optimizer_actor.zero_grad()
                self.optimizer_encoder.zero_grad()
                self.optimizer_magnitude.zero_grad()
                (policy_loss + lambda_bc * bc_loss + mag_loss).backward()

                # Clip and step each optimizer group
                nn.utils.clip_grad_norm_(
                    list(self.model.state_encoder.parameters())
                    + list(self.model.direction_head.parameters()),
                    self.cfg.gradient_clip,
                )
                nn.utils.clip_grad_norm_(
                    self.model.obs_encoder.parameters(), self.cfg.gradient_clip
                )
                nn.utils.clip_grad_norm_(
                    self.model.magnitude_head.parameters(), self.cfg.gradient_clip
                )
                self.optimizer_actor.step()
                self.optimizer_encoder.step()
                self.optimizer_magnitude.step()

                epoch_logs["mag_loss"] += mag_loss.item()
                epoch_logs["q1_loss"] += q1_bellman
                epoch_logs["q2_loss"] += q2_bellman
                epoch_logs["cql_penalty"] += (q1_cql + q2_cql) / 2.0
                epoch_logs["policy_loss"] += policy_loss.item()
                epoch_logs["bc_loss"] += bc_loss.item() if lambda_bc > 0 else 0.0
                epoch_batches += 1
                batch_count += 1

            # Step LR schedulers
            self.scheduler_encoder.step()
            self.scheduler_actor.step()
            self.scheduler_critic.step()
            self.scheduler_magnitude.step()

            # End-of-epoch logging
            n = max(epoch_batches, 1)
            cur_lr_enc = self.scheduler_encoder.get_last_lr()[0]
            cur_lr_act = self.scheduler_actor.get_last_lr()[0]
            log_msg = f"Epoch {epoch}/{total_epochs}"
            log_msg += f" | batches={epoch_batches}"
            log_msg += f" | mag={epoch_logs['mag_loss']/n:.4f}"
            log_msg += f" | q1={epoch_logs['q1_loss']/n:.4f}"
            log_msg += f" | cql={epoch_logs['cql_penalty']/n:.4f}"
            log_msg += f" | policy={epoch_logs['policy_loss']/n:.4f}"
            log_msg += f" | bc={epoch_logs['bc_loss']/n:.4f}"
            log_msg += f" | alpha={self.alpha:.4f}"
            log_msg += f" | lr_enc={cur_lr_enc:.2e} lr_act={cur_lr_act:.2e}"
            logger.info(log_msg)

            # Log training metrics to TensorBoard / WandB
            metrics = {
                "train/mag_loss": epoch_logs["mag_loss"] / n,
                "train/q1_loss": epoch_logs["q1_loss"] / n,
                "train/cql_penalty": epoch_logs["cql_penalty"] / n,
                "train/policy_loss": epoch_logs["policy_loss"] / n,
                "train/bc_loss": epoch_logs["bc_loss"] / n,
                "train/alpha": self.alpha,
                "train/lr_encoder": cur_lr_enc,
                "train/lr_actor": cur_lr_act,
            }
            if tb_writer:
                for k, v in metrics.items():
                    tb_writer.add_scalar(k, v, epoch)
            if wandb_run:
                wandb_run.log(metrics, step=epoch)

            # Evaluation
            if val_loader is not None and epoch % self.cfg.eval_interval == 0 and epoch > 0:
                val_metrics = self.evaluate(val_loader)
                logger.info(
                    f"  Val ADE={val_metrics['ADE']:.2f} FDE={val_metrics['FDE']:.2f} "
                    f"FD={val_metrics['FD']:.2f}"
                )

                # Log validation metrics
                val_log = {
                    "val/ADE": val_metrics["ADE"],
                    "val/FDE": val_metrics["FDE"],
                    "val/FD": val_metrics["FD"],
                }
                if tb_writer:
                    for k, v in val_log.items():
                        tb_writer.add_scalar(k, v, epoch)
                if wandb_run:
                    wandb_run.log(val_log, step=epoch)

                if self.cfg.save_best and val_metrics["ADE"] < self.best_val_ade:
                    self.best_val_ade = val_metrics["ADE"]
                    self.save_checkpoint(os.path.join(save_dir, "best_model.pt"))
                    logger.info(f"  New best model saved (ADE={self.best_val_ade:.2f})")

            # Periodic checkpoint
            if epoch % self.cfg.save_every == 0 and epoch > 0:
                self.save_checkpoint(os.path.join(save_dir, f"checkpoint_epoch{epoch}.pt"))

    def _rollout_batch(self, batch):
        """Run softmax-weighted rollout on a single batch.

        Returns:
            pred_px, gt_px: [B, max_pred, 2] in pixel space.
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

        # Convert to pixel space
        pred_px = predicted.clone()
        pred_px[:, :, 0] *= 1263
        pred_px[:, :, 1] *= 901
        gt_px = pred_coords.clone()
        gt_px[:, :, 0] *= 1263
        gt_px[:, :, 1] *= 901

        meta = {"pred_mask": pred_mask, "pred_is_kf": pred_is_kf, "n_pred": n_pred}
        return pred_px, gt_px, meta

    def evaluate_keyframe(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on keyframe positions only.

        Args:
            data_loader: Evaluation data loader.

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

                    all_ade.append(errors[i, :np_i][kf_mask].mean().item())
                    all_fde.append(errors[i, kf_indices[-1].item()].item())

                    pred_kf = pred_px[i, :np_i][kf_mask].cpu().numpy()
                    gt_kf = gt_px[i, :np_i][kf_mask].cpu().numpy()
                    if len(pred_kf) >= 2:
                        all_fd.append(compute_frechet_distance(pred_kf, gt_kf))

        self.model.train()
        return {
            "ADE": float(np.mean(all_ade)),
            "FDE": float(np.mean(all_fde)),
            "FD": float(np.mean(all_fd)) if all_fd else float("nan"),
        }

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Default evaluation using keyframe-only metrics."""
        return self.evaluate_keyframe(data_loader)

    def save_checkpoint(self, path: str) -> None:
        """Save model, Q-networks, optimizers, schedulers, and training state."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "q1_state_dict": self.q1.state_dict(),
                "q2_state_dict": self.q2.state_dict(),
                "q1_target_state_dict": self.q1_target.state_dict(),
                "q2_target_state_dict": self.q2_target.state_dict(),
                "alpha": self.alpha,
                "optimizer_encoder": self.optimizer_encoder.state_dict(),
                "optimizer_actor": self.optimizer_actor.state_dict(),
                "optimizer_critic": self.optimizer_critic.state_dict(),
                "optimizer_magnitude": self.optimizer_magnitude.state_dict(),
                "scheduler_encoder": self.scheduler_encoder.state_dict(),
                "scheduler_actor": self.scheduler_actor.state_dict(),
                "scheduler_critic": self.scheduler_critic.state_dict(),
                "scheduler_magnitude": self.scheduler_magnitude.state_dict(),
                "epoch": self.global_epoch,
                "best_val_ade": self.best_val_ade,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model, Q-networks, optimizers, schedulers, and training state."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.q1.load_state_dict(ckpt["q1_state_dict"])
        self.q2.load_state_dict(ckpt["q2_state_dict"])
        self.q1_target.load_state_dict(ckpt["q1_target_state_dict"])
        self.q2_target.load_state_dict(ckpt["q2_target_state_dict"])
        self.alpha = ckpt.get("alpha", self.alpha)
        self.optimizer_encoder.load_state_dict(ckpt["optimizer_encoder"])
        self.optimizer_actor.load_state_dict(ckpt["optimizer_actor"])
        self.optimizer_critic.load_state_dict(ckpt["optimizer_critic"])
        self.optimizer_magnitude.load_state_dict(ckpt["optimizer_magnitude"])
        if "scheduler_encoder" in ckpt:
            self.scheduler_encoder.load_state_dict(ckpt["scheduler_encoder"])
            self.scheduler_actor.load_state_dict(ckpt["scheduler_actor"])
            self.scheduler_critic.load_state_dict(ckpt["scheduler_critic"])
            self.scheduler_magnitude.load_state_dict(ckpt["scheduler_magnitude"])
        self.global_epoch = ckpt.get("epoch", 0)
        self.best_val_ade = ckpt.get("best_val_ade", float("inf"))
        logger.info(f"Loaded checkpoint from epoch {self.global_epoch}")
