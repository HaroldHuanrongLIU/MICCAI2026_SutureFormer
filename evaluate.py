"""Evaluation entry point for SutureFormer with Hydra configuration."""

import logging

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from sutureformer.datasets import BucketedSampler, TrajectoryDataset, collate_variable_length, get_split
from sutureformer.models.sutureformer import SutureFormer
from sutureformer.trainers.evaluator import Evaluator
from sutureformer.utils.seed import set_seed

logger = logging.getLogger("sutureformer")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load test dataset
    test_dataset = TrajectoryDataset(
        patient_ids=get_split("test"),
        data_root=cfg.dataset.data_root,
        crop_size=cfg.dataset.crop_size,
        obs_keyframe_end=cfg.dataset.obs_keyframe_end,
        min_keyframes=cfg.dataset.min_keyframes,
        cache_dir=cfg.dataset.get("cache_dir", None),
    )
    logger.info(f"Test samples: {len(test_dataset)}")

    test_sampler = BucketedSampler(
        test_dataset,
        batch_size=cfg.trainer.batch_size,
        num_buckets=cfg.dataset.num_buckets,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=cfg.trainer.num_workers,
        collate_fn=collate_variable_length,
        pin_memory=True,
    )

    # Create model
    model = SutureFormer(
        cnn_in_channels=cfg.model.cnn_in_channels,
        cnn_feature_dim=cfg.model.cnn_feature_dim,
        coord_embed_dim=cfg.model.coord_embed_dim,
        coord_num_frequencies=cfg.model.coord_num_frequencies,
        transformer_d_model=cfg.model.transformer_d_model,
        transformer_nhead=cfg.model.transformer_nhead,
        transformer_layers=cfg.model.transformer_layers,
        transformer_ff_dim=cfg.model.transformer_ff_dim,
        transformer_dropout=cfg.model.transformer_dropout,
        state_dim=cfg.model.state_dim,
        num_directions=cfg.model.num_directions,
        delta_max=cfg.model.delta_max,
        q_hidden_dim=cfg.model.q_hidden_dim,
    )

    # Load checkpoint
    checkpoint_path = cfg.get("checkpoint", "logs/hydra/best_model.pt")
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # Evaluate
    evaluator = Evaluator(model=model, device=device)
    results = evaluator.evaluate(test_loader)

    print(f"\n{'='*40}")
    print("Test Results:")
    print(f"  ADE: {results['ADE']:.2f} pixels")
    print(f"  FDE: {results['FDE']:.2f} pixels")
    print(f"  FD:  {results['FD']:.2f} pixels")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
