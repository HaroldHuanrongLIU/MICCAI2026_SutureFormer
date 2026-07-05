"""Training entry point for SutureFormer with Hydra configuration."""

import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from sutureformer.datasets import BucketedSampler, TrajectoryDataset, collate_variable_length, get_split
from sutureformer.models.sutureformer import SutureFormer
from sutureformer.trainers.cql_trainer import CQLTrainer
from sutureformer.utils.seed import set_seed

logger = logging.getLogger("sutureformer")


def _init_loggers(cfg, save_dir):
    """Initialize WandB and/or TensorBoard loggers based on config."""
    wandb_run = None
    tb_writer = None

    if getattr(cfg.trainer, "use_wandb", False):
        import wandb

        wandb_dir = os.path.join(os.path.dirname(__file__), "logs", "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_run = wandb.init(
            project=getattr(cfg.trainer, "wandb_project", "SutureFormer"),
            entity=getattr(cfg.trainer, "wandb_entity", None),
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=wandb_dir,
        )
        logger.info(f"WandB run: {wandb_run.url}")

    if getattr(cfg.trainer, "use_tensorboard", False):
        from torch.utils.tensorboard import SummaryWriter

        tb_dir = os.path.join(os.path.dirname(__file__), "logs", "runs")
        os.makedirs(tb_dir, exist_ok=True)
        # Use Hydra run dir basename as run name for unique subdirectory
        run_name = os.path.basename(save_dir)
        tb_writer = SummaryWriter(log_dir=os.path.join(tb_dir, run_name))
        logger.info(f"TensorBoard logging to: {tb_dir}/{run_name}")

    return wandb_run, tb_writer


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Setup
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Create datasets
    logger.info("Loading training dataset...")
    train_dataset = TrajectoryDataset(
        patient_ids=get_split("train"),
        data_root=cfg.dataset.data_root,
        crop_size=cfg.dataset.crop_size,
        obs_keyframe_end=cfg.dataset.obs_keyframe_end,
        min_keyframes=cfg.dataset.min_keyframes,
        cache_dir=cfg.dataset.get("cache_dir", None),
    )
    logger.info(f"Training samples: {len(train_dataset)}")

    logger.info("Loading validation dataset...")
    val_dataset = TrajectoryDataset(
        patient_ids=get_split("val"),
        data_root=cfg.dataset.data_root,
        crop_size=cfg.dataset.crop_size,
        obs_keyframe_end=cfg.dataset.obs_keyframe_end,
        min_keyframes=cfg.dataset.min_keyframes,
        cache_dir=cfg.dataset.get("cache_dir", None),
    )
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders with bucketed sampling
    train_sampler = BucketedSampler(
        train_dataset,
        batch_size=cfg.trainer.batch_size,
        num_buckets=cfg.dataset.num_buckets,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=cfg.trainer.num_workers,
        collate_fn=collate_variable_length,
        pin_memory=True,
    )

    val_sampler = BucketedSampler(
        val_dataset,
        batch_size=cfg.trainer.batch_size,
        num_buckets=cfg.dataset.num_buckets,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
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
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Create trainer
    trainer = CQLTrainer(
        model=model,
        cfg_trainer=cfg.trainer,
        cfg_model=cfg.model,
        device=device,
    )

    # Train
    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"Saving to: {save_dir}")

    wandb_run, tb_writer = _init_loggers(cfg, save_dir)
    try:
        trainer.train(train_loader, val_loader, save_dir=save_dir, tb_writer=tb_writer, wandb_run=wandb_run)
    finally:
        if tb_writer is not None:
            tb_writer.close()
        if wandb_run is not None:
            import wandb

            wandb.finish()


if __name__ == "__main__":
    main()
