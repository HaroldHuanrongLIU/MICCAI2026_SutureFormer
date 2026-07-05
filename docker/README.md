# Docker Setup

## Prerequisites

- Docker with BuildKit support
- NVIDIA Container Toolkit (`nvidia-container-toolkit`)
- Your user in the `docker` group (or use `sudo`)

## Build

```bash
# From the repository root
docker build -t sutureformer -f docker/Dockerfile .
```

## Run

```bash
# Using docker-compose (recommended)
docker compose -f docker/docker-compose.yml up

# With config overrides
docker compose -f docker/docker-compose.yml run sutureformer train.py trainer.batch_size=32 trainer.total_epochs=50

# Direct docker run
docker run --gpus 1 --shm-size=16g \
    -v $(pwd):/workspace \
    -v /path/to/dataset:/path/to/dataset \
    sutureformer train.py
```

## Volume Mounts

The `docker-compose.yml` mounts two volumes:

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `../` (repo root) | `/workspace` | Source code and configs |
| `/path/to/dataset` | `/path/to/dataset` | Dataset (read-only at runtime) |

The source code mount means local edits are reflected inside the container without rebuilding.

## Config Overrides

All Hydra overrides work as CLI arguments after `train.py`:

```bash
docker compose -f docker/docker-compose.yml run sutureformer \
    train.py trainer.use_wandb=true trainer.wandb_project=MyProject
```
