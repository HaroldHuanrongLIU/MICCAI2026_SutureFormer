"""Pre-populate the per-sample disk cache for TrajectoryDataset.

Iterates over all samples in all splits, triggering the cache-miss path
once per sample. Subsequent training/evaluation runs load from cache
(~360x faster than decoding raw PNGs).

Usage:
    python scripts/warm_cache.py
    python scripts/warm_cache.py --cache_dir /path/to/cache
    python scripts/warm_cache.py --splits train val test
"""

import argparse
import time

from sutureformer.datasets import TrajectoryDataset, get_split

DEFAULT_DATA_ROOT = (
    "/path/to/dataset/"
    "MICCAI2026_Surgical_Trajectory_Prediction_Dataset"
)
DEFAULT_CACHE_DIR = (
    "/path/to/dataset/cache"
)


def main():
    parser = argparse.ArgumentParser(description="Warm the TrajectoryDataset disk cache")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--obs_keyframe_end", type=int, default=6)
    parser.add_argument("--min_keyframes", type=int, default=7)
    args = parser.parse_args()

    for split_name in args.splits:
        patient_ids = get_split(split_name)
        ds = TrajectoryDataset(
            patient_ids=patient_ids,
            data_root=args.data_root,
            crop_size=128,
            obs_keyframe_end=args.obs_keyframe_end,
            min_keyframes=args.min_keyframes,
            cache_dir=args.cache_dir,
        )
        print(f"[{split_name}] {len(ds)} samples")
        t0 = time.time()
        for i in range(len(ds)):
            _ = ds[i]
            if (i + 1) % 50 == 0 or i == len(ds) - 1:
                elapsed = time.time() - t0
                print(f"  [{i + 1}/{len(ds)}] {elapsed:.1f}s elapsed")
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s ({elapsed / len(ds):.2f}s/sample)\n")

    print("Cache warm-up complete.")


if __name__ == "__main__":
    main()
