"""Bucketed batch sampler for variable-length sequences."""

import random
from typing import Iterator, List

import torch.utils.data


class BucketedSampler(torch.utils.data.Sampler):
    """Batch sampler that groups samples by total sequence length.

    Sorts dataset indices by ``n_obs + n_pred``, splits into roughly equal
    buckets, then shuffles buckets and samples within each bucket to form
    batches.  This reduces padding waste while still providing randomisation.

    Args:
        dataset: A dataset with ``samples`` attribute where each entry has
            ``n_obs`` and ``n_pred`` fields (or a 4-tuple with them at indices 2,3).
        batch_size: Number of samples per batch.
        num_buckets: Number of length-based buckets.
        drop_last: Whether to drop the last incomplete batch.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_buckets: int = 10,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_buckets = num_buckets
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        # Compute total lengths and sort indices
        lengths = []
        for i, sample in enumerate(self.dataset.samples):
            if isinstance(sample, dict):
                total = sample["n_obs"] + sample["n_pred"]
            else:
                # tuple: (patient_id, traj_id, n_obs, n_pred)
                total = sample[2] + sample[3]
            lengths.append((total, i))

        lengths.sort(key=lambda x: x[0])
        sorted_indices = [idx for _, idx in lengths]

        # Split into buckets
        bucket_size = max(1, len(sorted_indices) // self.num_buckets)
        buckets = []
        for start in range(0, len(sorted_indices), bucket_size):
            buckets.append(sorted_indices[start : start + bucket_size])

        # Shuffle bucket order
        random.shuffle(buckets)

        # Shuffle within each bucket and yield batches
        batches = []
        for bucket in buckets:
            random.shuffle(bucket)
            for start in range(0, len(bucket), self.batch_size):
                batch = bucket[start : start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        # Shuffle batches for extra randomness
        random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
