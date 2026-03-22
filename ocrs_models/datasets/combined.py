"""
Combined dataset that merges multiple detection datasets with configurable ratios.

This allows training on a weighted mixture of datasets (e.g. 40% HierText,
40% SynthText, 20% ICDAR) to benefit from the strengths of each.
"""

from typing import Optional

import torch
from torch.utils.data import Dataset

from .util import SizedDataset


class CombinedDataset(SizedDataset):
    """
    A dataset that combines multiple datasets with configurable sampling ratios.

    Each epoch samples from the constituent datasets according to the specified
    ratios. The total epoch size equals the size of the largest dataset scaled
    by its ratio, so that each dataset contributes proportionally.

    Example::

        combined = CombinedDataset(
            datasets=[hiertext_ds, synthtext_ds, icdar_ds],
            ratios=[0.4, 0.4, 0.2],
        )
    """

    def __init__(
        self,
        datasets: list[Dataset],
        ratios: list[float],
        max_images: Optional[int] = None,
    ):
        super().__init__()

        if len(datasets) != len(ratios):
            raise ValueError("Number of datasets must match number of ratios")

        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")

        self._datasets = datasets
        self._ratios = ratios

        # Compute the effective size: scale so that the largest dataset
        # (relative to its ratio) determines the epoch length.
        max_scaled = max(
            len(ds) / ratio for ds, ratio in zip(datasets, ratios) if ratio > 0
        )
        self._total_size = int(max_scaled)

        # Compute per-dataset sample counts
        self._counts = [int(self._total_size * r) for r in ratios]
        # Adjust last count to match total
        self._counts[-1] = self._total_size - sum(self._counts[:-1])

        if max_images:
            self._total_size = min(self._total_size, max_images)
            # Scale counts proportionally
            scale = self._total_size / sum(self._counts)
            self._counts = [int(c * scale) for c in self._counts]
            self._counts[-1] = self._total_size - sum(self._counts[:-1])

        # Build index mapping: (dataset_idx, sample_idx) for each global index
        self._index_map: list[tuple[int, int]] = []
        for ds_idx, count in enumerate(self._counts):
            ds_len = len(self._datasets[ds_idx])
            for i in range(count):
                # Wrap around if we need more samples than the dataset has
                self._index_map.append((ds_idx, i % ds_len))

    def __len__(self):
        return len(self._index_map)

    def __getitem__(self, idx: int):
        ds_idx, sample_idx = self._index_map[idx]
        return self._datasets[ds_idx][sample_idx]
