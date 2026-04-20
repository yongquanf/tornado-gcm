"""Gap-tolerant data loading for NeuralGCM training.

Implements GapTolerantSampling that handles datasets with missing timesteps
by splitting into contiguous segments and sampling windows from each.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclasses.dataclass
class TemporalWindow:
    """A valid training window from a contiguous segment.

    Attributes:
        start_idx: starting index into the full time axis.
        length: number of timesteps in this window.
        weight: sampling weight (0.5 for short windows, 1.0 normal).
    """

    start_idx: int
    length: int
    weight: float = 1.0


def find_contiguous_segments(
    times: np.ndarray,
    gap_factor: float = 1.5,
) -> list[tuple[int, int]]:
    """Split a time axis into contiguous segments.

    A gap is detected where the time difference exceeds gap_factor * median(dt).

    Returns:
        list of (start_idx, end_idx) pairs (end exclusive).
    """
    if len(times) < 2:
        return [(0, len(times))]

    dt = np.diff(times)
    median_dt = np.median(dt)
    gap_mask = dt > gap_factor * median_dt

    segments = []
    start = 0
    for i, is_gap in enumerate(gap_mask):
        if is_gap:
            segments.append((start, i + 1))
            start = i + 1
    segments.append((start, len(times)))
    return segments


def gap_tolerant_sampling(
    times: np.ndarray,
    window_length: int,
    max_missing_rate: float = 0.05,
    gap_factor: float = 1.5,
) -> list[TemporalWindow]:
    """Build valid training windows tolerating temporal gaps.

    Algorithm (from PZHA §GapTolerantSampling):
      1. Infer nominal dt from median of diffs
      2. Detect gaps where dt > 1.5 × nominal
      3. Split into contiguous segments
      4. For segments >= W*(1-r_max): standard sliding window
      5. For segments >= W/2 but shorter: use as short-rollout sample (weight=0.5)

    Args:
        times: 1-D array of timestamps (sorted).
        window_length: target rollout window length W.
        max_missing_rate: maximum tolerable missing fraction r_max.
        gap_factor: gap detection threshold multiplier.

    Returns:
        List of TemporalWindow describing valid sample origins.
    """
    segments = find_contiguous_segments(times, gap_factor)
    min_full = int(window_length * (1.0 - max_missing_rate))
    min_short = window_length // 2

    windows: list[TemporalWindow] = []
    for start, end in segments:
        seg_len = end - start
        if seg_len >= min_full:
            # Full-length sliding windows
            for i in range(start, end - window_length + 1):
                windows.append(TemporalWindow(i, window_length, weight=1.0))
        elif seg_len >= min_short:
            # Short windows with reduced weight
            w = seg_len
            for i in range(start, end - w + 1):
                windows.append(TemporalWindow(i, w, weight=0.5))

    return windows


class TrajectoryDataset(Dataset):
    """Dataset that yields trajectory windows from a weather dataset.

    Uses gap-tolerant sampling to handle missing timesteps.
    """

    def __init__(
        self,
        data: dict[str, np.ndarray],
        times: np.ndarray,
        window_length: int,
        max_missing_rate: float = 0.05,
    ):
        """
        Args:
            data: dict of variable_name → array with shape (T, ...).
            times: 1-D time coordinate, length T.
            window_length: target rollout window length.
            max_missing_rate: tolerance for gaps.
        """
        self.data = data
        self.times = times
        self.windows = gap_tolerant_sampling(
            times, window_length, max_missing_rate
        )
        self._weights = np.array([w.weight for w in self.windows])
        self._weights /= self._weights.sum()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        win = self.windows[idx]
        sl = slice(win.start_idx, win.start_idx + win.length)
        sample = {}
        for key, arr in self.data.items():
            sample[key] = torch.from_numpy(arr[sl].copy()).float()
        sample["weight"] = torch.tensor(win.weight, dtype=torch.float32)
        return sample


class WeightedRandomSampler(torch.utils.data.Sampler):
    """Sampler that respects per-window weights from gap-tolerant sampling."""

    def __init__(self, dataset: TrajectoryDataset, num_samples: int | None = None):
        self.weights = torch.from_numpy(dataset._weights).double()
        self.num_samples = num_samples or len(dataset)

    def __iter__(self):
        indices = torch.multinomial(
            self.weights, self.num_samples, replacement=True
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


def build_dataloader(
    data: dict[str, np.ndarray],
    times: np.ndarray,
    window_length: int,
    batch_size: int,
    num_workers: int = 0,
    max_missing_rate: float = 0.05,
    use_distributed_sampler: bool = False,
) -> DataLoader:
    """Build a DataLoader with gap-tolerant weighted sampling.

    Args:
        use_distributed_sampler: if True and distributed is active,
            use DistributedSampler for cross-process data partitioning
            (falls back to WeightedRandomSampler on single GPU).
    """
    dataset = TrajectoryDataset(data, times, window_length, max_missing_rate)
    if use_distributed_sampler:
        from pytorch_src.distributed import get_distributed_sampler
        sampler = get_distributed_sampler(dataset)
        if sampler is None:
            # Single-process: fall back to gap-tolerant weighted sampling
            sampler = WeightedRandomSampler(dataset)
    else:
        sampler = WeightedRandomSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
