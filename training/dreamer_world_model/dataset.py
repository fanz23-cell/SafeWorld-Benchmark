"""
Goal2 offline dataset loader.

Each .npz file is one episode chunk with keys:
  image  (T, 60) float32   — vector observation (named 'image' by export convention)
  action (T, 2)  float32
  reward (T,)    float32
  is_first (T,)  bool
  is_last  (T,)  bool
  is_terminal (T,) bool
  cost     (T,)  float32
  speed    (T,)  float32
  goal_distance (T,) float32
  nearest_hazard_distance (T,) float32
  nearest_vase_distance   (T,) float32
  level    (T,)  int32
  bucket_success / bucket_near_success / bucket_failure_or_recovery  (T,) bool

The loader slices each episode into overlapping windows of length seq_len,
then serves random batches of shape (batch_size, seq_len, ...).
"""

import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Keys we actually feed to the world model
OBS_KEY = "image"
ACT_KEY = "action"
REWARD_KEY = "reward"
TERMINAL_KEY = "is_terminal"
FIRST_KEY = "is_first"
AUX_KEYS = [
    "cost", "speed", "goal_distance",
    "nearest_hazard_distance", "nearest_vase_distance",
    "human_distance",
]


class Goal2EpisodeDataset(Dataset):
    """
    Loads all .npz episode chunks and slices them into fixed-length windows.
    Each item is a dict of tensors with shape (seq_len, ...).
    """

    def __init__(self, data_dir: str, seq_len: int, split: str = "train",
                 val_fraction: float = 0.05, seed: int = 42):
        self.seq_len = seq_len
        self.windows: List[Dict[str, np.ndarray]] = []

        npz_files = sorted(Path(data_dir).glob("*.npz"))
        assert len(npz_files) > 0, f"No .npz files found in {data_dir}"

        rng = random.Random(seed)
        rng.shuffle(npz_files)
        n_val = max(1, int(len(npz_files) * val_fraction))
        if split == "val":
            npz_files = npz_files[:n_val]
        else:
            npz_files = npz_files[n_val:]

        for path in npz_files:
            ep = np.load(path)
            T = ep[OBS_KEY].shape[0]
            if T < seq_len:
                continue
            # Slice into non-overlapping windows; drop the tail
            for start in range(0, T - seq_len + 1, seq_len):
                window = {
                    "obs":      ep[OBS_KEY][start:start + seq_len].astype(np.float32),
                    "action":   ep[ACT_KEY][start:start + seq_len].astype(np.float32),
                    "reward":   ep[REWARD_KEY][start:start + seq_len].astype(np.float32),
                    "terminal": ep[TERMINAL_KEY][start:start + seq_len].astype(np.float32),
                    "is_first": ep[FIRST_KEY][start:start + seq_len].astype(np.float32),
                }
                for k in AUX_KEYS:
                    arr = ep[k][start:start + seq_len].astype(np.float32)
                    # replace NaN/inf with 0 (missing sensor readings)
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    window[k] = arr
                self.windows.append(window)

        assert len(self.windows) > 0, f"No windows created from {data_dir} (split={split})"

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        w = self.windows[idx]
        return {k: torch.from_numpy(v) for k, v in w.items()}


def make_dataloader(data_dir: str, seq_len: int, batch_size: int,
                    split: str = "train", num_workers: int = 4) -> DataLoader:
    ds = Goal2EpisodeDataset(data_dir, seq_len, split=split)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
