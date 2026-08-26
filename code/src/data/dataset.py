"""PyTorch Dataset for memmap-backed inference prefetch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset

from src.data.cache import open_image_cache


class ClearECDataset(Dataset):
    """Lightweight index into the shared memmap (inference prefetch only)."""

    def __init__(
        self,
        cache_dir: Path,
        indices: list[int] | None = None,
    ):
        self.memmap, self.index_df = open_image_cache(cache_dir)
        if indices is None:
            self.indices = self.index_df["idx"].astype(int).tolist()
        else:
            self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[int, str, np.ndarray]:
        idx = self.indices[i]
        row = self.index_df.loc[self.index_df["idx"] == idx].iloc[0]
        return idx, str(row["ID"]), self.memmap[idx]


def make_dataloader(
    dataset: ClearECDataset,
    *,
    num_workers: int = 8,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    batch_size: int = 1,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        shuffle=False,
    )
