"""Memmap image cache built from MHA files."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.io_utils import _load_mha
from src.data.splits import build_splits

IMAGE_FILENAME = "images_u8.npy"
INDEX_FILENAME = "index.csv"
EXPECTED_SHAPE = (972, 1296)
EXPECTED_COUNT = 9000


def _normalize_id(value: str) -> str:
    """Normalize CSV/MHA IDs so minor spacing variants still join."""
    s = " ".join(str(value).strip().split())
    s = re.sub(r"-\s+(ODCN|OSCN)$", r"-\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+(ODCN|OSCN)$", r"-\1", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)(ODCN|OSCN)$", r"-\1", s, flags=re.IGNORECASE)
    s = s.replace(" ", "")
    s = re.sub(r"-+", "-", s)
    return s.upper()


def _build_mha_lookup(data_dir: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for path in data_dir.glob("*.mha"):
        lookup[_normalize_id(path.stem)] = path
    return lookup


def _decode_gray_mha(path: Path) -> np.ndarray:
    """Decode a gray MHA into uint8 (H, W)."""
    arr = _load_mha(path)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected 2D gray image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def build_image_cache(
    data_dir: Path,
    labels_csv: Path,
    cache_dir: Path,
    *,
    seed: int = 42,
    force: bool = False,
    limit: int | None = None,
    expected_count: int = EXPECTED_COUNT,
) -> tuple[np.memmap, pd.DataFrame]:
    """Build or open the shared uint8 memmap and index.csv."""
    data_dir = Path(data_dir)
    labels_csv = Path(labels_csv)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    memmap_path = cache_dir / IMAGE_FILENAME
    index_path = cache_dir / INDEX_FILENAME

    labels_df = pd.read_csv(labels_csv)
    labels_df["ID"] = labels_df["ID"].astype(str).str.strip()

    mha_by_id = _build_mha_lookup(data_dir)

    rows: list[dict] = []
    for _, label_row in labels_df.iterrows():
        image_id = str(label_row["ID"]).strip()
        mha_path = mha_by_id.get(_normalize_id(image_id))
        if mha_path is None:
            raise FileNotFoundError(f"No MHA found for ID={image_id!r} under {data_dir}")
        rows.append(
            {
                "ID": image_id,
                "slide_id": str(label_row["slide_id"]),
                "mha_path": str(mha_path.resolve()),
            }
        )

    if limit is not None and limit > 0:
        rows = rows[:limit]

    n = len(rows)
    if limit is None and expected_count and n != expected_count:
        raise ValueError(f"Expected {expected_count} labeled images, got {n}")

    if (
        not force
        and memmap_path.exists()
        and index_path.exists()
    ):
        index_df = pd.read_csv(index_path)
        if len(index_df) == n:
            print(f"Cache already exists at {cache_dir} ({n} rows); skipping rebuild.")
            build_splits(index_df, cache_dir, seed=seed)
            return open_image_cache(cache_dir)
        print(
            f"Existing cache row count ({len(index_df)}) != expected ({n}); rebuilding."
        )

    print(f"Building memmap cache: {n} images -> {memmap_path}")
    memmap = np.memmap(
        memmap_path,
        dtype=np.uint8,
        mode="w+",
        shape=(n, *EXPECTED_SHAPE),
    )

    for idx, row in enumerate(rows):
        gray = _decode_gray_mha(Path(row["mha_path"]))
        if gray.shape != EXPECTED_SHAPE:
            raise ValueError(
                f"{row['ID']}: shape {gray.shape} != expected {EXPECTED_SHAPE}"
            )
        memmap[idx] = gray
        if (idx + 1) % 500 == 0 or idx + 1 == n:
            print(f"  decoded {idx + 1}/{n}")

    memmap.flush()

    index_df = pd.DataFrame(
        {
            "idx": range(n),
            "ID": [r["ID"] for r in rows],
            "slide_id": [r["slide_id"] for r in rows],
            "mha_path": [r["mha_path"] for r in rows],
        }
    )
    index_df.to_csv(index_path, index=False)
    print(f"Wrote index -> {index_path}")

    size_gb = memmap_path.stat().st_size / (1024**3)
    print(f"Memmap shape={memmap.shape}, dtype={memmap.dtype}, size={size_gb:.2f} GB")

    build_splits(index_df, cache_dir, seed=seed)
    return open_image_cache(cache_dir)


def open_image_cache(cache_dir: Path) -> tuple[np.memmap, pd.DataFrame]:
    """Open a read-only memmap and index DataFrame."""
    cache_dir = Path(cache_dir)
    memmap_path = cache_dir / IMAGE_FILENAME
    index_path = cache_dir / INDEX_FILENAME

    if not memmap_path.exists() or not index_path.exists():
        raise FileNotFoundError(
            f"Cache not found under {cache_dir}; run build_cache.py first."
        )

    index_df = pd.read_csv(index_path)
    n = len(index_df)
    memmap = np.memmap(
        memmap_path,
        dtype=np.uint8,
        mode="r",
        shape=(n, *EXPECTED_SHAPE),
    )
    return memmap, index_df
