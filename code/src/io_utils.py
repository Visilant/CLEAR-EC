from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


# MetaImage ElementType -> numpy dtype (matches convert_to_mha.py).
_MHA_DTYPE = {
    "MET_UCHAR": np.uint8,
    "MET_CHAR": np.int8,
    "MET_USHORT": np.uint16,
    "MET_SHORT": np.int16,
    "MET_UINT": np.uint32,
    "MET_INT": np.int32,
    "MET_FLOAT": np.float32,
    "MET_DOUBLE": np.float64,
}


def _load_mha(path: Path) -> np.ndarray:
    """Read an uncompressed 2D MetaImage (.mha) written by convert_to_mha.py."""
    with open(path, "rb") as f:
        data = f.read()
    sep = b"ElementDataFile = LOCAL\n"
    idx = data.find(sep)
    if idx < 0:
        raise ValueError(f"{path}: missing 'ElementDataFile = LOCAL' marker")
    header = data[:idx].decode("ascii")
    raw = data[idx + len(sep):]

    fields = {}
    for line in header.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fields[k.strip()] = v.strip()

    dim = fields["DimSize"].split()
    width, height = int(dim[0]), int(dim[1])
    dtype = np.dtype(_MHA_DTYPE[fields["ElementType"]])
    if fields.get("CompressedData", "False").lower() == "true":
        raise ValueError("Compressed MHA requires SimpleITK")
    if fields.get("BinaryDataByteOrderMSB", "False").lower() == "true":
        dtype = dtype.newbyteorder(">")
    else:
        dtype = dtype.newbyteorder("<")
    arr = np.frombuffer(raw, dtype=dtype).reshape(height, width).copy()
    return arr.astype(dtype.newbyteorder("="), copy=False)


def _load_mha_simpleitk(path: Path) -> np.ndarray:
    """Read an MHA via SimpleITK — handles compressed/uncompressed/all dtypes."""
    import SimpleITK  # imported lazily so non-MHA paths don't require it
    img = SimpleITK.ReadImage(str(path))
    return SimpleITK.GetArrayFromImage(img)


def load_image(path: str | Path) -> np.ndarray:
    """
    Load an image from the given path and return it as a numpy RGB array.

    Args:
        path (str | Path): Path to the image file (.mha, .tif/.tiff, .bmp, .png).
    """
    path = Path(path)
    if path.suffix.lower() == ".mha":
        # Prefer SimpleITK (handles compression and every dtype Grand Challenge
        # might send). Fall back to the hand-rolled parser for environments
        # without SimpleITK installed.
        try:
            arr = _load_mha_simpleitk(path)
        except ImportError:
            arr = _load_mha(path)
        # 3D volumes from SimpleITK come back as (Z, Y, X); endothelial frames
        # are single slice, so squeeze a leading singleton if present.
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        # Cellpose downstream expects 3-channel RGB; replicate grayscale.
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr
    return np.array(Image.open(path).convert("RGB"))
