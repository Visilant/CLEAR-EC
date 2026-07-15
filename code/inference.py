"""
CLEAR-EC submission algorithm.

Grand Challenge runtime contract:
    /input/inputs.json                                         metadata
    /input/images/corneal-specular-microscopy-image/*.mha      input image
    /output/cell-density.json                                  CD prediction
    /output/coefficient-of-variation.json                      CV prediction
    /output/hexagonality.json                                  HEX prediction

Pipeline: load the MHA, run Cellpose v1.0 on the full image, restrict
metrics to a deterministic random crop (40% of H x 40% of W with seed 42),
emit CD / CV / HEX.
"""

import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.infer_cellpose_sam import get_segmentation


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("/opt/app/resources")

SEED = 42
RANDOM_CROP_FRAC = 0.4
MODEL_TYPE = "cyto"
DIAMETER = None
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0
MIN_SIZE = 15


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run() -> int:
    interface_key = get_interface_key()
    handler = {
        ("corneal-specular-microscopy-image",): interf0_handler,
    }[interface_key]
    return handler()


def interf0_handler() -> int:
    set_seed(SEED)
    _show_torch_cuda_info()

    image_dir = INPUT_PATH / "images" / "corneal-specular-microscopy-image"
    image_files = (
        glob.glob(str(image_dir / "*.mha"))
        + glob.glob(str(image_dir / "*.tif"))
        + glob.glob(str(image_dir / "*.tiff"))
    )
    if not image_files:
        raise FileNotFoundError(f"No MHA/TIFF input found under {image_dir}")
    image_path = Path(sorted(image_files)[0])
    print(f"Processing input: {image_path.name}")

    predictions, _ = get_segmentation(
        image_paths=[image_path],
        plotting=False,
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
        min_size=MIN_SIZE,
        model_type=MODEL_TYPE,
        diameter=DIAMETER,
        annotations=None,
        match_dots=False,
        vis_output_dir=str(OUTPUT_PATH / "visualizations"),
        random_crop_frac=RANDOM_CROP_FRAC,
        random_crop_seed=SEED,
        tile=True,
    )

    if not predictions:
        raise RuntimeError("Segmentation returned no predictions")

    prediction = predictions[0]
    cd = float(prediction["CD"])
    cv = float(prediction["CV"])
    hex_ = float(prediction["HEX"])
    print(f"Predictions: CD={cd:.2f}  CV={cv:.2f}  HEX={hex_:.2f}")

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    write_json_file(location=OUTPUT_PATH / "cell-density.json", content=cd)
    write_json_file(location=OUTPUT_PATH / "coefficient-of-variation.json", content=cv)
    write_json_file(location=OUTPUT_PATH / "hexagonality.json", content=hex_)

    return 0


def get_interface_key() -> tuple:
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    socket_slugs = [sv["socket"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location: Path):
    with open(location) as f:
        return json.loads(f.read())


def write_json_file(*, location: Path, content) -> None:
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def _show_torch_cuda_info() -> None:
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: {(current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
