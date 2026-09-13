"""
CLEAR-EC submission algorithm.

Grand Challenge runtime contract:
    /input/inputs.json                                         metadata
    /input/images/corneal-specular-microscopy-image/*.mha      input image
    /output/cell-density.json                                  CD prediction
    /output/coefficient-of-variation.json                      CV prediction
    /output/hexagonality.json                                  HEX prediction

Pipeline: load the MHA; if /opt/ml/model/submission.json describes an ensemble
bundle, run every member (ConvNeXt-Tiny / ConvNeXt-V2-Tiny whole-image regressors)
with flip TTA and emit the weighted geometric mean of CD / CV / HEX. Without a
bundle, fall back to the Cellpose baseline.
"""

import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.infer_cellpose_sam import get_segmentation
from src.io_utils import load_image


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("/opt/app/resources")
MODEL_PATH = Path("/opt/ml/model")
BAKED_MODEL_PATH = Path("/opt/app/model")  # bundle copied into the image; /opt/ml/model (platform-mounted) wins if present

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

    bundle_dir = next((d for d in (MODEL_PATH, BAKED_MODEL_PATH) if (d / "submission.json").exists()), None)
    if bundle_dir is not None:
        print(f"Using model bundle at {bundle_dir}")
        prediction = predict_model_bundle(image_path, bundle_dir)
    else:
        prediction = predict_baseline(image_path)

    cd = float(prediction["CD"])
    cv = float(prediction["CV"])
    hex_ = float(prediction["HEX"])
    if not np.isfinite([cd, cv, hex_]).all():
        raise ValueError("Predictions must be finite JSON numbers")
    print(f"Predictions: CD={cd:.2f}  CV={cv:.2f}  HEX={hex_:.2f}")

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    write_json_file(location=OUTPUT_PATH / "cell-density.json", content=cd)
    write_json_file(location=OUTPUT_PATH / "coefficient-of-variation.json", content=cv)
    write_json_file(location=OUTPUT_PATH / "hexagonality.json", content=hex_)
    return 0


def predict_model_bundle(image_path: Path, model_dir: Path) -> dict:
    """Apply the exported ensemble bundle to one image.

    submission.json lists members (checkpoint file, weight, sha256); each member is a
    whole-image regressor. Per member: geometric mean over flip views; across members:
    weighted geometric mean; then physical clamps.
    """
    import hashlib
    from src.training.common import METRICS, denormalize_targets
    from src.training.regression_cnn import (
        RegressionConfig, _inverse_target_space, build_regression_model,
    )

    metadata = load_json_file(location=model_dir / "submission.json")
    if metadata.get("method") != "ensemble":
        raise ValueError("Unknown submission bundle method")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(_cpu_budget())
    views = {"none": [(False, False)],
             "flips": [(False, False), (True, False), (False, True), (True, True)]}[metadata.get("tta", "none")]
    # Verify every member checksum before touching the image or any model.
    for member in metadata["members"]:
        path = model_dir / member["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != member["sha256"]:
            raise ValueError(f"Submission checkpoint checksum mismatch for {member['file']}")
    image = load_image(image_path)[..., 0].copy()
    x_u8 = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)

    log_preds, weights = [], []
    for member in metadata["members"]:
        path = model_dir / member["file"]
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config = dict(ckpt["config"])
        config.setdefault("uint8_inputs", False)
        cfg = RegressionConfig(**config)
        model = build_regression_model(cfg, load_pretrained=False)
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        stats = ckpt["target_stats"]
        x = x_u8 if cfg.uint8_inputs else x_u8.float() / 255.0
        member_logs = []
        with torch.inference_mode():
            for hflip, vflip in views:
                xv = x
                if hflip:
                    xv = torch.flip(xv, dims=[-1])
                if vflip:
                    xv = torch.flip(xv, dims=[-2])
                out = model(xv).float().cpu().numpy()
                out = _inverse_target_space(denormalize_targets(out, stats), cfg.target_space)[0]
                member_logs.append(np.log(np.clip(out, 1e-6, None)))
        log_preds.append(np.mean(member_logs, axis=0))
        weights.append(float(member.get("weight", 1.0)))
        del model
        print(f"member {member['file']}: {np.exp(log_preds[-1]).round(4).tolist()}", flush=True)

    prediction = np.exp(np.average(np.stack(log_preds), axis=0, weights=weights))
    clamp = metadata.get("clamp", {})
    prediction = np.array([
        float(np.clip(prediction[k], *clamp[m])) if m in clamp else float(prediction[k])
        for k, m in enumerate(METRICS)
    ])
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite model bundle prediction")
    return dict(zip(METRICS, map(float, prediction)))


def _cpu_budget() -> int:
    """Threads to use on CPU: the cgroup CPU quota if set, else the affinity count, capped at 8."""
    n = None
    for quota_file in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            parts = Path(quota_file).read_text().split()
            quota = int(parts[0])
            period = int(parts[1]) if len(parts) > 1 else int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0:
                n = max(1, quota // period)
            break
        except (OSError, ValueError, IndexError):
            continue
    if n is None:
        try:
            n = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            n = os.cpu_count() or 1
    return max(1, min(int(n), 8))


def predict_baseline(image_path: Path) -> dict:
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

    return predictions[0]


def get_interface_key() -> tuple:
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    socket_slugs = [sv["socket"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location: Path):
    with open(location) as f:
        return json.loads(f.read())


def write_json_file(*, location: Path, content) -> None:
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4, allow_nan=False))


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
