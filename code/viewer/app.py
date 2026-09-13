#!/usr/bin/env python3
"""Streamlit viewer for CLEAR-EC training data (memmap cache + mask pipeline)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.data.cache import open_image_cache
from src.data.config import MetricConfig, SegConfig, config_hash
from src.data.crop import crop_rng_for_image, random_crop_bbox
from src.data.mask_cache import get_or_compute_mask, mask_path
from src.data.metrics import metrics_from_mask
from src.infer_cellpose_sam import visualize_segmentation
from src.utils.evaluate import evaluate_results
from viewer._cache import METRIC_COLS, load_viewer_frame, resolve_repo_data_dir, split_summary

# Frozen overnight Cellpose (val-selected): diameter_40, crop 0.50.
DEFAULT_DIAMETER = 40.0
DEFAULT_CROP_FRAC = 0.50
VIEWER_URL = "http://localhost:8501"


def _default_paths() -> tuple[Path, Path, Path]:
    data_dir = resolve_repo_data_dir(CODE_ROOT)
    return (
        data_dir / "cache",
        data_dir / "final_train_ids.csv",
        data_dir / "train_green",
    )


def _apply_query_defaults() -> None:
    """Seed widget state from ?id=&split=&crop=&diameter= on first load."""
    if st.session_state.get("_qp_applied"):
        return
    qp = st.query_params
    if "split" in qp and qp["split"] in ("all", "train", "val", "test"):
        st.session_state["viewer_split"] = qp["split"]
    if "id" in qp and qp["id"]:
        st.session_state["viewer_id"] = qp["id"]
    if "crop" in qp:
        try:
            st.session_state["crop_frac"] = float(qp["crop"])
        except ValueError:
            pass
    if "diameter" in qp:
        try:
            st.session_state["diameter"] = float(qp["diameter"])
        except ValueError:
            pass
    st.session_state["_qp_applied"] = True


def _percent_errors(pred: dict, gt_row: pd.Series) -> dict[str, float | None]:
    errors: dict[str, float | None] = {}
    for col in METRIC_COLS:
        gt_val = gt_row.get(col)
        pred_val = pred.get(col)
        if pd.isna(gt_val) or gt_val == 0 or pred_val is None:
            errors[col] = None
        else:
            errors[col] = abs(float(pred_val) - float(gt_val)) / float(gt_val) * 100
    return errors


def _find_green_reference(green_dir: Path, image_id: str) -> Path | None:
    if not green_dir.is_dir():
        return None
    stem = str(image_id).strip()
    for ext in (".tiff", ".tif", ".png", ".bmp"):
        candidate = green_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


@st.cache_resource
def load_cellpose_model(model_type: str, use_gpu: bool):
    from cellpose import models

    return models.Cellpose(gpu=use_gpu, model_type=model_type)


@st.cache_resource(show_spinner="Opening memmap cache…")
def cached_memmap(cache_dir: str):
    memmap, _ = open_image_cache(cache_dir)
    return memmap


@st.cache_data(show_spinner="Loading cache index…")
def cached_viewer_frame(cache_dir: str, labels_csv: str, split: str):
    _, frame_df = load_viewer_frame(cache_dir, labels_csv, split=split)
    return frame_df


def main() -> None:
    st.set_page_config(page_title="CLEAR-EC Training Viewer", layout="wide")
    st.title("CLEAR-EC Training Data Viewer")
    st.caption(
        f"Defaults match the frozen overnight Cellpose: diameter={DEFAULT_DIAMETER:.0f} px, "
        f"crop={DEFAULT_CROP_FRAC:.2f}. Deep-link with "
        f"`{VIEWER_URL}/?id=<ID>&split=test`."
    )

    _apply_query_defaults()
    st.session_state.setdefault("viewer_split", "all")
    st.session_state.setdefault("diameter", DEFAULT_DIAMETER)
    st.session_state.setdefault("crop_frac", DEFAULT_CROP_FRAC)
    default_cache, default_labels, default_green = _default_paths()

    with st.sidebar:
        st.header("Data paths")
        cache_dir = Path(
            st.text_input("Cache directory", value=str(default_cache))
        ).expanduser()
        labels_csv = Path(
            st.text_input("Labels CSV", value=str(default_labels))
        ).expanduser()
        green_dir = Path(
            st.text_input("Green ROI directory (optional)", value=str(default_green))
        ).expanduser()

        st.header("Split & image")
        split = st.selectbox(
            "Split filter",
            ["all", "train", "val", "test"],
            key="viewer_split",
        )

        try:
            memmap = cached_memmap(str(cache_dir))
            frame_df = cached_viewer_frame(str(cache_dir), str(labels_csv), split)
            counts = split_summary(cache_dir)
            total = len(pd.read_csv(cache_dir / "index.csv"))
            if split == "all":
                st.caption(f"{len(frame_df)} images (cache N={total} total)")
            else:
                st.caption(
                    f"{len(frame_df)} images in {split} split "
                    f"(cache N={total} total; {split}={counts.get(split, 0)})"
                )
        except Exception as exc:
            st.error(f"Failed to load cache: {exc}")
            st.stop()

        id_options = frame_df["ID"].astype(str).tolist()
        if not id_options:
            st.error(f"No images in split {split!r}.")
            st.stop()
        if "viewer_id" not in st.session_state or st.session_state["viewer_id"] not in id_options:
            st.session_state["viewer_id"] = id_options[0]

        col_prev, col_next = st.columns(2)
        cur_pos = id_options.index(st.session_state["viewer_id"])
        with col_prev:
            if st.button("Previous", use_container_width=True, disabled=cur_pos == 0):
                st.session_state["viewer_id"] = id_options[cur_pos - 1]
                st.rerun()
        with col_next:
            if st.button("Next", use_container_width=True, disabled=cur_pos + 1 >= len(id_options)):
                st.session_state["viewer_id"] = id_options[cur_pos + 1]
                st.rerun()

        selected_id = st.selectbox("Image ID", id_options, key="viewer_id")
        row = frame_df.loc[frame_df["ID"].astype(str) == selected_id].iloc[0]
        idx = int(row["idx"])

        st.text_input("Cache index (idx)", value=str(idx), disabled=True)

        st.header("Segmentation (SegConfig)")
        model_type = st.selectbox("Model type", ["cyto", "cyto2", "nuclei"], index=0)
        diameter = st.number_input(
            "Diameter (px, 0 = auto)",
            min_value=0.0,
            step=1.0,
            key="diameter",
        )
        flow_threshold = st.slider("Flow threshold", 0.0, 1.0, 0.4, 0.05)
        cellprob_threshold = st.slider("Cellprob threshold", -6.0, 6.0, 0.0, 0.5)
        min_size = st.number_input("Min size", min_value=1, value=15, step=1)
        use_tile = st.checkbox("Tile inference", value=True)

        st.header("Metrics (MetricConfig)")
        crop_frac = st.slider(
            "Random crop fraction",
            0.1,
            1.0,
            step=0.05,
            key="crop_frac",
        )
        crop_seed = st.number_input("Random crop seed", value=42, step=1)

        device = st.radio("Device", ["cuda", "cpu"], index=0)
        show_green = st.checkbox(
            "Show green ROI reference",
            value=_find_green_reference(green_dir, selected_id) is not None,
        )

        run_seg = st.button("Run segmentation", type="primary", use_container_width=True)

    image_gray = memmap[idx]
    image_id = str(row["ID"])

    left, right = st.columns(2)
    with left:
        st.subheader("Input image")
        st.image(image_gray, use_container_width=True, caption=f"idx={idx}  ID={image_id}")
    with right:
        st.subheader("Segmentation overlay")

    gt_cols = st.columns(4)
    gt_cols[0].metric("Slide", str(row.get("slide_id", "")))
    for i, col_name in enumerate(METRIC_COLS, start=1):
        val = row.get(col_name)
        gt_cols[i].metric(f"GT {col_name}", f"{val:.2f}" if pd.notna(val) else "—")

    pred_metrics: dict | None = None
    cache_hit = False
    crop_bbox = None

    seg_config = SegConfig(
        model_type=model_type,
        diameter=diameter if diameter > 0 else None,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=int(min_size),
        tile=use_tile,
        net_avg=False,
        batch_size=8,
    )
    metric_config = MetricConfig(
        random_crop_frac=crop_frac,
        random_crop_seed=int(crop_seed),
    )
    seg_hash = config_hash(seg_config)

    if run_seg or mask_path(cache_dir, seg_hash, idx).exists():
        path = mask_path(cache_dir, seg_hash, idx)
        cache_hit = path.exists() and not run_seg

        if run_seg:
            use_gpu = device == "cuda"
            model = load_cellpose_model(model_type, use_gpu)
            had_mask = path.exists()
            masks_full = get_or_compute_mask(
                idx=idx,
                image_gray=image_gray,
                seg_config=seg_config,
                model=model,
                cache_dir=cache_dir,
                skip_existing=True,
            )
            cache_hit = had_mask
        else:
            from src.data.mask_cache import load_mask

            masks_full = load_mask(cache_dir, seg_hash, idx)

        image_for_crop = np.stack([image_gray] * 3, axis=-1)
        rng = crop_rng_for_image(image_for_crop, metric_config.random_crop_seed)
        crop_bbox = random_crop_bbox(
            image_for_crop.shape,
            frac=metric_config.random_crop_frac or DEFAULT_CROP_FRAC,
            rng=rng,
        )

        pred_metrics = metrics_from_mask(
            masks_full,
            image_gray,
            image_id,
            metric_config,
            crop_bbox=crop_bbox,
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            overlay_path = tmp.name
        visualize_segmentation(
            image=image_gray,
            masks=masks_full,
            image_id=image_id,
            save_path=overlay_path,
            show=False,
            crop_bbox=crop_bbox,
        )
        with right:
            if cache_hit:
                st.success(f"Mask cache hit (seg_hash={seg_hash})")
            elif run_seg:
                st.info(f"Computed mask saved (seg_hash={seg_hash})")
            st.image(overlay_path, use_container_width=True)

    if pred_metrics:
        st.subheader("Predicted vs ground truth")
        pred_cols = st.columns(4)
        pred_cols[0].write("")
        for i, col_name in enumerate(METRIC_COLS, start=1):
            pred_cols[i].metric(f"Pred {col_name}", f"{pred_metrics[col_name]:.2f}")

        errors = _percent_errors(pred_metrics, row)
        err_cols = st.columns(4)
        err_cols[0].write("**% error**")
        for i, col_name in enumerate(METRIC_COLS, start=1):
            err = errors[col_name]
            err_cols[i].write(f"{err:.1f}%" if err is not None else "—")

        pred_df = pd.DataFrame([{k: pred_metrics[k] for k in ("ID", *METRIC_COLS)}])
        gt_df = pd.DataFrame([{k: row[k] for k in ("ID", *METRIC_COLS)}])
        _, avg_err = evaluate_results(pred_df, gt_df)
        if not avg_err.empty:
            st.caption(
                "Average percent error (evaluate.py formula): "
                + ", ".join(
                    f"{col.replace(' Error (%)', '')}={avg_err[col].iloc[0]:.1f}%"
                    for col in avg_err.columns
                    if "Error" in col
                )
            )
    elif not run_seg:
        with right:
            st.info('Click "Run segmentation" to compute or load a cached mask.')

    if show_green:
        green_path = _find_green_reference(green_dir, image_id)
        if green_path:
            st.subheader("Green ROI reference")
            st.image(str(green_path), use_container_width=True, caption=str(green_path.name))
        else:
            st.warning(f"No green reference found for ID={image_id!r} under {green_dir}")


if __name__ == "__main__":
    main()
