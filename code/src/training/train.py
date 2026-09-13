"""Train a regression model on a split or fold and score a frozen checkpoint on a split."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.artifacts import atomic_path
from src.data.cache import open_image_cache
from src.data.splits import load_split
from src.training.common import labels_for_indices, score_by_id, write_manifest
from src.training.config import RegressionConfig
from src.training.data import MetricRegressionDataset, apply_exclusions
from src.training.folds import _slide_group_folds
from src.training.loop import _EMA, _build_scheduler, _run_epoch
from src.training.losses import _build_loss
from src.training.models import _load_checkpoint, build_regression_model
from src.training.predict import predict_indices
from src.training.targets import target_space_stats


def train_regression_cnn(
    cache_dir: Path,
    labels_csv: Path,
    results_dir: Path,
    *,
    gpu: int = 0,
    config: RegressionConfig | None = None,
) -> dict:
    """Fit on train, select checkpoint on val. Does not load the test split.

    --fold/--n_folds replace train/val with a slide-grouped fold; --all_data trains
    on train+val+test with no validation at all (fixed epochs, last-epoch checkpoint).
    """
    cfg = config or RegressionConfig()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if (results_dir / "best_model.pt").exists() or (results_dir / "last.pt").exists():
        raise FileExistsError(f"Checkpoint already exists in {results_dir}; use a new results_dir")
    if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.num_workers < 0 or cfg.cpu_threads < 1:
        raise ValueError("Invalid training epochs/batch_size/num_workers/cpu_threads")
    if cfg.sched_epochs < 0 or cfg.sched_epochs > cfg.epochs:
        raise ValueError("sched_epochs must be between 0 and epochs")
    if any(e < 1 or e > cfg.epochs for e in cfg.save_epochs):
        raise ValueError("save_epochs must be within the training budget")
    if cfg.fold >= 0 and cfg.all_data:
        raise ValueError("--fold and --all_data are mutually exclusive")
    if cfg.fold >= 0 and (cfg.n_folds < 2 or cfg.fold >= cfg.n_folds):
        raise ValueError("--fold requires 0 <= fold < n_folds and n_folds >= 2")
    torch.set_num_threads(cfg.cpu_threads)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    memmap, _index_df = open_image_cache(cache_dir)

    has_val = True
    if cfg.all_data:
        train_idx = sorted(
            set(load_split(cache_dir, "train"))
            | set(load_split(cache_dir, "val"))
            | set(load_split(cache_dir, "test"))
        )
        val_idx: list[int] = []
        has_val = False
    elif cfg.fold >= 0:
        folds = _slide_group_folds(cache_dir, cfg.n_folds, seed=42)
        val_idx = folds[cfg.fold]
        train_idx = sorted(i for k, idxs in folds.items() if k != cfg.fold for i in idxs)
    else:
        train_idx = load_split(cache_dir, "train")
        val_idx = load_split(cache_dir, "val")
    if cfg.train_fraction < 1.0:
        slide_of = dict(zip(_index_df["idx"].astype(int), _index_df["slide_id"].astype(str)))
        slides = sorted({slide_of[i] for i in train_idx})
        keep = set(np.random.default_rng(cfg.seed).permutation(slides)[: int(round(len(slides) * cfg.train_fraction))])
        train_idx = sorted(i for i in train_idx if slide_of[i] in keep)
        print(f"[regression_cnn seed={cfg.seed}] train_fraction={cfg.train_fraction}: {len(train_idx)} train images", flush=True)

    train_idx = apply_exclusions(train_idx, cfg.exclude_idx_file)
    train_df = labels_for_indices(cache_dir, labels_csv, train_idx)
    val_df = labels_for_indices(cache_dir, labels_csv, val_idx) if has_val else None

    stats = target_space_stats(train_df, cfg.target_space)
    train_ds = MetricRegressionDataset(memmap, train_df, stats, cfg.uint8_inputs,
                                       augment_flips=cfg.augment_flips, target_space=cfg.target_space)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = None
    if has_val:
        val_ds = MetricRegressionDataset(memmap, val_df, stats, cfg.uint8_inputs, target_space=cfg.target_space)
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=cfg.num_workers > 0,
        )

    model = build_regression_model(cfg).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    criterion = _build_loss(cfg.loss, stats, target_space=cfg.target_space).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    lr_scheduler = _build_scheduler(optimizer, cfg.sched, cfg.warmup_epochs,
                                    cfg.sched_epochs or cfg.epochs, len(train_loader))

    ema = _EMA(model, cfg.ema) if cfg.ema > 0 else None
    ema_update = (lambda: ema.update(model)) if ema is not None else None
    eval_model = ema.model if ema is not None else model

    no_early_stop = cfg.patience <= 0
    best_val = float("inf")
    best_state: dict | None = None
    stale = 0
    history: list[dict] = []
    last_epoch_completed = 0

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_mape = _run_epoch(
            model, train_loader, criterion, device, stats, optimizer, scaler=scaler, amp=cfg.amp,
            target_space=cfg.target_space, photometric=cfg.photometric,
            ema_update=ema_update, lr_scheduler=lr_scheduler, crop_scale=cfg.crop_scale, clip_grad=cfg.clip_grad,
        )
        row = {
            "epoch": epoch,
            "seconds": time.perf_counter() - epoch_start,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_mape_{k}": v for k, v in train_mape.items()},
        }
        val_mape = None
        if has_val:
            val_loss, val_mape = _run_epoch(eval_model, val_loader, criterion, device, stats,
                                            target_space=cfg.target_space)
            row["val_loss"] = val_loss
            row.update({f"val_mape_{k}": v for k, v in val_mape.items()})
        history.append(row)
        msg = (f"[regression_cnn seed={cfg.seed}] epoch {epoch}/{cfg.epochs} "
               f"train_loss={train_loss:.4f} lr={row['lr']:.2e}")
        if has_val:
            msg += f" val_loss={val_loss:.4f} val_mape_mean={val_mape.get('mean', float('nan')):.2f}%"
        print(msg, flush=True)

        last_epoch_completed = epoch
        if has_val:
            val_key = float(val_mape["mean"])
            if val_key < best_val:
                best_val = val_key
                best_state = {k: v.cpu().clone() for k, v in eval_model.state_dict().items()}
                with atomic_path(results_dir / "best_model.pt") as temporary:
                    torch.save({"model_state": best_state, "target_stats": stats,
                                "config": asdict(cfg), "epoch": epoch,
                                "val_mape": val_mape}, temporary)
                stale = 0
            else:
                stale += 1
        with atomic_path(results_dir / "history.csv") as temporary:
            pd.DataFrame(history).to_csv(temporary, index=False)
        if cfg.save_epochs:
            # Recovery state is separate from inference-only EMA checkpoints.
            with atomic_path(results_dir / "training_state.pt") as temporary:
                torch.save({
                    "model_state": model.state_dict(),
                    "ema_state": ema.model.state_dict() if ema is not None else None,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                    "scaler_state": scaler.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "numpy_rng_state": np.random.get_state(),
                    "target_stats": stats, "config": asdict(cfg), "epoch": epoch,
                    "history": history, "best_val": best_val, "stale": stale,
                    "train_indices": train_idx, "val_indices": val_idx,
                }, temporary)
        if epoch in cfg.save_epochs:
            with atomic_path(results_dir / f"epoch_{epoch:03d}.pt") as temporary:
                torch.save({"model_state": eval_model.state_dict(), "target_stats": stats,
                            "config": asdict(cfg), "epoch": epoch,
                            "val_mape": val_mape}, temporary)
        if has_val and not no_early_stop and stale >= cfg.patience:
            print(f"[regression_cnn seed={cfg.seed}] early stop at epoch {epoch}")
            break

    # 'last' checkpoint: --patience 0 keeps it for resuming/inspection; a no-val
    # run (--all_data / --fold has_val still true, so only all_data lands here
    # without a best checkpoint) always needs a checkpoint of its own.
    last_state = {k: v.cpu().clone() for k, v in eval_model.state_dict().items()}
    if no_early_stop or not has_val:
        with atomic_path(results_dir / "last.pt") as temporary:
            torch.save({"model_state": last_state, "target_stats": stats,
                        "config": asdict(cfg), "epoch": last_epoch_completed}, temporary)

    if has_val:
        if best_state is None:
            raise RuntimeError("No finite validation checkpoint was produced")
        eval_model.load_state_dict(best_state)
        ckpt_path = results_dir / "best_model.pt"
    else:
        eval_model.load_state_dict(last_state)
        ckpt_path = results_dir / "last.pt"

    pd.DataFrame(history).to_csv(results_dir / "history.csv", index=False)

    metrics: dict = {
        "n_train": len(train_df),
        "seed": cfg.seed,
        "epochs_run": last_epoch_completed,
    }
    if has_val:
        val_pred = predict_indices(eval_model, memmap, val_df, stats, device, batch_size=cfg.batch_size,
                                   uint8_inputs=cfg.uint8_inputs, target_space=cfg.target_space)
        val_pred.to_csv(results_dir / "predictions_val.csv", index=False)
        val_mape = score_by_id(val_pred, val_df, expected_ids=val_df.ID)
        metrics.update({"val_mape": val_mape, "val_best_mape_mean": best_val, "n_val": len(val_df)})
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    write_manifest(
        results_dir.parent.parent if results_dir.name.startswith("seed_") else results_dir.parent,
        method=f"regression_cnn_seed_{cfg.seed}",
        metrics=metrics,
        extra={"checkpoint": str(ckpt_path), "config": asdict(cfg)},
    )
    if has_val:
        print(f"[regression_cnn seed={cfg.seed}] val MAPE: {metrics['val_mape']}")
    else:
        print(f"[regression_cnn seed={cfg.seed}] all_data run complete, no validation")
    return metrics


def evaluate_regression_split(
    cache_dir: Path,
    labels_csv: Path,
    results_dir: Path,
    *,
    split: str,
    gpu: int = 0,
) -> dict:
    """Score a frozen regression checkpoint on a split (test only after freeze)."""
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model, stats, cfg = _load_checkpoint(Path(results_dir), device)
    memmap, _ = open_image_cache(cache_dir)
    indices = load_split(cache_dir, split)
    frame = labels_for_indices(cache_dir, labels_csv, indices)
    pred = predict_indices(model, memmap, frame, stats, device, batch_size=cfg.batch_size,
                           uint8_inputs=cfg.uint8_inputs, target_space=cfg.target_space)
    pred.to_csv(Path(results_dir) / f"predictions_{split}.csv", index=False)
    scores = score_by_id(pred, frame, expected_ids=frame.ID)
    metrics_path = Path(results_dir) / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics[f"{split}_mape"] = scores
    metrics[f"n_{split}"] = len(frame)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[regression_cnn seed={cfg.seed}] {split} MAPE: {scores}")
    return scores
