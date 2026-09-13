#!/usr/bin/env python3
"""Launch a durable two-fold experiment and write matched-checkpoint comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

REPO = Path("/home/visilant/CLEAR-EC")
EPOCHS = (8, 12, 20, 32, 40)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def launch(root):
    root.mkdir(parents=True, exist_ok=False)
    source = root / "source" / "code"
    current = Path(__file__).resolve().parents[1]
    for directory in ("src", "scripts", "tests"):
        shutil.copytree(current / directory, source / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(current / "evaluate.py", source / "evaluate.py")
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "timm:convnextv2_tiny.fcmae_ft_in22k_in1k",
        "folds": [0, 1], "n_folds": 5, "fold_seed": 42, "training_seed": 123,
        "epochs": 40, "cosine_epochs": 8, "tail_lr": 1e-6,
        "checkpoint_epochs": EPOCHS,
        "primary_comparison": "epoch 40 minus epoch 8, same run and held-out donors",
        "secondary_comparisons": "epochs 12, 20, 32; exploratory budget selection",
        "promotion": "Screen only; require consistent gains and remaining-fold confirmation before submission.",
        "caveats": ["Existing all-label CV folds, not the historical independent test split.",
                    "Known exact duplicates remain in the existing fold protocol.",
                    "Best validation checkpoint is selected and is not an unbiased final estimate."],
        "source_sha256": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(source.rglob("*.py"))},
    }
    write_json(root / "plan.json", plan)
    with (root / "driver.log").open("x") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(source / "scripts" / Path(__file__).name),
             "--mode", "supervise", "--results", str(root)],
            cwd=source, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write_json(root / "launch.json", {"pid": process.pid, "results": str(root)})
    print(json.dumps({"pid": process.pid, "results": str(root)}), flush=True)


def worker(root, fold, gpu):
    code = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code))
    import pandas as pd
    from src.training.regression_cnn import _slide_group_folds

    cache = REPO / "data/cache"
    index = pd.read_csv(cache / "index.csv")
    folds = _slide_group_folds(cache, 5, seed=42)
    val = set(folds[fold])
    train = {i for k, indices in folds.items() if k != fold for i in indices}
    assert train.isdisjoint(val)
    train_slides = set(index.loc[index.idx.isin(train), "slide_id"])
    val_slides = set(index.loc[index.idx.isin(val), "slide_id"])
    assert train_slides.isdisjoint(val_slides)
    output = root / f"fold{fold}"
    output.mkdir(exist_ok=False)
    write_json(output / "partition.json", {"train": sorted(train), "val": sorted(val),
                                           "n_train_slides": len(train_slides),
                                           "n_val_slides": len(val_slides)})
    command = [
        sys.executable, "-u", str(code / "scripts/run_training.py"),
        "--method", "regression", "--gpu", "0", "--cache_dir", str(cache),
        "--labels_csv", str(REPO / "data/final_train_ids.csv"),
        "--results_dir", str(output), "--seeds", "123",
        "--model", "timm:convnextv2_tiny.fcmae_ft_in22k_in1k",
        "--input_mode", "whole", "--loss", "relative", "--batch_size", "8",
        "--lr", "1e-4", "--weight_decay", "1e-4", "--amp", "--channels_last",
        "--augment_flips", "--ema", "0.999", "--sched", "cosine",
        "--warmup_epochs", "1", "--patience", "0", "--epochs", "40",
        "--sched_epochs", "8", "--save_epochs", ",".join(map(str, EPOCHS)),
        "--clip_grad", "1.0", "--fold", str(fold), "--n_folds", "5",
    ]
    write_json(output / "command.json", {"argv": command, "physical_gpu": gpu})
    subprocess.run(command, check=True, cwd=code)
    score_checkpoints(output, cache, sorted(val))


def score_checkpoints(output, cache, indices):
    import torch
    import pandas as pd
    from scripts.night_predict import predict, VIEWS
    from src.data.cache import open_image_cache
    from src.training.common import METRICS, labels_for_indices, score_by_id
    from src.training.regression_cnn import RegressionConfig, build_regression_model

    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    memmap, _ = open_image_cache(cache)
    frame = labels_for_indices(cache, REPO / "data/final_train_ids.csv", indices)
    directory = output / "regression_cnn/seed_123"
    scores = {}
    for epoch in EPOCHS:
        checkpoint = torch.load(directory / f"epoch_{epoch:03d}.pt", map_location="cpu", weights_only=False)
        cfg = RegressionConfig(**checkpoint["config"])
        model = build_regression_model(cfg, load_pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        values = predict(model, checkpoint["target_stats"], cfg, memmap,
                         list(frame.idx), device, VIEWS["flips"], batch_size=8)
        predictions = frame[["idx", "ID"]].copy()
        for j, metric in enumerate(METRICS):
            predictions[metric] = values[:, j]
        predictions.to_csv(output / f"epoch_{epoch:03d}_flips.csv", index=False)
        scores[str(epoch)] = score_by_id(predictions, frame, expected_ids=frame.ID)
        write_json(output / "checkpoint_scores.json", scores)
        print(f"epoch {epoch} flip-TTA scores: {scores[str(epoch)]}", flush=True)
        del model, checkpoint
    write_json(output / "completion.json", {"status": "complete", "scores": scores})


def report(root):
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.training.common import labels_for_indices, score_by_id

    scores = {}
    for epoch in EPOCHS:
        predictions = pd.concat([pd.read_csv(root / f"fold{k}/epoch_{epoch:03d}_flips.csv")
                                 for k in (0, 1)], ignore_index=True)
        assert predictions.ID.is_unique and predictions.idx.is_unique
        frame = labels_for_indices(REPO / "data/cache", REPO / "data/final_train_ids.csv",
                                   predictions.idx.tolist())
        scores[str(epoch)] = score_by_id(predictions, frame, expected_ids=frame.ID)
    write_json(root / "comparison.json", scores)
    lines = ["# Long V2-Tiny training: two-fold screening", "",
             "Matched eight-epoch cosine prefix, then constant LR 1e-6. Lower is better.", "",
             "| Epoch | CD | CV | HEX | Mean | Delta vs epoch 8 |",
             "|---|---:|---:|---:|---:|---:|"]
    baseline = scores["8"]["mean"]
    for epoch in EPOCHS:
        row = scores[str(epoch)]
        lines.append(f"| {epoch} | {row['CD']:.4f} | {row['CV']:.4f} | {row['HEX']:.4f} | "
                     f"{row['mean']:.4f} | {row['mean'] - baseline:+.4f} |")
    lines += ["", "Primary comparison: epoch 40 versus epoch 8. Other budgets are exploratory.",
              "These are two existing CV folds, not a full OOF or untouched test estimate.",
              "Remaining-fold confirmation is required before any submission decision."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


def supervise(root):
    processes = []
    status = {"status": "running", "pid": os.getpid(), "workers": {}}
    for fold, gpu in ((0, 0), (1, 1)):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
        with (root / f"fold{fold}.log").open("x") as log:
            process = subprocess.Popen([
                sys.executable, "-u", str(Path(__file__).resolve()), "--mode", "worker",
                "--results", str(root), "--fold", str(fold), "--gpu", str(gpu),
            ], env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        processes.append((fold, process))
        status["workers"][str(fold)] = {"pid": process.pid, "gpu": gpu, "status": "running"}
    write_json(root / "status.json", status)
    for fold, process in processes:
        code = process.wait()
        status["workers"][str(fold)].update(exit_code=code, status="complete" if code == 0 else "failed")
        write_json(root / "status.json", status)
    if all(p.returncode == 0 for _, p in processes):
        try:
            report(root)
            status["status"] = "complete"
        except Exception as exc:
            status.update(status="report_failed", error=str(exc))
            raise
        finally:
            write_json(root / "status.json", status)
    else:
        status["status"] = "failed"
        write_json(root / "status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("launch", "supervise", "worker"), default="launch")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    root = args.results.resolve()
    if args.mode == "launch":
        launch(root)
    elif args.mode == "supervise":
        supervise(root)
    else:
        worker(root, args.fold, args.gpu)
