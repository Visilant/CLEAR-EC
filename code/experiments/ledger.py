"""Consolidated experiment ledger: every campaign under results/ -> docs/experiments.csv + docs/EXPERIMENTS.md.

Adapters read files only; the only recomputation is scoring concatenated out-of-fold prediction CSVs
against the labels (that is how the night's OOF numbers were produced). Prose-only facts, verdicts,
campaign titles and deleted artifacts come from experiments/ledger_manual.yaml.
"""
from __future__ import annotations

import csv
import fnmatch
import glob as globmod
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments import CODE, REPO
from src.training.common import METRICS, load_labels, mape_per_metric
from src.training.config import config_from_checkpoint, recipe_hash

COLUMNS = ["campaign", "run", "date", "model", "recipe", "seed", "fold", "epochs", "split", "n",
           "CD", "CV", "HEX", "mean", "ci_low", "ci_high", "reference", "status", "verdict", "path"]
SPLIT_ORDER = ["oof9000", "platform100", "oof2fold", "fold-k/5", "val892", "test906", "val300", "overlay19", "floor"]
MANUAL = CODE / "experiments" / "ledger_manual.yaml"


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date().isoformat()


def _date(value: str | None, fallback: Path) -> str:
    return value[:10] if value else _mtime(fallback)


def _row(**kw) -> dict:
    row = {c: "" for c in COLUMNS}
    row.update({k: v for k, v in kw.items() if v is not None})
    return row


def _split_of_indices(indices: str, n: int | None) -> str:
    if indices == "val":
        return "val892"
    if indices.startswith("fold:"):
        k, m = indices[5:].split("/")
        return f"fold-{k}/{m}"
    if indices == "all" or n == 9000:
        return "oof9000"
    return indices


def _labels() -> pd.DataFrame:
    return load_labels(REPO / "data" / "final_train_ids.csv").set_index("ID")


def _score_frame(frame: pd.DataFrame, labels: pd.DataFrame) -> dict:
    frame = frame.assign(ID=frame["ID"].astype(str).str.strip())
    if frame["ID"].duplicated().any():
        raise ValueError("duplicate IDs in prediction frame")
    gt = labels.loc[frame["ID"], list(METRICS)].to_numpy(float)
    return mape_per_metric(frame[list(METRICS)].to_numpy(float), gt)


# ---------------------------------------------------------------- generic run dirs (manifest.json)

def rows_from_manifests(results: Path) -> list[dict]:
    rows = []
    for manifest in sorted(results.glob("*/**/manifest.json")):
        campaign = manifest.relative_to(results).parts[0]
        data = json.loads(manifest.read_text())
        methods = {k: v for k, v in data.get("methods", {}).items() if k.startswith("regression_cnn")}
        if not methods:
            continue
        run = str(manifest.parent.relative_to(results / campaign)) if manifest.parent != results / campaign else "."
        for method, entry in methods.items():
            cfg_dict = entry.get("config") or {}
            metrics = entry.get("metrics") or {}
            try:
                cfg = config_from_checkpoint(cfg_dict)
                recipe, model = recipe_hash(cfg), cfg.model
                fold, n_folds, all_data, epochs = cfg.fold, cfg.n_folds, cfg.all_data, cfg.epochs
            except TypeError:
                recipe, model = "", cfg_dict.get("model", "")
                fold, n_folds, all_data, epochs = -1, 0, False, cfg_dict.get("epochs", "")
            val = metrics.get("val_mape") or {}
            if all_data:
                split, status = "none (all data)", "no-val"
            elif fold >= 0:
                split, status = f"fold-{fold}/{n_folds}", "done" if val else "failed"
            else:
                split, status = "val892", "done" if val else "failed"
            rows.append(_row(campaign=campaign, run=run if run != "." else f"({method})", date=_date(entry.get("updated_at") or data.get("created_at"), manifest),
                             model=model, recipe=recipe, seed=metrics.get("seed", cfg_dict.get("seed", "")),
                             fold="" if fold < 0 else fold, epochs=f"{metrics.get('epochs_run', '')}/{epochs}",
                             split=split, n=metrics.get("n_val", ""), CD=val.get("CD", ""), CV=val.get("CV", ""),
                             HEX=val.get("HEX", ""), mean=val.get("mean", ""), status=status,
                             path=_rel(manifest.parent)))
    return rows


# ---------------------------------------------------------------- scored predictions (night tta/oof, scores/, *_oof.json)

def _score_jsons(results: Path) -> list[Path]:
    found = set()
    for pattern in ("*/tta/*.json", "*/oof/*.json", "*/scores/*.json", "*/*_oof.json"):
        found.update(results.glob(pattern))
    return sorted(found)


def rows_from_scores(results: Path, labels: pd.DataFrame) -> list[dict]:
    rows, families = [], {}
    for path in _score_jsons(results):
        data = json.loads(path.read_text())
        if not {"ckpt_dir", "which", "indices", "mape"} <= set(data):
            continue
        campaign = path.relative_to(results).parts[0]
        ckpt = Path(data["ckpt_dir"])
        run = ckpt.parent.parent.name if ckpt.name.startswith("seed_") else ckpt.name
        which, tta, indices, n = data["which"], data.get("tta", "none"), data["indices"], data.get("n")
        split = _split_of_indices(indices, n)
        m = data["mape"]
        rows.append(_row(campaign=campaign, run=f"{run} [{which}+{tta}]", date=_mtime(path), model="", seed="",
                         epochs=data.get("epoch", ""), split=split, n=n, CD=m["CD"], CV=m["CV"], HEX=m["HEX"],
                         mean=m["mean"], status="scored", path=_rel(path)))
        if indices.startswith("fold:"):
            k, total = indices[5:].split("/")
            family = run[: -len(k)] if run.endswith(k) else run
            families.setdefault((campaign, family, which, tta, int(total)), {})[int(k)] = path.with_suffix(".csv")
    for (campaign, family, which, tta, total), members in families.items():
        if set(members) != set(range(total)) or not all(p.exists() for p in members.values()):
            continue
        frame = pd.concat([pd.read_csv(p) for _, p in sorted(members.items())], ignore_index=True)
        if frame["ID"].duplicated().any():
            continue
        m = _score_frame(frame, labels)
        rows.append(_row(campaign=campaign, run=f"{family} OOF over {total} folds [{which}+{tta}]", date=_mtime(members[0]),
                         split="oof9000" if len(frame) == 9000 else f"oof{len(frame)}", n=len(frame), CD=m["CD"], CV=m["CV"],
                         HEX=m["HEX"], mean=m["mean"], status="computed", path=_rel(members[0].parent)))
    return rows


def rows_night_ensemble(results: Path, labels: pd.DataFrame) -> list[dict]:
    manifest = results / "night_20260912" / "candidate_manifest.json"
    if not manifest.exists():
        return []
    data = json.loads(manifest.read_text())
    cand = data["candidate_1"]
    frames, weights = [], []
    for member in cand["members"]:
        csv_path = results / "night_20260912" / "oof" / f"{member['name']}_last_flips.csv"
        if not csv_path.exists():
            return []
        f = pd.read_csv(csv_path)
        f["ID"] = f["ID"].astype(str).str.strip()
        frames.append(f.set_index("ID"))
        weights.append(float(member.get("weight", 1)))
    by_family: dict[str, list] = {}
    for member, f, w in zip(cand["members"], frames, weights):
        by_family.setdefault(member["family"], []).append((f, w))
    combined = None
    total_w = 0.0
    for family, parts in by_family.items():
        oof = pd.concat([f for f, _ in parts])
        w = parts[0][1]
        logs = np.log(oof[list(METRICS)].clip(lower=1e-6)) * w
        combined = logs if combined is None else combined.add(logs, fill_value=np.nan)
        total_w += w
    ens = np.exp(combined / total_w).dropna().reset_index()
    m = _score_frame(ens, labels)
    families = ", ".join(f"{len(p)}x {fam} (w {p[0][1]:g})" for fam, p in by_family.items())
    return [_row(campaign="night_20260912", run="v2ens = candidate_1 (submitted slot 1)", date=_mtime(manifest),
                 model=families, split="oof9000", n=len(ens), CD=m["CD"], CV=m["CV"], HEX=m["HEX"], mean=m["mean"],
                 status="computed", path=_rel(manifest))]


# ---------------------------------------------------------------- campaign-specific files

def rows_round1(results: Path) -> list[dict]:
    root = next(iter(sorted(results.glob("round1_*"))), None)
    if root is None or not (root / "scores.csv").exists():
        return []
    plan = json.loads((root / "round_plan.json").read_text()) if (root / "round_plan.json").exists() else {}
    date = _date(plan.get("started_at"), root / "scores.csv")
    rows = []
    for r in csv.DictReader(open(root / "scores.csv")):
        is_inc = r["candidate"] == "incumbent"
        rows.append(_row(campaign=root.name, run=f"{r['candidate']} (scores.csv)", date=date,
                         model="small CNN " + ("(audited incumbent)" if is_inc else r["candidate"].split("_")[0] + "norm"),
                         split="val892", n=892, CD=float(r["CD"]), CV=float(r["CV"]), HEX=float(r["HEX"]), mean=float(r["mean"]),
                         ci_low="" if is_inc else float(r["delta_ci_low"]), ci_high="" if is_inc else float(r["delta_ci_high"]),
                         reference="" if is_inc else "incumbent (paired delta CI)", status="scored", path=_rel(root / "scores.csv")))
    return rows


def rows_long_training(results: Path) -> list[dict]:
    rows = []
    for root in sorted(results.glob("long_training_*")):
        plan = json.loads((root / "plan.json").read_text()) if (root / "plan.json").exists() else {}
        date = _date(plan.get("created_at"), root)
        model = plan.get("model", "")
        n_val = 0
        for fold_dir in sorted(root.glob("fold*")):
            part = fold_dir / "partition.json"
            n_val += len(json.loads(part.read_text())["val"]) if part.exists() else 0
            scores = fold_dir / "checkpoint_scores.json"
            if scores.exists():
                for epoch, m in json.loads(scores.read_text()).items():
                    rows.append(_row(campaign=root.name, run=f"{fold_dir.name} epoch {epoch} [flips]", date=date, model=model,
                                     seed=plan.get("training_seed", ""), fold=fold_dir.name[4:], epochs=epoch,
                                     split=f"fold-{fold_dir.name[4:]}/{plan.get('n_folds', 5)}", CD=m["CD"], CV=m["CV"],
                                     HEX=m["HEX"], mean=m["mean"], status="scored", path=_rel(scores)))
        comp = root / "comparison.json"
        if comp.exists():
            for epoch, m in json.loads(comp.read_text()).items():
                rows.append(_row(campaign=root.name, run=f"epoch {epoch}, both folds [flips]", date=date, model=model,
                                 seed=plan.get("training_seed", ""), epochs=epoch, split="oof2fold", n=n_val or "",
                                 CD=m["CD"], CV=m["CV"], HEX=m["HEX"], mean=m["mean"], status="computed",
                                 reference="epoch 8 (primary comparison)", path=_rel(comp)))
    return rows


def rows_noise_floor(results: Path) -> list[dict]:
    root = results / "noise_floor_20260912"
    if not root.exists():
        return []
    rows = []
    for name, label in (("B1_compare", "B1 label cleaning: v2fold0 minus top-2% flagged"),
                        ("B3_compare", "B3 capacity: V2-Base 5 folds")):
        path = root / f"{name}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        pm = d["per_metric"]
        rows.append(_row(campaign=root.name, run=label, date=_mtime(path), model=Path(d["a"]).name,
                         split="oof9000" if d["n"] == 9000 else f"fold-0/5", n=d["n"], CD=pm["CD"]["a"], CV=pm["CV"]["a"],
                         HEX=pm["HEX"]["a"], mean=d["a_mean"], ci_low=d["delta_ci95_slide_boot"][0], ci_high=d["delta_ci95_slide_boot"][1],
                         reference=f"{Path(d['b']).name} = {d['b_mean']:.4f} (paired delta {d['delta_a_minus_b']:+.4f}; per-image mean APE, about 0.03 off the metric-level MAPE)",
                         status="paired", path=_rel(path)))
    a1 = root / "A1_counting_floor.json"
    if a1.exists():
        d = json.loads(a1.read_text())
        floors = {m: d[m]["floor_mape"] for m in METRICS}
        cis = [d[m]["ci95"] for m in METRICS]
        rows.append(_row(campaign=root.name, run="A1 counting-statistics floor (cell bootstrap, 25 overlays)", date=_mtime(a1),
                         model="label noise floor", split="floor", n=25, **floors, mean=float(np.mean(list(floors.values()))),
                         ci_low=float(np.mean([c[0] for c in cis])), ci_high=float(np.mean([c[1] for c in cis])),
                         status="floor", path=_rel(a1)))
    a2 = root / "A2_oracle_floor.json"
    if a2.exists():
        d = json.loads(a2.read_text())
        rows.append(_row(campaign=root.name, run="A2 oracle Voronoi readout on the annotator's clicks", date=_mtime(a2),
                         model="reference readout", split="floor", n=25, CD=d["CD_oracle_mape"], HEX=d["HEX_oracle_mape"],
                         ci_low=d["CD_oracle_mape_ci95"][0], ci_high=d["CD_oracle_mape_ci95"][1],
                         reference="CI is for CD; label CV = 2.0x Voronoi CV (corr 0.91)", status="floor", path=_rel(a2)))
    a3 = root / "A3_summary.json"
    if a3.exists():
        d = json.loads(a3.read_text())["scores"]
        names = {"v2": "5x ConvNeXt-V2-Tiny folds", "tiny": "5x ConvNeXt-Tiny folds", "ens": "v2ens (V2 w2 + Tiny w1)"}
        for key, m in d.items():
            rows.append(_row(campaign=root.name, run=f"A3 OOF {key}", date=_mtime(a3), model=names.get(key, key), split="oof9000",
                             n=9000, CD=m["CD"], CV=m["CV"], HEX=m["HEX"], mean=m["mean"], status="scored", path=_rel(a3)))
    return rows


def rows_overnight(results: Path) -> list[dict]:
    root = results / "overnight"
    scores = root / "test_scores.json"
    if not scores.exists():
        return []
    manifest = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {}
    date = _date(manifest.get("started_at"), scores)
    d = json.loads(scores.read_text())
    rows = []
    def add(name, model, m, seed=""):
        rows.append(_row(campaign="overnight", run=name, date=date, model=model, seed=seed, split="test906", n=m.get("n", 906),
                         CD=m["CD"], CV=m["CV"], HEX=m["HEX"], mean=m["mean"], ci_low=m.get("ci_low", ""), ci_high=m.get("ci_high", ""),
                         reference="score CI (iid image bootstrap)", status="one-shot test", path=_rel(scores)))
    if "cellpose_baseline" in d:
        add("cellpose_baseline (test)", f"Cellpose {manifest.get('best_seg_name', 'cyto')} crop {manifest.get('best_crop_frac', '')}", d["cellpose_baseline"])
    if "ridge_calibration" in d:
        add("ridge_calibration (test)", "ridge on Cellpose metrics", d["ridge_calibration"])
    for m in d.get("regression_cnn", []):
        add(f"regression_cnn seed {m['seed']} (test)", "small CNN (pre-audit selection bug)", m, seed=m["seed"])
    comp = root / "config_comparison.csv"
    if comp.exists():
        for r in csv.DictReader(open(comp)):
            rows.append(_row(campaign="overnight", run=f"cellpose sweep {r['name']}", date=date, model=f"Cellpose {r['seg']} crop {r['crop_frac']}",
                             split="val892", n=892, mean=float(r["mean_error_pct"]), status="sweep", path=_rel(comp)))
    return rows


def rows_sam_spike(results: Path) -> list[dict]:
    root = results / "sam_spike_20260912"
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.glob("*.csv")):
        if path.name.startswith("fake"):
            continue
        f = pd.read_csv(path)
        if path.name.startswith("val_"):
            m = {k: float(np.mean(np.abs(f[k] - f[f"{k}_gt"]) / f[f"{k}_gt"].abs() * 100)) for k in METRICS}
            rows.append(_row(campaign=root.name, run=path.stem, date=_mtime(path), model="Cellpose-SAM detector + Voronoi readout",
                             split="val300", n=len(f), **m, mean=float(np.mean(list(m.values()))), status="scored", path=_rel(path)))
        elif {"CD_det", "CD_gt", "split"} <= set(f.columns):
            train = f[f["split"] == "train"]
            det = {k: float(np.mean(np.abs(train[f"{k}_det"] - train[f"{k}_gt"]) / train[f"{k}_gt"].abs() * 100)) for k in METRICS}
            rows.append(_row(campaign=root.name, run=f"{path.stem} (detections, train overlays; F1 {train['f1'].mean():.2f})", date=_mtime(path),
                             model=path.stem.split("_")[0], split="overlay19", n=len(train), **det, mean=float(np.mean(list(det.values()))),
                             reference=f"oracle readout on clicks: CD {np.mean(train['CD_ape_oracle']):.2f}", status="scored", path=_rel(path)))
    return rows


def rows_manual(manual: dict) -> list[dict]:
    return [_row(**{k: v for k, v in r.items() if k in COLUMNS}) for r in manual.get("rows", [])]


def attach_verdicts(rows: list[dict], manual: dict) -> None:
    for row in rows:
        if row["verdict"]:
            continue
        key = f"{row['campaign']}/{row['run']}"
        for v in manual.get("verdicts", []):
            if any(fnmatch.fnmatch(key, pattern) for pattern in v.get("runs", [])):
                row["verdict"] = v["verdict"] + (" [do not rerun]" if v.get("do_not_rerun") else "")
                break


# ---------------------------------------------------------------- rendering

def _fmt(v, nd=4) -> str:
    if v == "" or v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _table(rows: list[dict], cols: list[str]) -> list[str]:
    head = "| " + " | ".join(cols) + " |"
    align = "|" + "|".join("---:" if c in ("n", "CD", "CV", "HEX", "mean", "ci_low", "ci_high", "seed", "fold") else "---" for c in cols) + "|"
    body = ["| " + " | ".join(_fmt(r[c]).replace("|", "\\|") for c in cols) + " |" for r in rows]
    return [head, align, *body]


def _sorted(rows: list[dict]) -> list[dict]:
    def key(r):
        m = r["mean"]
        return (SPLIT_ORDER.index(r["split"]) if r["split"] in SPLIT_ORDER else len(SPLIT_ORDER), m if isinstance(m, float) else 1e9, r["run"])
    return sorted(rows, key=key)


def render_markdown(rows: list[dict], manual: dict, results: Path) -> str:
    campaigns = manual.get("campaigns", {})
    seen = sorted({r["campaign"] for r in rows}, key=lambda c: (campaigns.get(c, {}).get("date", "9999"), c))
    lines = ["# CLEAR-EC experiment ledger", "",
             f"Generated {datetime.now(timezone.utc).date().isoformat()} by `python -m experiments.run ledger` from `{_rel(results)}/` "
             f"and `code/experiments/ledger_manual.yaml`; {len(rows)} rows, machine-readable copy in `docs/experiments.csv`. "
             "Do not edit by hand: change the YAML or the results and regenerate.", "",
             "Score is the equal-weight mean of the CD, CV and HEX MAPEs (percent, lower is better). Splits: `oof9000` = every "
             "labelled image scored by the fold model that did not train on it (the compass); `val892` = the original slide-disjoint "
             "val split (best-epoch numbers there are optimistic); `platform100` = the hidden Phase I test set; `test906` = the "
             "August one-shot test split; `floor` = label-noise floors, not models.", ""]
    lines += ["## Leaderboard", ""]
    for split, title, limit in (("oof9000", "Out-of-fold over all 9,000 labelled images", 20), ("platform100", "Hidden test set (100 images)", 10),
                                ("val892", "Original 892-image val split (top 15)", 15), ("test906", "August one-shot test split", 10)):
        sub = [r for r in _sorted(rows) if r["split"] == split and isinstance(r["mean"], float)][:limit]
        if sub:
            lines += [f"### {title}", "", *_table(sub, ["campaign", "run", "model", "CD", "CV", "HEX", "mean", "verdict"]), ""]
    lines += ["## Levers tested, with verdicts", "", *_table(
        [{"lever": v["lever"], "runs": ", ".join(v.get("runs", [])), "verdict": v["verdict"], "do not rerun": "yes" if v.get("do_not_rerun") else "no"}
         for v in manual.get("verdicts", [])], ["lever", "runs", "verdict", "do not rerun"]), ""]
    lines += ["## Campaigns", ""]
    for c in seen:
        meta = campaigns.get(c, {})
        sub = _sorted([r for r in rows if r["campaign"] == c])
        report = results / c / "REPORT.md"
        lines += [f"### {meta.get('date', '')} {meta.get('title', c)}".strip(), "",
                  f"`{_rel(results / c) if (results / c).exists() else c}`" + (f", report `{_rel(report)}`" if report.exists() else "") + f"; {len(sub)} rows.",
                  meta.get("summary", ""), "",
                  *_table(sub, ["run", "model", "seed", "fold", "epochs", "split", "n", "CD", "CV", "HEX", "mean", "ci_low", "ci_high", "reference", "status"]), ""]
    for c, meta in campaigns.items():
        if c not in seen and (results / c).exists():
            lines += [f"### {meta.get('date', '')} {meta.get('title', c)}", "", f"`{_rel(results / c)}`; no scored rows. {meta.get('summary', '')}", ""]
    deleted = manual.get("deleted", []) or []
    if deleted:
        lines += ["## Removed artifacts", "", *_table(deleted, ["path", "size", "reason", "date"]), ""]
    return "\n".join(lines)


def build_ledger(results: Path, out_dir: Path) -> tuple[Path, Path, int]:
    manual = yaml.safe_load(MANUAL.read_text()) or {}
    labels = _labels() if (REPO / "data" / "final_train_ids.csv").exists() else None
    rows: list[dict] = []
    rows += rows_from_manifests(results)
    if labels is not None:
        rows += rows_from_scores(results, labels)
        rows += rows_night_ensemble(results, labels)
    rows += rows_round1(results)
    rows += rows_long_training(results)
    rows += rows_noise_floor(results)
    rows += rows_overnight(results)
    rows += rows_sam_spike(results)
    rows += rows_manual(manual)
    attach_verdicts(rows, manual)
    rows = sorted(rows, key=lambda r: (r["campaign"], r["split"], r["run"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "experiments.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    md_path = out_dir / "EXPERIMENTS.md"
    md_path.write_text(render_markdown(rows, manual, results) + "\n")
    return csv_path, md_path, len(rows)
