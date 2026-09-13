#!/usr/bin/env python3
"""Score the fixed normalization round on validation, with paired cluster CIs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.splits import load_split
from experiments.stats import paired_cluster_ci
from src.training.common import METRICS, labels_for_indices, score_by_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('round_dir', type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = args.round_dir.resolve()
    cache = repo/'data/cache'
    gt = labels_for_indices(cache, repo/'data/final_train_ids.csv', load_split(cache, 'val'))
    gt = gt.set_index('ID').sort_index()
    groups = pd.read_csv(cache/'index.csv').set_index('ID').loc[gt.index, 'slide_id'].to_numpy()
    targets = gt[list(METRICS)].to_numpy(float)

    def read_predictions(path):
        frame = pd.read_csv(path)
        score_by_id(frame, gt.reset_index(), expected_ids=gt.index)
        return frame.set_index('ID').loc[gt.index, list(METRICS)].to_numpy(float)

    incumbent = read_predictions(repo/'results/audit_20260911/selection_fix/regression_cnn/seed_42/predictions_val.csv')
    predictions = {'incumbent': incumbent}
    artifacts = {}
    for norm in ('batch', 'group'):
        for seed in (42, 123, 456):
            directory = root/norm/'regression_cnn'/f'seed_{seed}'
            name = f'{norm}_{seed}'
            predictions[name] = read_predictions(directory/'predictions_val.csv')
            artifacts[name] = {'sha256': hashlib.sha256((directory/'best_model.pt').read_bytes()).hexdigest(),
                               'epoch_seconds': float(pd.read_csv(directory/'history.csv').seconds.sum())}
        predictions[f'{norm}_ensemble'] = np.mean([predictions[f'{norm}_{s}'] for s in (42,123,456)], axis=0)

    rows = []
    for name, values in predictions.items():
        frame = pd.DataFrame(values, columns=METRICS, index=gt.index).reset_index()
        scores = score_by_id(frame, gt.reset_index(), expected_ids=gt.index)
        low, high = paired_cluster_ci(values, incumbent, targets, groups)
        rows.append({'candidate': name, **scores, 'delta_ci_low': low, 'delta_ci_high': high})
        if name.endswith('ensemble'):
            frame.to_csv(root/f'{name}_predictions_val.csv', index=False)
    comparisons = {}
    for seed in (42,123,456):
        comparisons[str(seed)] = paired_cluster_ci(predictions[f'group_{seed}'], predictions[f'batch_{seed}'],
                                                    targets, groups)
    pd.DataFrame(rows).to_csv(root/'scores.csv', index=False)
    (root/'analysis.json').write_text(json.dumps({'scores': rows, 'artifacts': artifacts,
        'group_minus_batch_paired_seed_ci': comparisons,
        'valid_counts': dict(zip(METRICS, (np.abs(targets)>1e-8).sum(axis=0).tolist())),
        'bootstrap_clusters': int(len(np.unique(groups))), 'bootstrap_resamples': 2000,
        'test_evaluated': False}, indent=2))
    lines = ['# First-round validation results', '',
             'Six fixed runs: BatchNorm / GroupNorm × seeds 42, 123, 456. Up to ten epochs, patience five; best validation checkpoint. No test evaluation.', '',
             '| Candidate | CD % | CV % | HEX % | Mean % | 95% CI of change vs incumbent (points) |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['candidate']} | {row['CD']:.3f} | {row['CV']:.3f} | {row['HEX']:.3f} | {row['mean']:.3f} | [{row['delta_ci_low']:.3f}, {row['delta_ci_high']:.3f}] |")
    lines += ['', 'Negative changes favor the candidate. Intervals use paired slide-cluster resampling, retain image weighting, and mask zeros separately per metric. Validation checkpoint/model selection makes these descriptive intervals optimistic; they do not include training-seed uncertainty.', '',
              'Ensembles are equal arithmetic means; no blend weights or calibrators were fitted on validation. The original submission bundle remains unchanged.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
