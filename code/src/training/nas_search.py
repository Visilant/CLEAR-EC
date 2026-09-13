"""Protocol and worker implementation for bounded successive-halving NAS."""
import hashlib
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
from src.training.common import labels_for_indices, target_stats, score_by_id
from src.training.nas_model import Architecture
from src.training.regression_cnn import (
    RegressionConfig, MetricRegressionDataset, build_regression_model, _run_epoch, predict_indices,
)


def write_json(path, data):
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(data, indent=2))


def grouped_holdback(frame, digests, fraction=0.2, seed=20260912):
    """Union slide and exact-pixel duplicate components before splitting train."""
    ids = frame.idx.astype(int).tolist()
    if len(set(ids)) != len(ids) or set(ids) != set(digests):
        raise ValueError('Duplicate or missing training indices/digests')
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    slides, pixels = {}, {}
    for row in frame.itertuples():
        idx = int(row.idx)
        for table, key in ((slides, str(row.slide_id)), (pixels, digests[idx])):
            if key in table:
                parent[find(idx)] = find(table[key])
            else:
                table[key] = idx
    components = sorted({find(i) for i in ids})
    if len(components) < 2:
        raise ValueError('Need at least two independent training components')
    shuffled = np.random.default_rng(seed).permutation(components)
    n = max(1, min(len(components)-1, round(len(components)*fraction)))
    held = set(shuffled[:n])
    fit = sorted(i for i in ids if find(i) not in held)
    holdback = sorted(i for i in ids if find(i) in held)
    return {'fit': fit, 'search': holdback, 'components': len(components), 'seed': seed}


def make_candidates(count=24, seed=20260912):
    if count < 3:
        raise ValueError('At least three architecture candidates required')
    rng = np.random.default_rng(seed)
    found, candidates = set(), []
    patterns = [(1,1,1,1), (1,1,2,2), (1,2,2,3), (2,2,2,2)]
    while len(candidates) < count:
        family = ('residual','inverted','convnext')[len(candidates) % 3]
        # First three are affordable family controls; remaining samples explore.
        first = len(candidates) < 3
        spec = asdict(Architecture(
            block=family, width=16 if first else int(rng.choice([16,24,32])),
            depths=(1,1,1,1) if first else patterns[int(rng.integers(len(patterns)))],
            kernel=3 if first else int(rng.choice([3,5,7])),
            downsample=4 if first else int(rng.choice([2,4])),
            pooling='mean_std' if first else str(rng.choice(['mean','mean_std','attention','multiscale'])),
            normalization='batch' if first else str(rng.choice(['batch','group']))))
        key = json.dumps(spec, sort_keys=True)
        if key not in found:
            found.add(key)
            candidates.append({'id': f'arch_{len(candidates):02d}', 'architecture': spec})
    return [{'id': 'control', 'architecture': None}] + candidates


def prepare(root, cache_dir, labels_csv, count):
    root = Path(root)
    memmap, index = open_image_cache(cache_dir)
    train = set(load_split(cache_dir, 'train'))
    frame = index[index.idx.isin(train)].copy()
    # Hash only training pixels. Never inspect held-out test images for search.
    digests = {int(i): hashlib.blake2b(memmap[int(i)].tobytes(), digest_size=16).hexdigest()
               for i in frame.idx}
    partition = grouped_holdback(frame, digests)
    partition['outer_train'] = sorted(train)
    partition['cache_index_sha256'] = hashlib.sha256((Path(cache_dir)/'index.csv').read_bytes()).hexdigest()
    partition['outer_splits_sha256'] = hashlib.sha256((Path(cache_dir)/'splits.json').read_bytes()).hexdigest()
    partition['labels_sha256'] = hashlib.sha256(Path(labels_csv).read_bytes()).hexdigest()
    write_json(root/'partition.json', partition)
    write_json(root/'train_pixel_hashes.json', digests)
    candidates = make_candidates(count)
    write_json(root/'candidates.json', candidates)
    return candidates


def verify_partition(partition, cache_dir, labels_csv):
    train = set(load_split(cache_dir, 'train'))
    fit, hold = set(partition['fit']), set(partition['search'])
    if not fit or not hold or fit & hold or fit | hold != train:
        raise ValueError('NAS partition must be disjoint and cover only outer train')
    for name, path in [('cache_index_sha256',Path(cache_dir)/'index.csv'),
                       ('outer_splits_sha256',Path(cache_dir)/'splits.json'),
                       ('labels_sha256',Path(labels_csv))]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != partition[name]:
            raise ValueError(f'Changed NAS input: {path}')


def run_worker(job_path, gpu, target_epoch, deadline):
    job = json.loads(Path(job_path).read_text())
    directory = Path(job['directory'])
    directory.mkdir(parents=True, exist_ok=True)
    cache = Path(job['cache_dir'])
    labels = Path(job['labels_csv'])
    partition = json.loads(Path(job['partition']).read_text())
    verify_partition(partition, cache, labels)
    role = job['role']
    if role not in ('search', 'confirmation', 'final'):
        raise ValueError('Unknown NAS role')
    fit_indices = load_split(cache, 'train') if role == 'final' else partition['fit']
    eval_indices = load_split(cache, 'val') if role == 'final' else partition['search']
    cfg = RegressionConfig(**job['config'])
    if not 0 < target_epoch <= cfg.epochs:
        raise ValueError('Invalid NAS rung budget')
    torch.set_num_threads(cfg.cpu_threads)
    if not torch.cuda.is_available():
        raise RuntimeError('NAS requires the planned CUDA infrastructure')
    device = torch.device(f'cuda:{gpu}')
    torch.cuda.set_device(device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    memmap, _ = open_image_cache(cache)
    train_df = labels_for_indices(cache, labels, fit_indices)
    eval_df = labels_for_indices(cache, labels, eval_indices)
    stats = target_stats(train_df)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(MetricRegressionDataset(memmap, train_df, stats), batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers, pin_memory=True,
                              persistent_workers=cfg.num_workers > 0, generator=generator)
    eval_loader = DataLoader(MetricRegressionDataset(memmap, eval_df, stats), batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers, pin_memory=True,
                             persistent_workers=cfg.num_workers > 0)
    model = build_regression_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = torch.nn.HuberLoss()
    start_epoch, history, best, best_epoch = 0, [], float('inf'), 0
    last = directory/'last_model.pt'
    if last.exists():
        checkpoint = torch.load(last, map_location=device, weights_only=False)
        if checkpoint['config'] != asdict(cfg) or checkpoint['target_stats'] != stats:
            raise ValueError('Resume configuration or training statistics changed')
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch, history = checkpoint['epoch'], checkpoint['history']
        best, best_epoch = checkpoint['best_score'], checkpoint['best_epoch']
        torch.set_rng_state(checkpoint['cpu_rng'].cpu())
        torch.cuda.set_rng_state(checkpoint['cuda_rng'].cpu(), device)
        generator.set_state(checkpoint['loader_rng'].cpu())
    completed_epoch = start_epoch
    for epoch in range(start_epoch + 1, target_epoch + 1):
        if time.time() >= deadline:
            break
        start = time.perf_counter()
        train_loss, train_score = _run_epoch(model, train_loader, criterion, device, stats, optimizer)
        eval_loss, score = _run_epoch(model, eval_loader, criterion, device, stats)
        if not np.isfinite(score['mean']):
            raise FloatingPointError('Nonfinite NAS ranking score')
        row = {'epoch': epoch, 'seconds': time.perf_counter()-start, 'train_loss':train_loss,
               'eval_loss':eval_loss, 'train_mape_mean':train_score['mean'],
               **{f'eval_mape_{k}':v for k,v in score.items()}}
        history.append(row)
        snapshot = {'model_state': model.state_dict(), 'target_stats': stats, 'config':asdict(cfg),
                    'epoch':epoch, 'evaluation_role':role, 'eval_mape':score}
        if score['mean'] < best:
            best, best_epoch = score['mean'], epoch
            with atomic_path(directory/'best_model.pt') as temp:
                torch.save(snapshot, temp)
        # Save optimizer + randomness at every epoch so promotions continue training.
        with atomic_path(last) as temp:
            torch.save({**snapshot, 'optimizer':optimizer.state_dict(), 'history':history,
                        'best_score':best, 'best_epoch':best_epoch, 'cpu_rng':torch.get_rng_state(),
                        'cuda_rng':torch.cuda.get_rng_state(device), 'loader_rng':generator.get_state()},temp)
        with atomic_path(directory/'history.csv') as temp:
            pd.DataFrame(history).to_csv(temp,index=False)
        completed_epoch = epoch
        write_json(directory/'progress.json', {'role':role,'epoch':epoch,'best_score':best,
                   'best_epoch':best_epoch,'last_score':score,'updated_at':time.time()})
        print(f"{job['id']} {role} seed={cfg.seed} epoch={epoch}/{target_epoch} "
              f"MAPE={score['mean']:.4f} best={best:.4f} seconds={row['seconds']:.1f}", flush=True)
    result = {'id':job['id'], 'role':role, 'seed':cfg.seed, 'epoch':completed_epoch,
              'target_epoch':target_epoch, 'complete':completed_epoch >= target_epoch,
              'score':best if np.isfinite(best) else None, 'best_epoch':best_epoch,
              'parameters':sum(p.numel() for p in model.parameters()),
              'train_count':len(train_df), 'evaluation_count':len(eval_df)}
    if role == 'final' and result['complete']:
        checkpoint = torch.load(directory/'best_model.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        pred = predict_indices(model,memmap,eval_df,stats,device,batch_size=cfg.batch_size)
        pred.to_csv(directory/'predictions_val.csv',index=False)
        result['val_mape'] = score_by_id(pred,eval_df,expected_ids=eval_df.ID)
    write_json(directory/f'result_{target_epoch}.json',result)
    return result
