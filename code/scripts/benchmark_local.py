#!/usr/bin/env python3
"""Bounded training throughput check on train images; never reads test labels."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.cache import open_image_cache
from src.data.splits import load_split
from src.training.common import labels_for_indices, target_stats
from src.training.regression_cnn import MetricRegressionDataset, SmallRegressionCNN, _run_epoch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache_dir', default='../data/cache')
    p.add_argument('--labels_csv', default='../data/final_train_ids.csv')
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--limit', type=int, default=512)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    device = torch.device(f'cuda:{args.gpu}')
    memmap, _ = open_image_cache(args.cache_dir)
    frame = labels_for_indices(args.cache_dir, args.labels_csv,
                              load_split(args.cache_dir, 'train')[:args.limit])
    stats = target_stats(frame)
    rows = []
    for uint8, batch, amp, channels_last, threads in [
        (False, 8, False, False, 16), (False, 8, False, False, 4),
        (True, 8, False, False, 4), (True, 32, False, False, 4),
        (True, 32, True, True, 4), (True, 64, True, True, 4),
    ]:
        torch.set_num_threads(threads)
        torch.manual_seed(42)
        model = SmallRegressionCNN().to(device)
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        ds = MetricRegressionDataset(memmap, frame, stats, uint8_inputs=uint8)
        loader = DataLoader(ds, batch_size=batch, num_workers=4, pin_memory=True,
                            persistent_workers=True)
        times = []
        for repeat in range(3):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            _run_epoch(model, loader, torch.nn.HuberLoss(), device, stats,
                       opt, scaler=scaler, amp=amp)
            torch.cuda.synchronize(device)
            if repeat:
                times.append(time.perf_counter() - start)
        row = dict(uint8=uint8, batch=batch, amp=amp, channels_last=channels_last,
                   threads=threads, seconds=float(np.median(times)),
                   images_per_second=len(frame)/float(np.median(times)),
                   peak_gpu_mb=torch.cuda.max_memory_allocated(device)/2**20)
        rows.append(row)
        print(json.dumps(row), flush=True)
        del loader, model, opt, scaler
    Path(args.output).write_text(json.dumps({'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(device), 'images':len(frame), 'results':rows}, indent=2))


if __name__ == '__main__':
    main()
