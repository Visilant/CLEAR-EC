#!/usr/bin/env python3
"""Export a selected checkpoint for the /opt/ml/model submission mount."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.training.regression_cnn import _load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    # Validate the checkpoint architecture/configuration before producing a bundle.
    _load_checkpoint(args.checkpoint_dir, torch.device('cpu'))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    destination = args.output_dir / 'best_model.pt'
    shutil.copyfile(args.checkpoint_dir / 'best_model.pt', destination)
    metadata = {'method':'regression_cnn',
                'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}
    (args.output_dir / 'submission.json').write_text(json.dumps(metadata, indent=2))
    print(f'Exported bundle -> {args.output_dir}')


if __name__ == '__main__':
    main()
