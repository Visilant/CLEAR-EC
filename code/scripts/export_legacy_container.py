#!/usr/bin/env python3
"""Convert a Docker save archive with OCI blobs to a legacy Docker archive.

Preserve the image configuration exactly and verify every uncompressed layer
against its rootfs diff_id. Does not change the tested container filesystem.
"""
import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    partial = args.destination.with_suffix(args.destination.suffix + '.partial')
    with tarfile.open(args.source) as source, partial.open('xb') as destination:
        manifest = json.load(source.extractfile('manifest.json'))
        if len(manifest) != 1:
            raise ValueError('Expected exactly one image')
        entry = manifest[0]
        config_data = source.extractfile(entry['Config']).read()
        config = json.loads(config_data)
        diff_ids = config['rootfs']['diff_ids']
        if len(diff_ids) != len(entry['Layers']):
            raise ValueError('Layer/config count mismatch')
        compressor = subprocess.Popen(['pigz', '-1', '-p', '4'], stdin=subprocess.PIPE, stdout=destination)
        try:
            with tarfile.open(fileobj=compressor.stdin, mode='w|') as output:
                def add_bytes(name, data):
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mode = 0o644
                    output.addfile(info, io.BytesIO(data))

                config_name = hashlib.sha256(config_data).hexdigest() + '.json'
                add_bytes(config_name, config_data)
                layers, written = [], set()
                for i, (layer_name, diff_id) in enumerate(zip(entry['Layers'], diff_ids)):
                    target = diff_id.split(':', 1)[1] + '/layer.tar'
                    layers.append(target)
                    if target in written:
                        continue
                    with source.extractfile(layer_name) as raw, tempfile.TemporaryFile() as unpacked:
                        prefix = raw.read(2)
                        raw.seek(0)
                        stream = gzip.GzipFile(fileobj=raw) if prefix == b'\x1f\x8b' else raw
                        digest = hashlib.sha256()
                        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                            digest.update(chunk)
                            unpacked.write(chunk)
                        if 'sha256:' + digest.hexdigest() != diff_id:
                            raise ValueError(f'Layer digest mismatch: {layer_name}')
                        size = unpacked.tell()
                        unpacked.seek(0)
                        info = tarfile.TarInfo(target)
                        info.size = size
                        info.mode = 0o644
                        output.addfile(info, unpacked)
                    written.add(target)
                    print(f'Verified and exported layer {i+1}/{len(diff_ids)}', flush=True)
                add_bytes('manifest.json', json.dumps([{'Config':config_name,
                          'RepoTags':entry['RepoTags'], 'Layers':layers}]).encode())
        finally:
            compressor.stdin.close()
            code = compressor.wait()
        if code != 0:
            raise RuntimeError(f'pigz failed with exit {code}')
    partial.rename(args.destination)
    print(args.destination, flush=True)


if __name__ == '__main__':
    main()
