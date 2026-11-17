"""Score default PAVE perplexity on prepared PG-19 and ArXiv token windows."""
from collections import defaultdict
from itertools import chain
import json
from pathlib import Path

from pave.inference import load
from .checks import read_json, verify_backbone, verify_sha256, write_json
from .hyper import evaluate_window, ppl_arguments, summarize_windows
from .ruler import FULL_PROTOCOL


DATASETS = ('pg19', 'arxiv')


def binary_windows(args):
    import numpy as np

    path = args.manifest or args.inputs.with_suffix(args.inputs.suffix + '.manifest.json')
    manifest = read_json(path)
    dtypes = {'uint16': np.dtype('<u2'), 'uint32': np.dtype('<u4')}
    dtype = dtypes.get(manifest.get('dtype'))
    if dtype is None or manifest.get('byte_order', 'little') != 'little':
        raise ValueError('Expected a little-endian uint16 or uint32 token-stream manifest')
    size = args.inputs.stat().st_size
    token_count = manifest['tokens_written']
    if size != token_count * dtype.itemsize or token_count < max(args.length):
        raise ValueError('Token file size disagrees with the manifest or is shorter than a requested window')
    expected_hash = manifest.get('sha256')
    if expected_hash:
        verify_sha256(args.inputs, expected_hash)
    if manifest.get('bos_token_id') is not None:
        backbone_bos = read_json(args.model_path / 'config.json').get('bos_token_id')
        if backbone_bos is not None and manifest['bos_token_id'] != backbone_bos:
            raise ValueError('Token-stream manifest and backbone specify different BOS token IDs')

    tokens = np.memmap(args.inputs, mode='r', dtype=dtype)
    for length in args.length:
        for index, offset in enumerate(range(0, token_count - length + 1, args.stride)):
            if args.max_windows is not None and index >= args.max_windows:
                break
            yield dict(dataset=args.dataset, id=str(index), start_token=offset,
                       input_ids=tokens[offset:offset + length].tolist(),
                       target_start=args.target_start or 1), length


def jsonl_windows(path, lengths):
    with path.open(encoding='utf-8') as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            size = len(row['input_ids'])
            if row['dataset'] not in DATASETS:
                raise ValueError(f"Unsupported PPL dataset {row['dataset']}")
            for length in lengths or [size]:
                if not 1 <= row['target_start'] < length <= size:
                    raise ValueError(f"Invalid score range for window {row['id']} at length {length}")
                yield row, length


def main():
    parser = ppl_arguments(__doc__, input_help='PG-19 or ArXiv token stream (.bin) or prepared windows (JSONL)')
    parser.add_argument('--length', type=int, action='append',
                        help='Window size for .bin input or scored prefix length for JSONL. May be repeated.')
    parser.add_argument('--dataset', choices=DATASETS, help='Dataset name for .bin input')
    parser.add_argument('--manifest', type=Path, help='Defaults to <input>.manifest.json for .bin input')
    parser.add_argument('--stride', type=int, help='Window stride in tokens for .bin input')
    parser.add_argument('--target-start', type=int, help='First token to score in each .bin window (default: 1)')
    parser.add_argument('--max-windows', type=int, help='Maximum .bin windows per requested length')
    args = parser.parse_args()
    binary = args.inputs.suffix.lower() == '.bin'
    if binary:
        if args.dataset is None or not args.length or args.stride is None:
            parser.error('.bin input requires --dataset, --length and --stride')
        start = 1 if args.target_start is None else args.target_start
        if not 1 <= start < min(args.length) or args.stride < 1:
            parser.error('Require 1 <= target-start < length and a positive stride')
        if args.max_windows is not None and args.max_windows < 1:
            parser.error('max-windows must be positive')
        windows = binary_windows(args)
    else:
        if any(getattr(args, name) is not None
               for name in ('dataset', 'manifest', 'stride', 'target_start', 'max_windows')):
            parser.error('Binary window options require a .bin input')
        windows = jsonl_windows(args.inputs, args.length)
    first = next(windows, None)
    if first is None:
        parser.error('No PPL windows supplied')

    verify_backbone(args.model_path, FULL_PROTOCOL['model_weight_sha256'])
    verify_sha256(args.checkpoint, FULL_PROTOCOL['checkpoint_sha256'])
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import torch

    torch.manual_seed(42)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, core, _, _ = load(args.model_path, args.checkpoint, decode=False)
    grouped = defaultdict(list)
    with (args.output_dir / 'windows.jsonl').open('w', encoding='utf-8') as output:
        for row, length in chain((first,), windows):
            result = evaluate_window(model, core, row, length)
            grouped[row['dataset'], length].append(result)
            record = dict(dataset=row['dataset'], length=length, id=row['id'], **result)
            if binary:
                record['start_token'] = row['start_token']
            output.write(json.dumps(record) + '\n')
    write_json(args.output_dir / 'summary.json', [
        dict(dataset=dataset, length=length, **summarize_windows(windows))
        for (dataset, length), windows in sorted(grouped.items())
    ])


if __name__ == '__main__':
    main()
