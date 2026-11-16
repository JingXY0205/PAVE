"""Run the PAVE salience temperature and sparsity decay sweeps."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from statistics import mean

from .checks import verify_backbone, verify_sha256, write_json
from .full import FULL_PROTOCOL
from .inference import load

INPUT_SHA256 = '55d9584c88057161e635fb4f77b999bf865cbc3dee2f1121555df222edfe7391'
DEFAULT = (0.5, 3.0)
SWEEPS = {
    'main': ((0.25, 3.0), DEFAULT, (0.75, 3.0), (1.0, 3.0),
             (0.5, 1.0), (0.5, 2.0), (0.5, 5.0)),
    'bounds': (DEFAULT, (0.5, 0.0), (0.5, 10.0), (0.5, 30.0), (0.05, 3.0), (2.0, 3.0)),
}


def ppl_arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True, help='Sealed PG-19 and ArXiv token windows (JSONL)')
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser


def setup(args):
    import torch

    verify_sha256(args.inputs, INPUT_SHA256)
    verify_sha256(args.checkpoint, FULL_PROTOCOL['checkpoint_sha256'])
    verify_backbone(args.model_path, FULL_PROTOCOL['model_weight_sha256'])
    with args.inputs.open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    expected = {(dataset, str(index)) for dataset in ('pg19', 'arxiv') for index in range(10)}
    if (len(rows) != len(expected)
            or {(row['dataset'], str(row['id'])) for row in rows} != expected
            or any(len(row['input_ids']) != 32768 or row['target_start'] != 1 for row in rows)):
        raise ValueError('Expected ten sealed 32K windows for each dataset')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, core, config, _ = load(args.model_path, args.checkpoint, decode=False)
    return rows, model, core, config


def configure(model, config, temperature, decay):
    config.salience_temperature = temperature
    config.sparsity_decay_strength = decay
    config.validate()
    for layer in model.model.layers:
        layer.self_attn.block_representative.set_temperature(temperature)


def evaluate_window(model, core, row, length, *, proposal_hook=None, remote_selector=None, score=True):
    import torch
    from torch.nn import functional

    ids = torch.tensor([row['input_ids'][:length]], device='cuda')
    core.reset_rotary_cache(model)
    loss = 0.0
    with torch.inference_mode(), core.execution_context(
            'sparse', proposal_hook=proposal_hook, remote_selector=remote_selector):
        hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
        if score:
            # Score targets with the preceding hidden state and bound LM-head memory.
            for start in range(row['target_start'], length, 128):
                stop = min(start + 128, length)
                logits = model.lm_head(hidden[:, start - 1:stop - 1]).float()
                loss += float(functional.cross_entropy(logits.flatten(0, 1), ids[:, start:stop].flatten(),
                                                        reduction='sum'))
    if not math.isfinite(loss):
        raise ValueError(f"Non-finite PAVE loss for window {row['id']}")
    return dict(nll_sum=loss, target_count=length - row['target_start'] if score else 0)


def summarize_windows(windows):
    totals = {key: sum(row[key] for row in windows) for key in ('nll_sum', 'target_count')}
    return dict(windows=len(windows),
                ppl_mean_window=math.exp(mean(row['nll_sum'] / row['target_count'] for row in windows)),
                ppl_token_weighted=math.exp(totals['nll_sum'] / totals['target_count']))


def main():
    parser = ppl_arguments(__doc__)
    parser.add_argument('--sweep', choices=tuple(SWEEPS), default='main')
    args = parser.parse_args()
    rows, model, core, config = setup(args)
    results = []
    summaries = []
    for temperature, decay in SWEEPS[args.sweep]:
        configure(model, config, temperature, decay)
        groups = defaultdict(list)
        for row in rows:
            result = evaluate_window(model, core, row, 32768)
            result.update(dataset=row['dataset'], id=row['id'], temperature=temperature, decay=decay)
            results.append(result)
            groups[row['dataset']].append(result)
            write_json(args.output_dir / 'windows.json', results)
        for dataset, windows in groups.items():
            summaries.append(dict(dataset=dataset, temperature=temperature, decay=decay,
                                  **summarize_windows(windows)))
        write_json(args.output_dir / 'summary.json', summaries)


if __name__ == '__main__':
    main()
