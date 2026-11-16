"""Compare PAVE chunk summaries and Remote selection policies on PG-19."""
from collections import defaultdict
from contextlib import contextmanager
import math
from statistics import mean
from types import MethodType

from .checks import write_json
from .hyper import DEFAULT, configure, evaluate_window, ppl_arguments, setup, summarize_windows

LENGTHS = (8192, 16384, 32768)
POOLING = ('first_slice', 'max_pool', 'mean_pool')


@contextmanager
def pooling(model, mode):
    def forward(module, blocks):
        if mode == 'first_slice':
            summary = blocks[..., 0, :]
        elif mode == 'max_pool':
            summary = blocks.amax(dim=-2)
        else:
            summary = blocks.mean(dim=-2)
        return summary, blocks.new_zeros(blocks.shape[:-1]), blocks.new_zeros(())

    modules = [layer.self_attn.block_representative for layer in model.model.layers]
    previous = [module.forward for module in modules]
    try:
        for module in modules:
            module.forward = MethodType(forward, module)
        yield
    finally:
        for module, original in zip(modules, previous, strict=True):
            module.forward = original


def stride_indices(scores, proposal):
    import torch

    # Sample the midpoint of each equally sized segment of the Remote region.
    slots = torch.arange(proposal.keep, device=scores.device)
    ids = ((2 * slots + 1) * proposal.remote) // (2 * proposal.keep)
    return ids.unsqueeze(0).expand(scores.shape[0], -1)


def schedules(rates, capacity):
    forward = [min(capacity, max(1, math.ceil(capacity * rate))) if capacity else 0
               for rate in rates]
    base, remainder = divmod(sum(forward), len(forward))
    uniform = [base + ((layer + 1) * remainder // len(forward) - layer * remainder // len(forward))
               for layer in range(len(forward))]
    return dict(uniform_semantic=uniform, reverse_layer=forward[::-1])


def trace_window(model, core, row, length, *, reference=None, quotas=None, selector=None, score=True):
    decisions = {}

    def observe(attn, q, k, v, proposal):
        scope = tuple(proposal.query_scope)
        key = (scope, proposal.layer)
        if key in decisions:
            raise RuntimeError('Duplicate PAVE routing refresh scope')
        if reference is not None:
            history, proposal.sink, proposal.remote, proposal.local, _, _ = reference[key]
            if proposal.history_blocks != history:
                raise RuntimeError('PAVE reference and comparison have different history lengths')
            proposal.keep = quotas[scope][proposal.layer]
            proposal.ratio = proposal.keep / proposal.remote if proposal.remote else 0.0
        decisions[key] = (proposal.history_blocks, proposal.sink, proposal.remote,
                          proposal.local, proposal.keep, proposal.ratio)

    result = evaluate_window(model, core, row, length, proposal_hook=observe,
                             remote_selector=selector, score=score)
    return decisions, result


def calibrate(model, core, calibration, length):
    by_layer = [list() for _ in model.model.layers]
    for row in calibration:
        trace, _ = trace_window(model, core, row, length, score=False)
        for (_, layer), entry in trace.items():
            by_layer[layer].append(entry[-1])
    rates = [mean(samples) for samples in by_layer]
    for layer in range(1, len(rates)):
        if rates[layer] > rates[layer - 1] + 1e-10:
            raise ValueError('Expected nonincreasing PAVE calibration rates')
        rates[layer] = min(rates[layer], rates[layer - 1])
    return rates


def compare_selection(model, core, row, length, rates, reference):
    capacities = defaultdict(list)
    for (scope, _), entry in reference.items():
        capacities[scope].append(entry[2])
    by_scope = {scope: schedules(rates, min(values)) for scope, values in capacities.items()}
    result = {}
    update = core.update_keep_rate
    core.update_keep_rate = lambda *args, **kwargs: None
    try:
        # Reuse the full model's region boundaries for both fixed schedules.
        for policy in ('uniform_semantic', 'reverse_layer'):
            quotas = {scope: budget[policy] for scope, budget in by_scope.items()}
            _, result[policy] = trace_window(model, core, row, length, reference=reference, quotas=quotas)
    finally:
        core.update_keep_rate = update
    return result


def main():
    parser = ppl_arguments(__doc__)
    args = parser.parse_args()
    rows, model, core, config = setup(args)
    configure(model, config, *DEFAULT)
    calibration = [row for row in rows if row['dataset'] == 'arxiv' and str(row['id']) in {'0', '2', '4'}]
    pg19 = [row for row in rows if row['dataset'] == 'pg19']
    records, summaries = [], []
    for length in LENGTHS:
        rates = calibrate(model, core, calibration, length)
        write_json(args.output_dir / f'calibration_{length}.json', dict(layer_keep_rates=rates))
        groups = defaultdict(list)
        for row in pg19:
            reference, full = trace_window(model, core, row, length)
            variants = dict(full=full)
            for policy in POOLING:
                with pooling(model, policy):
                    _, variants[policy] = trace_window(model, core, row, length)
            _, variants['fixed_stride'] = trace_window(model, core, row, length, selector=stride_indices)
            variants.update(compare_selection(model, core, row, length, rates, reference))
            for policy, value in variants.items():
                groups[policy].append(value)
            records.append(dict(length=length, window=row['id'], variants=variants))
            write_json(args.output_dir / 'windows.json', records)
        for policy, windows in groups.items():
            summaries.append(dict(length=length, variant=policy,
                                  **summarize_windows(windows)))
        write_json(args.output_dir / 'summary.json', summaries)


if __name__ == '__main__':
    main()
