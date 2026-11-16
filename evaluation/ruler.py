"""Run the full PAVE model on the Llama-2 RULER benchmark."""
import argparse
from itertools import islice
import json
from pathlib import Path
import random
import re
from statistics import mean
import subprocess
import sys

from . import metrics
from .checks import sha256, verify_backbone, verify_sha256, write_json


FULL_PROTOCOL = {'lengths': [8192, 16384, 32768],
 'tasks': ['niah_single_1',
           'niah_single_2',
           'niah_single_3',
           'niah_multikey_1',
           'niah_multikey_2',
           'niah_multikey_3',
           'niah_multivalue',
           'niah_multiquery',
           'vt',
           'cwe',
           'fwe',
           'qa_1',
           'qa_2'],
 'generation_caps': {'niah_single_1': 128,
                     'niah_single_2': 128,
                     'niah_single_3': 128,
                     'niah_multikey_1': 128,
                     'niah_multikey_2': 128,
                     'niah_multikey_3': 128,
                     'niah_multivalue': 128,
                     'niah_multiquery': 128,
                     'vt': 30,
                     'cwe': 120,
                     'fwe': 50,
                     'qa_1': 32,
                     'qa_2': 32},
 'samples_per_task': 500,
 'variants': ['full'],
 'routing_interval': 512,
 'seed': 42,
 'checkpoint_step': 110,
 'checkpoint_sha256': 'f7e2e56c9b3424c9e51b41f8e632557e3a464f48f04aabb0fcef1e74148e9d24',
 'ruler_commit': 'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a',
 'model_weight_sha256': {'model-00001-of-00003.safetensors': '6f7bf3db7d7a78e921b68973dc39d9a08df6cae71c3273750dcb60fb9192f68a',
                         'model-00002-of-00003.safetensors': '7cee8cf83361acfd75e732be678c7e3d2b1134994c9de3e7919d61003e487fe3',
                         'model-00003-of-00003.safetensors': '7aa9a887731fb9044d3d7c597144272319da5f43fb432ff35410c77cb3d05471'},
 'tokenizer_sha256': 'bf467c9e0f536bda271283c6ef85eb1a943e3196b621c8a912d64953b205df83'}


def iter_inputs(data, protocol, tokenizer):
    count = protocol['samples_per_task']
    for length in protocol['lengths']:
        for task in protocol['tasks']:
            path = data / str(length) / task / 'validation.jsonl'
            cap = protocol['generation_caps'][task]
            with path.open(encoding='utf-8') as stream:
                records = (json.loads(line) for line in stream if line.strip())
                seen = 0
                for index, record in enumerate(islice(records, count)):
                    prompt = record['input'] + record.get('answer_prefix', '')
                    references = record['outputs']
                    if not prompt or not isinstance(references, list) or not references or not all(
                            isinstance(ref, str) and ref for ref in references):
                        raise ValueError(f'Invalid RULER record {index} in {path}')
                    ids = tokenizer(prompt, add_special_tokens=True, truncation=False)['input_ids']
                    if not 1 <= len(ids) <= length - cap:
                        raise ValueError(f'{path} row {index} has {len(ids)} tokens for length {length} and cap {cap}')
                    seen += 1
                    yield dict(key=f'{length}_{task}_{index}', length=length, task=task,
                               ids=ids, references=references, cap=cap)
                if seen != count or next(records, None) is not None:
                    raise ValueError(f'{path} must contain exactly {count} RULER samples')


def score(task, predictions, references):
    metric = metrics.string_match_part if task.startswith('qa_') else metrics.string_match_all
    clean = [re.sub(r'[\x00-\x1f]', '\n', text.strip()) for text in predictions]
    return metric(clean, references)


def summarize(records, protocol):
    scores = {}
    for length in protocol['lengths']:
        by_task = {}
        for task in protocol['tasks']:
            rows = [row for row in records if row['length'] == length and row['task'] == task]
            if len(rows) != protocol['samples_per_task']:
                raise ValueError(f'Unexpected number of results for {length}/{task}')
            by_task[task] = score(task, [row['prediction'] for row in rows], [row['references'] for row in rows])
        scores[str(length)] = dict(macro_score=mean(by_task.values()), per_task=by_task)
    return dict(samples=len(records), lengths=scores,
                overall_macro=mean(entry['macro_score'] for entry in scores.values()))


def worker(args, protocol, install_variant):
    import torch
    from transformers import AutoTokenizer
    from PAVE.inference import generate_tokens, load

    torch.set_num_threads(8)
    torch.manual_seed(protocol['seed'])
    random.seed(protocol['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    output = args.output_dir / args.worker
    output.mkdir()

    def intervention(model, core, config):
        install_variant(model, core, config, args.worker, output)

    model, core, _, reset_decode = load(
        args.model_path, args.checkpoint, routing_interval=protocol['routing_interval'],
        intervention=intervention if install_variant else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    def execute(row):
        generated = generate_tokens(model, core, reset_decode, row['ids'], row['cap'], tokenizer.eos_token_id)
        prediction = tokenizer.decode(generated, skip_special_tokens=True)
        return dict(key=row['key'], length=row['length'], task=row['task'],
                    prediction=prediction, references=row['references'], output_token_ids=generated,
                    finish_reason='eos' if generated[-1] == tokenizer.eos_token_id else 'length',
                    score=score(row['task'], [prediction], [row['references']]))

    with (torch.inference_mode(),
          (args.output_dir / 'prepared_inputs.jsonl').open(encoding='utf-8') as inputs,
          (output / 'quality.jsonl').open('w', encoding='utf-8', buffering=1) as results):
        for row in map(json.loads, inputs):
            results.write(json.dumps(execute(row)) + '\n')


def main(protocols=None, install_variant=None):
    parser = argparse.ArgumentParser(description=__doc__ if protocols is None else 'Run PAVE RULER ablations')
    if protocols is not None:
        parser.add_argument('--length', type=int, choices=tuple(protocols), required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--allow-different-checkpoint', action='store_true', help='Evaluate a newly exported local checkpoint')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    protocol = FULL_PROTOCOL if protocols is None else protocols[args.length]
    if args.worker:
        return worker(args, protocol, install_variant)
    if not args.validate_only and (args.checkpoint is None or args.output_dir is None):
        parser.error('--checkpoint and --output-dir are required for inference')
    if args.checkpoint is not None and not args.allow_different_checkpoint:
        verify_sha256(args.checkpoint, protocol['checkpoint_sha256'])
    verify_backbone(args.model_path, protocol['model_weight_sha256'])
    verify_sha256(args.model_path / 'tokenizer.json', protocol['tokenizer_sha256'])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    inputs = iter_inputs(args.data_dir, protocol, tokenizer)
    if args.validate_only:
        print(f'Validated {sum(1 for _ in inputs)} prompts per variant', flush=True)
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    keys = []
    with (args.output_dir / 'prepared_inputs.jsonl').open('w', encoding='utf-8') as stream:
        for row in inputs:
            keys.append(row['key'])
            stream.write(json.dumps(row) + '\n')
    print(f'Validated {len(keys)} prompts per variant', flush=True)
    summaries, predictions = {}, {}
    module = 'evaluation.ruler' if protocols is None else 'evaluation.ablation'
    for variant in protocol['variants']:
        # Each variant gets its own process because the archived hooks patch module state.
        subprocess.run([sys.executable, '-m', module, *sys.argv[1:], '--worker', variant], check=True)
        with (args.output_dir / variant / 'quality.jsonl').open(encoding='utf-8') as stream:
            results = list(map(json.loads, stream))
        if [row['key'] for row in results] != keys:
            raise ValueError(f'Unmatched predictions in {variant}')
        predictions[variant] = results
        summaries[variant] = summarize(results, protocol)
    full = predictions['full']
    paired = {}
    for variant, results in predictions.items():
        differences = [row['score'] - base['score'] for row, base in zip(results, full)]
        paired[variant] = dict(improved=sum(delta > 0 for delta in differences),
                               worsened=sum(delta < 0 for delta in differences),
                               tied=sum(delta == 0 for delta in differences))
    write_json(args.output_dir / 'summary.json', dict(
        complete=True, variants=summaries, paired_vs_full=paired,
        checkpoint_sha256=sha256(args.checkpoint),
        samples_per_task=protocol['samples_per_task'], routing_interval=protocol['routing_interval'],
    ))


if __name__ == '__main__':
    main()
