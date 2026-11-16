"""YaRN-32K (LCEG, Llama-2-7B): download assets and evaluate prepared RULER inputs."""

import argparse
import hashlib
import inspect
from itertools import islice
import json
from pathlib import Path
import re
from statistics import mean
import subprocess
import sys
import textwrap
from types import MethodType


TASK_CAPS = {
    'niah_single_1': 128,
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
    'qa_2': 32,
}
SAMPLES_PER_TASK = 500
SEED = 42
RULER_REVISION = 'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_model(path, config):
    model = config['model']
    expected = model['expected_config']
    checkpoint = read_json(path / 'config.json')
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Checkpoint configuration does not match {config['name']}")
    for name, expected_hash in model['sha256'].items():
        artifact = path / name
        digest = hashlib.sha256()
        with artifact.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                digest.update(block)
        if digest.hexdigest() != expected_hash:
            raise ValueError(f'SHA-256 mismatch for {artifact}')


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def ensure_source(directory, config, *, fetch=False):
    source = config['source']
    if fetch and not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', source['repository'], str(directory)], check=True)
        subprocess.run(['git', '-C', str(directory), 'checkout', '--detach', source['revision']], check=True)
    revision = subprocess.check_output(
        ['git', '-C', str(directory), 'rev-parse', '--verify', 'HEAD'], text=True,
    ).strip()
    if revision != source['revision']:
        raise ValueError(f"Expected source revision {source['revision']}, got {revision}")


def download(args, config):
    ensure_source(args.source_dir.resolve(), config, fetch=True)
    if args.model_output is None and args.tokenizer_output is None:
        return
    from huggingface_hub import snapshot_download

    if args.model_output is not None:
        model = config['model']
        patterns = ['*.json', 'tokenizer.model', *model['weight_files']]
        snapshot_download(repo_id=model['repository'], revision=model['revision'],
                          local_dir=args.model_output.resolve(), allow_patterns=patterns)
        verify_model(args.model_output, config)
    if args.tokenizer_output is not None:
        tokenizer = config['tokenizer']
        snapshot_download(repo_id=tokenizer['repository'], revision=tokenizer['revision'],
                          local_dir=args.tokenizer_output.resolve(),
                          allow_patterns=['tokenizer*', 'special_tokens_map.json', 'config.json'])
        verify_tokenizer(args.tokenizer_output, config)


def verify_tokenizer(path, config):
    tokenizer_file = path / 'tokenizer.json'
    if hashlib.sha256(tokenizer_file.read_bytes()).hexdigest() != config['tokenizer']['sha256']:
        raise ValueError('Expected the pinned Llama-2 Base tokenizer used by the archived YaRN run')


def rewrite_forward(module, changes, **bindings):
    # Adapt cache ownership in memory. Keep upstream source files intact.
    original = module.forward
    source = textwrap.dedent(inspect.getsource(original))
    source = source[source.index('def forward('):]
    for old, new in changes.items():
        if source.count(old) != 1:
            raise RuntimeError(f'Unexpected forward implementation in {type(module).__name__}')
        source = source.replace(old, new, 1)
    namespace = dict(original.__func__.__globals__, **bindings)
    exec(compile(source, '<baseline-forward-adapter>', 'exec'), namespace)
    module.forward = MethodType(namespace['forward'], module)


def final_position(_module, inputs):
    return (inputs[0][:, -1:, :], *inputs[1:])


class LayerCache(list):
    def set_layer(self, index, values):
        if index == len(self):
            self.append(values)
        else:
            self[index] = values


def preserve_layer_cache(model):
    rewrite_forward(model.model, {
        'next_decoder_cache = () if use_cache else None':
            'next_decoder_cache = (past_key_values if isinstance(past_key_values, LayerCache) else LayerCache()) if use_cache else None',
        'next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)':
            'next_decoder_cache.set_layer(idx, layer_outputs[2 if output_attentions else 1])',
    }, LayerCache=LayerCache)


def greedy_generate(model, input_ids, max_tokens, **kwargs):
    from transformers import GenerationConfig

    generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_tokens,
                                  bos_token_id=1, eos_token_id=2, pad_token_id=0, use_cache=True)
    tokens = model.generate(input_ids=input_ids, attention_mask=input_ids.new_ones(input_ids.shape),
                            generation_config=generation, **kwargs)
    return tokens[0, input_ids.shape[1]:].tolist()


def load_generator(args, config):
    import torch
    from models.llama_yarn.configuration_llama import LlamaConfig
    from models.llama_yarn.modeling_llama_yarn import LlamaForCausalLM

    model_config = LlamaConfig.from_pretrained(args.model_path, local_files_only=True)
    # LCEG dispatches FlashAttention through the Transformers 4.35-era flag.
    model_config._flash_attn_2_enabled = True
    model = LlamaForCausalLM.from_pretrained(
        args.model_path, config=model_config, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', use_safetensors=True,
    ).to(args.device).eval()
    if any(type(layer.self_attn).__name__ != 'LlamaFlashAttention2' for layer in model.model.layers):
        raise RuntimeError('Expected the LCEG FlashAttention-2 backend')
    preserve_layer_cache(model)
    model.lm_head.register_forward_pre_hook(final_position)

    def generate(input_ids, max_tokens, _length):
        for layer in model.model.layers:
            layer.self_attn._init_rope()
            layer.self_attn.rotary_emb.to(input_ids.device)
        return greedy_generate(model, input_ids, max_tokens)

    return generate


def iter_inputs(data, protocol, tokenizer):
    count = protocol['samples_per_task']
    for length in protocol['lengths']:
        for task, cap in TASK_CAPS.items():
            path = data / str(length) / task / 'validation.jsonl'
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
    values = []
    for prediction, answers in zip(predictions, references):
        text = re.sub(r'[\x00-\x1f]', '\n', prediction.strip()).lower()
        matches = sum(answer.lower() in text for answer in answers)
        values.append(bool(matches) if task.startswith('qa_') else matches / len(answers))
    return round(sum(values) / len(values) * 100, 2)


def summarize(records, protocol):
    scores = {}
    for length in protocol['lengths']:
        by_task = {}
        for task in TASK_CAPS:
            rows = [row for row in records if row['length'] == length and row['task'] == task]
            if len(rows) != protocol['samples_per_task']:
                raise ValueError(f'Unexpected number of results for {length}/{task}')
            by_task[task] = score(task, [row['prediction'] for row in rows], [row['references'] for row in rows])
        scores[str(length)] = dict(macro_score=mean(by_task.values()), per_task=by_task)
    return dict(samples=len(records), lengths=scores,
                overall_macro=mean(entry['macro_score'] for entry in scores.values()))


def evaluate(args, config):
    import torch
    import transformers
    from transformers import AutoTokenizer

    if transformers.__version__ != config['transformers_version']:
        raise RuntimeError(f"Install transformers=={config['transformers_version']} for this runner")
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU with BF16 support is required')
    torch.cuda.set_device(device if device.index is not None else torch.cuda.current_device())
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU with BF16 support is required')
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False

    source_dir = args.source_dir.resolve()
    ensure_source(source_dir, config)
    sys.path.insert(0, str(source_dir))
    verify_model(args.model_path, config)
    verify_tokenizer(args.tokenizer_path, config)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    if (tokenizer.bos_token_id, tokenizer.eos_token_id) != (1, 2):
        raise ValueError('Expected Llama-2 BOS=1 and EOS=2')

    protocol = dict(lengths=list(dict.fromkeys(args.length or config['default_lengths'])),
                    samples_per_task=args.samples)
    if args.output_dir.exists():
        raise FileExistsError(f'Choose a new output directory: {args.output_dir}')
    generate = load_generator(args, config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions = args.output_dir / 'predictions.jsonl'
    rows = []
    with predictions.open('x', encoding='utf-8') as stream, torch.inference_mode():
        for sample in iter_inputs(args.data_dir, protocol, tokenizer):
            ids = sample.pop('ids')
            if ids[0] != 1 or ids.count(1) != 1:
                raise ValueError(f"Expected one initial BOS for {sample['key']}")
            token_ids = generate(torch.tensor([ids], device=device), sample['cap'], sample['length'])
            row = dict(sample, input_tokens=len(ids), output_tokens=len(token_ids),
                       prediction=tokenizer.decode(token_ids, skip_special_tokens=True))
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            rows.append(row)

    result = summarize(rows, protocol)
    result.update(config=config, seed=SEED, samples_per_task=args.samples,
                  ruler_revision=RULER_REVISION)
    write_json(args.output_dir / 'summary.json', result)


def main():
    config = read_json(Path(__file__).with_name('config.json'))
    parser = argparse.ArgumentParser(
        description=config['name'],
        epilog=f"Evaluation requires a BF16 CUDA GPU, transformers=={config['transformers_version']}, "
               + ', '.join(config['dependencies']) + '.',
    )
    commands = parser.add_subparsers(dest='command', required=True)
    fetch = commands.add_parser('download', help='Fetch official artifacts to user-selected directories')
    fetch.add_argument('--source-dir', type=Path, required=True)
    fetch.add_argument('--model-output', type=Path, help='Optional local directory for the pinned checkpoint')
    fetch.add_argument('--tokenizer-output', type=Path, help='Optional directory for the shared Base tokenizer')
    run = commands.add_parser('evaluate', help='Run RULER with a local model and prepared prompts')
    run.add_argument('--tokenizer-path', type=Path, required=True, help='Shared local Llama-2 Base tokenizer')
    run.add_argument('--source-dir', type=Path, required=True, help='Pinned upstream source checkout')
    run.add_argument('--model-path', type=Path, required=True)
    run.add_argument('--data-dir', type=Path, required=True, help='Prepared RULER llama2 directory')
    run.add_argument('--output-dir', type=Path, required=True, help='New directory for predictions and scores')
    run.add_argument('--length', '--lengths', dest='length', action='extend', nargs='+', type=int,
                     metavar='TOKENS',
                     help=f"Specify one or more lengths. Supported: {config['lengths']}. Default: {config['default_lengths']}")
    run.add_argument('--samples', type=int, default=SAMPLES_PER_TASK,
                     help='Exact prompt count per task. Default: %(default)s')
    run.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.command == 'download':
        download(args, config)
    else:
        if args.samples < 1:
            run.error('--samples must be positive')
        for length in args.length or ():
            if length not in config['lengths']:
                reason = config.get('unavailable_lengths', {}).get(str(length), 'No configured RULER profile.')
                run.error(f'{length} tokens: {reason} Supported lengths: {config["lengths"]}.')
        evaluate(args, config)


if __name__ == '__main__':
    main()
