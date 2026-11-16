"""Run the official ChunkLlama DCA patch on Llama-2-7B Base."""

import argparse
import hashlib
from itertools import islice
import importlib
import json
from pathlib import Path
import re
from statistics import mean
import subprocess
import sys


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
    expected = dict(model_type='llama', num_hidden_layers=32, max_position_embeddings=4096,
                    rope_scaling=None, vocab_size=model['vocab_size'])
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


def git(directory, *args, **kwargs):
    return subprocess.run(['git', '-C', str(directory), *args], check=True, **kwargs)


def check_source(directory, source):
    revision = git(directory, 'rev-parse', '--verify', 'HEAD', capture_output=True, text=True).stdout.strip()
    if revision != source['revision']:
        raise ValueError(f"Expected source revision {source['revision']}, got {revision} in {directory}")


def download(args, config):
    if 'source' in config:
        directory = args.source_dir.resolve()
        source = config['source']
        if not directory.exists():
            directory.parent.mkdir(parents=True, exist_ok=True)
            git(directory.parent, 'clone', '--no-checkout', source['repository'], str(directory))
            git(directory, 'checkout', '--detach', source['revision'])
        check_source(directory, source)

    if args.model_output is not None:
        from huggingface_hub import snapshot_download

        model = config['model']
        patterns = ['*.json', 'tokenizer.model', *model['weight_files']]
        snapshot_download(repo_id=model['repository'], revision=model.get('revision'),
                          local_dir=args.model_output.resolve(), allow_patterns=patterns)
        verify_model(args.model_output, config)


def import_source(directory, name):
    sys.path.insert(0, str(directory.resolve()))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.pop(0)


def load_model(args, config):
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        args.model_path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation=config['attention_implementation'],
        use_safetensors=config['model']['use_safetensors'],
    ).to(args.device).eval()


def generate_dense(model, input_ids, max_tokens, cache):
    from transformers import GenerationConfig

    generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_tokens,
                                  bos_token_id=1, eos_token_id=2, pad_token_id=0, use_cache=True)
    tokens = model.generate(input_ids=input_ids, attention_mask=input_ids.new_ones(input_ids.shape),
                            generation_config=generation, past_key_values=cache)
    return tokens[0, input_ids.shape[1]:].tolist()


def load_generator(args, config):
    import torch
    from transformers.cache_utils import DynamicCache

    class PartialCPUCache(DynamicCache):
        def __init__(self, first_cpu_layer):
            super().__init__()
            self.first_cpu_layer = first_cpu_layer

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            offload = layer_idx >= self.first_cpu_layer
            if offload and layer_idx < len(self.key_cache):
                self.key_cache[layer_idx] = self.key_cache[layer_idx].to(key_states.device)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].to(value_states.device)
            keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)
            if offload:
                self.key_cache[layer_idx] = keys.to('cpu')
                self.value_cache[layer_idx] = values.to('cpu')
            return keys, values

    source = import_source(args.source_dir, 'chunkllama_attn_replace')
    source.replace_with_chunkllama(**config['attention'])
    model = load_model(args, config)
    layers = model.model.layers
    if any(layer.self_attn.forward.__func__ is not source.forward for layer in layers):
        raise RuntimeError('ChunkLlama patch is missing from a Llama attention layer')

    def final_position(_module, inputs):
        return (inputs[0][:, -1:, :], *inputs[1:])

    def generate(input_ids, max_tokens, length):
        # Reset the extended rotary tables between independent requests.
        native_length = config['attention']['pretraining_length']
        for layer in layers:
            rotary = layer.self_attn.rotary_emb
            rotary._set_cos_sin_cache(native_length, rotary.inv_freq.device, torch.float32)
            rotary.max_seq_len = native_length
        first_cpu_layer = config['cpu_kv_from_layer'].get(str(length))
        cache = DynamicCache() if first_cpu_layer is None else PartialCPUCache(first_cpu_layer)
        # The archived 4K and CPU-KV runs projected only the final position.
        if length == native_length or first_cpu_layer is not None:
            with model.lm_head.register_forward_pre_hook(final_position):
                return generate_dense(model, input_ids, max_tokens, cache)
        return generate_dense(model, input_ids, max_tokens, cache)

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

    verify_model(args.model_path, config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if (tokenizer.bos_token_id, tokenizer.eos_token_id) != (1, 2):
        raise ValueError('Expected Llama-2 BOS=1 and EOS=2')
    if 'source' in config:
        check_source(args.source_dir, config['source'])

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
        epilog=f"Evaluation requires a BF16 CUDA GPU, transformers=={config['transformers_version']}, and FlashAttention-2.",
    )
    commands = parser.add_subparsers(dest='command', required=True)
    fetch = commands.add_parser('download', help='Fetch official artifacts to user-selected directories')
    fetch.add_argument('--model-output', type=Path, required=False)
    run = commands.add_parser('evaluate', help='Run RULER with a local model and prepared prompts')
    run.add_argument('--model-path', type=Path, required=True)
    run.add_argument('--data-dir', type=Path, required=True, help='Prepared RULER llama2 directory')
    run.add_argument('--output-dir', type=Path, required=True, help='New directory for predictions and scores')
    run.add_argument('--length', '--lengths', dest='length', action='extend', nargs='+', type=int,
                     metavar='TOKENS',
                     help=f"Specify one or more lengths. Supported: {config['lengths']}. Default: {config['default_lengths']}")
    run.add_argument('--samples', type=int, default=SAMPLES_PER_TASK,
                     help='Exact prompt count per task. Default: %(default)s')
    run.add_argument('--device', default='cuda')
    fetch.add_argument('--source-dir', type=Path, required=True)
    run.add_argument('--source-dir', type=Path, required=True)

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
