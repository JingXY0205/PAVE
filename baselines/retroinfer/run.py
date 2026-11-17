"""RetroInfer + fixed-per-request NTK (Llama-2-7B Base pilot): download assets and evaluate prepared RULER inputs."""

import argparse
from contextlib import contextmanager
import hashlib
from itertools import islice
import json
import math
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


TOKENIZER_WORKER = '''
import json
import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
print(json.dumps([tokenizer.bos_token_id, tokenizer.eos_token_id]), flush=True)
for line in sys.stdin:
    operation, value = json.loads(line)
    if operation == 'encode':
        answer = tokenizer.encode(value, add_special_tokens=False, truncation=False)
    else:
        answer = tokenizer.decode(value, skip_special_tokens=True)
    print(json.dumps(answer), flush=True)
'''


class Llama3Tokenizer:
    def __init__(self, process):
        self.process = process

    def read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Llama-3 tokenizer process stopped. Check its Transformers environment.')
        return json.loads(line)

    def call(self, operation, value):
        self.process.stdin.write(json.dumps([operation, value]) + '\n')
        self.process.stdin.flush()
        return self.read()

    def __call__(self, prompt, **_kwargs):
        tokens = self.call('encode', prompt)
        if not tokens or tokens[0] != 128000:
            tokens.insert(0, 128000)
        return {'input_ids': tokens}

    def decode(self, tokens, **_kwargs):
        return self.call('decode', tokens)


@contextmanager
def llama3_tokenizer(args, config):
    interpreter = str(args.tokenizer_python or sys.executable)
    process = subprocess.Popen(
        [interpreter, '-u', '-c', TOKENIZER_WORKER, str(args.model_path.resolve())],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf-8',
    )
    try:
        tokenizer = Llama3Tokenizer(process)
        if tokenizer.read() != [config['bos_token_id'], config['eos_token_id']]:
            raise ValueError('Expected the Meta-Llama-3-8B Base tokenizer')
        yield tokenizer
    finally:
        process.stdin.close()
        process.stdout.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


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
    if config.get('source'):
        ensure_source(args.source_dir.resolve(), config, fetch=True)
    if args.model_output is None:
        return
    from huggingface_hub import snapshot_download

    model = config['model']
    patterns = ['*.json', 'tokenizer.model', *model['weight_files']]
    snapshot_download(repo_id=model['repository'], revision=model.get('revision'),
                      local_dir=args.model_output.resolve(), allow_patterns=patterns)
    verify_model(args.model_output, config)


def retrieval_options(tokens, defaults):
    segments = max(round(tokens / 8192), 1)
    multiple = math.lcm(8, segments)
    desired = max(round(tokens / 16), multiple)
    lower = (desired // multiple) * multiple
    centroids = lower if desired - lower <= multiple / 2 else lower + multiple
    return {'RetroInfer': dict(defaults, n_centroids=centroids, n_segment=segments)}


def load_generator(args, config):
    import torch
    import torch.nn.functional as F
    from flash_attn import flash_attn_func
    from model_hub.llama import LlamaModel
    from transformers import AutoConfig, AutoTokenizer
    from transformers.modeling_rope_utils import _compute_yarn_parameters

    rope = config['rope']
    eos = config.get('eos_token_id', 2)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    class HFNumericMixin:
        # Preserve the rounding points from the archived Hugging Face adapter.
        def layernorm(self, hidden_states, epsilon, weight):
            dtype = hidden_states.dtype
            x = hidden_states.float()
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + epsilon)
            return weight * x.to(dtype)

        def wqkv(self, hidden_states, layer):
            qn = self.hidden_size
            kn = self.num_key_value_heads * self.head_dim
            wq, wk, wv = layer.wqkv.split([qn, kn, kn], dim=0)
            return (F.linear(hidden_states, wq), F.linear(hidden_states, wk),
                    F.linear(hidden_states, wv))

        def mlp(self, hidden_states, layer):
            gate, up = layer.gate_up_proj.chunk(2, dim=0)
            return F.linear(F.silu(F.linear(hidden_states, gate)) *
                            F.linear(hidden_states, up), layer.down_proj)

        def apply_rotary_pos_emb(self, query_states, key_states, position_ids):
            b, s, qdim = query_states.shape
            kdim = key_states.shape[-1]
            # Cache stores FP32 half-width cos/sin, already with the fixed request scaling.
            cs = self.cos_sin_cache[position_ids]
            c, sn = cs.chunk(2, dim=-1)
            c = torch.cat((c, c), dim=-1).to(query_states.dtype).unsqueeze(2)
            sn = torch.cat((sn, sn), dim=-1).to(query_states.dtype).unsqueeze(2)
            def rotate(x, heads):
                x = x.reshape(b, s, heads, self.head_dim)
                x1, x2 = x.chunk(2, dim=-1)
                rotated = torch.cat((-x2, x1), dim=-1)
                return x * c + rotated * sn
            q = rotate(query_states, self.num_heads).reshape(b, s, qdim)
            k = rotate(key_states, self.num_key_value_heads).reshape(b, s, kdim)
            return q, k

        def prefill_attention(self, query_states, key_states, value_states, layer_idx):
            assert self.prefill_method == 'full'
            return flash_attn_func(query_states, key_states, value_states,
                                   dropout_p=0.0, causal=True)


    class Llama2Runtime(HFNumericMixin, LlamaModel):
        pass

    numeric_base = LlamaModel if rope['type'] == 'yarn' else Llama2Runtime

    class RetroInferLlama(numeric_base):
        # These frequencies are set from the actual prefill length before each request.
        frozen_inv_freq = None
        frozen_attention_scale = 1.0

        def _set_cos_sin_cache(self):
            if self.frozen_inv_freq is not None:
                self.inv_freq = self.frozen_inv_freq
                self.attention_scaling = self.frozen_attention_scale
            return super()._set_cos_sin_cache()

        def inference(self, inputs_ids, do_sample=False, **_kwargs):
            if do_sample:
                raise ValueError('This profile requires greedy decoding')
            logits = self.prefill_forward(inputs_ids)
            token = logits.argmax(-1)
            result = [token.item()]
            if result[-1] == eos or self.max_new_length == 1:
                return [result]
            self.move()
            for _ in range(self.max_new_length - 1):
                logits = self.decode_forward(token)
                token = logits.argmax(-1)
                result.append(token.item())
                if result[-1] == eos:
                    break
            return [result]

    model = None
    model_length = None

    def generate(input_ids, max_tokens, length):
        nonlocal model, model_length
        if model_length != length:
            model = None
            model = RetroInferLlama(str(args.model_path.resolve()), length, torch.bfloat16,
                                   str(input_ids.device), tokenizer)
            model_length = length
        prefill_length = input_ids.shape[1]
        if rope['type'] == 'yarn':
            native = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
            factor = max(1.0, length / rope['native_context'])
            native.rope_scaling = {
                'rope_type': 'yarn', 'factor': factor, 'beta_fast': rope['beta_fast'],
                'beta_slow': rope['beta_slow'], 'attention_factor': 1.0 + 0.1 * math.log(factor),
            }
            frequencies, model.frozen_attention_scale = _compute_yarn_parameters(native, torch.device('cpu'))
        else:
            head_dim = model.head_dim
            alpha = max(1.0, rope['factor'] * prefill_length / rope['native_context'] - (rope['factor'] - 1.0))
            theta = rope['theta'] * alpha ** (head_dim / (head_dim - 2))
            frequencies = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        model.frozen_inv_freq = frequencies.to(model.inv_freq.device)
        model._set_cos_sin_cache()
        options = retrieval_options(prefill_length, config['attention'])
        generated = model.generate('RetroInfer', input_ids, torch.ones_like(input_ids), max_tokens, options,
                                   do_sample=False, ignore_eos=False, prefill_bsz=1, prefill_method='full')
        return generated[0]

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
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False

    if config.get('source'):
        source_dir = args.source_dir.resolve()
        ensure_source(source_dir, config)
        sys.path.insert(0, str(source_dir))
    verify_model(args.model_path, config)
    if config.get('backbone') == 'llama3':
        with llama3_tokenizer(args, config) as tokenizer:
            return evaluate_examples(args, config, torch, device, tokenizer)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if (tokenizer.bos_token_id, tokenizer.eos_token_id) != (1, 2):
        raise ValueError('Expected Llama-2 BOS=1 and EOS=2')
    return evaluate_examples(args, config, torch, device, tokenizer)


def evaluate_examples(args, config, torch, device, tokenizer):

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
            bos = config.get('bos_token_id', 1)
            if ids[0] != bos or ids.count(bos) != 1:
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
    fetch.add_argument('--source-dir', type=Path)
    fetch.add_argument('--model-output', type=Path, help='Optional local directory for the pinned checkpoint')
    run = commands.add_parser('evaluate', help='Run RULER with a local model and prepared prompts')
    run.epilog = 'Build the CUDA extensions using the upstream source before evaluation.'
    run.add_argument('--source-dir', type=Path, help='Pinned upstream source checkout')
    for command in (fetch, run):
        command.add_argument('--backbone', choices=('llama2', 'llama3'), default='llama2')
    run.add_argument('--tokenizer-python', type=Path,
                     help='Python executable in the Llama-3 tokenizer environment')
    run.add_argument('--model-path', type=Path, required=True)
    run.add_argument('--data-dir', type=Path, required=True, help='Prepared RULER directory for the selected backbone')
    run.add_argument('--output-dir', type=Path, required=True, help='New directory for predictions and scores')
    run.add_argument('--length', '--lengths', dest='length', action='extend', nargs='+', type=int,
                     metavar='TOKENS',
                     help='Specify one or more token budgets for the selected backbone.')
    run.add_argument('--samples', type=int, default=SAMPLES_PER_TASK,
                     help='Exact prompt count per task. Default: %(default)s')
    run.add_argument('--device', default='cuda')
    llama3 = config['llama3']
    run.epilog = (f"Llama-2 lengths: {config['lengths']}. "
                  + (f"Llama-3 lengths: {llama3['lengths']} (Transformers {llama3['transformers_version']})."
                     if llama3['lengths'] else 'A Llama-3 profile is unavailable.'))
    args = parser.parse_args()
    if args.backbone == 'llama3':
        overrides = config['llama3']
        config = dict(config, **overrides, backbone='llama3')
        if not config['lengths']:
            run.error(next(iter(config['unavailable_lengths'].values())))
    config.pop('llama3')
    if config.get('source') and args.source_dir is None:
        parser.error('--source-dir is required for this backbone')
    if args.command == 'download':
        download(args, config)
    else:
        if config.get('separate_tokenizer') and args.tokenizer_python is None:
            run.error('This Llama-3 backend requires --tokenizer-python (a Transformers 4.40.2+ environment).')
        if args.samples < 1:
            run.error('--samples must be positive')
        for length in args.length or ():
            if length not in config['lengths']:
                reason = config.get('unavailable_lengths', {}).get(str(length), 'No configured RULER profile.')
                run.error(f'{length} tokens: {reason} Supported lengths: {config["lengths"]}.')
        evaluate(args, config)


if __name__ == '__main__':
    main()
