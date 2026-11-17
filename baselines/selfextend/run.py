"""SelfExtend (Llama-2-7B Base): download assets and evaluate prepared RULER inputs."""

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import importlib
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


def greedy_generate(model, input_ids, max_tokens, config, **kwargs):
    from transformers import GenerationConfig

    generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_tokens,
                                  bos_token_id=config.get('bos_token_id', 1), eos_token_id=config.get('eos_token_id', 2),
                                  pad_token_id=config.get('pad_token_id', 0), use_cache=True)
    tokens = model.generate(input_ids=input_ids, attention_mask=input_ids.new_ones(input_ids.shape),
                            generation_config=generation, **kwargs)
    return tokens[0, input_ids.shape[1]:].tolist()


def load_generator(args, config):
    import torch
    import SelfExtend
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', use_safetensors=True,
    ).to(args.device).eval()
    model.lm_head.register_forward_pre_hook(final_position)
    if config.get('backbone') == 'llama3':
        # Select the author's documented Llama-3 port without changing the checkout.
        SelfExtend.SE.Llama = importlib.import_module('self_extend_patch.Llama_4_40')
    else:
        rewrite_forward(model.model, {
            'if not isinstance(past_key_values, StaticCache):': 'if not isinstance(past_key_values, Cache):',
            'next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache':
                'next_decoder_cache',
        })
    settings = config['attention']
    bound_length = None

    def generate(input_ids, max_tokens, length):
        nonlocal bound_length
        if bound_length != length:
            group_size = settings['group_sizes'][str(length)]
            SelfExtend.apply(model, group_size, settings['neighbor_window'], enable_flash_attention=True,
                             scale_base=settings['scale_base'], flash_attention_impl='flash_attn')
            for layer in model.model.layers:
                bound = layer.self_attn.forward.__func__
                if getattr(getattr(bound, 'func', None), '__name__', '') != 'flash_self_extend_forward':
                    raise RuntimeError('SelfExtend FlashAttention is missing from a model layer')
                if (bound.keywords['group_size_1'], bound.keywords['group_size_2']) != (
                        group_size, settings['neighbor_window']):
                    raise RuntimeError('SelfExtend group or neighbor window differs from the configuration')
            bound_length = length
        for layer in model.model.layers:
            layer.self_attn._init_rope()
            layer.self_attn.rotary_emb.to(input_ids.device)
        return greedy_generate(model, input_ids, max_tokens, config, past_key_values=DynamicCache())

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
