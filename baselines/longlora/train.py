"""Train a local Llama-3 baseline using the PAVE LongAlpaca-12k data convention."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random


METHOD = 'longlora'


def shifted_attention(query, key, value, ratio):
    import torch
    from torch.nn import functional as F

    length = query.shape[-2]
    group_size = max(2, int(length * ratio))
    query_half = query.shape[1] // 2
    kv_half = key.shape[1] // 2
    if query_half * 2 != query.shape[1] or kv_half * 2 != key.shape[1]:
        raise ValueError('Shifted Short Attention needs two evenly divisible head groups')
    groups = []
    for head_group, offset in enumerate((group_size, group_size // 2)):
        q_slice = slice(head_group * query_half, (head_group + 1) * query_half)
        kv_slice = slice(head_group * kv_half, (head_group + 1) * kv_half)
        boundaries = [0, *range(offset, length, group_size), length]
        fragments = []
        for start, end in zip(boundaries, boundaries[1:]):
            # Match the author's GQA expansion within the current shifted group.
            repeat = query_half // kv_half
            group_key = key[:, kv_slice, start:end].repeat_interleave(repeat, dim=1)
            group_value = value[:, kv_slice, start:end].repeat_interleave(repeat, dim=1)
            fragments.append(F.scaled_dot_product_attention(
                query[:, q_slice, start:end], group_key, group_value, is_causal=True, dropout_p=0.0,
            ))
        groups.append(torch.cat(fragments, dim=2))
    return torch.cat(groups, dim=1)


def build_model(model_path, device, settings):
    from types import MethodType
    import torch
    from torch import nn
    from torch.nn import functional as F
    from transformers import AutoConfig, AutoModelForCausalLM

    class LoRALinear(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            rank = settings['rank']
            self.scale = settings['alpha'] / rank
            self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
            self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

        def forward(self, inputs):
            extra = F.linear(F.linear(inputs.float(), self.a), self.b) * self.scale
            return self.base(inputs) + extra.to(inputs.dtype)

    cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
    cfg.rope_scaling = dict(settings['rope'])
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=cfg, use_safetensors=True, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa',
    ).to(device)
    model.requires_grad_(False)
    for layer in model.model.layers:
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            target = getattr(layer.self_attn, name)
            setattr(layer.self_attn, name, LoRALinear(target))
    if METHOD == 'longlora':
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        model.get_input_embeddings().weight.requires_grad_(True)
        for module in model.modules():
            if module.__class__.__name__ == 'LlamaRMSNorm':
                module.weight.requires_grad_(True)

        def shifted_forward(self, hidden_states, position_embeddings, attention_mask=None,
                            past_key_value=None, **kwargs):
            batch, length, _ = hidden_states.shape
            if past_key_value is not None or attention_mask is not None:
                raise ValueError('Shifted training expects one complete, unpadded sequence with no cache')
            shape = (batch, length, -1, self.head_dim)
            q = self.q_proj(hidden_states).view(shape).transpose(1, 2)
            k = self.k_proj(hidden_states).view(shape).transpose(1, 2)
            v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
            output = shifted_attention(q, k, v, settings['shifted_group_ratio'])
            return self.o_proj(output.transpose(1, 2).reshape(batch, length, -1)), None

        for layer in model.model.layers:
            attention = layer.self_attn
            original = attention.forward
            # Keep the author's shifted groups during training and dense attention at inference.
            def dispatch(self, *args, original=original, **kwargs):
                if self.training:
                    return shifted_forward(self, *args, **kwargs)
                return original(*args, **kwargs)
            attention.forward = MethodType(dispatch, attention)
    return model


def ordinary_example(row, device):
    import torch
    ids = torch.tensor([row['input_ids']], dtype=torch.long, device=device)
    start = row['prompt_tokens']
    positions = torch.arange(start - 1, ids.shape[-1] - 1, device=device)
    return ids, positions, ids[:, start:]


def training_model(args, settings):
    model = build_model(args.model_path, args.device, settings)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    def hidden(ids):
        return model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state

    return model, hidden, model.lm_head, ordinary_example


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_split(path, name, settings, manifest):
    dataset = settings['dataset']
    if (manifest.get('dataset_id'), manifest.get('revision'), manifest.get('seed')) != (
            dataset['id'], dataset['revision'], dataset['split_seed']):
        raise ValueError('Expected the pinned LongAlpaca-12k split manifest')
    entry = manifest.get('splits', {}).get(name, {})
    if entry.get('file') != path.name or entry.get('sha256') != file_hash(path):
        raise ValueError(f'LongAlpaca {name} split does not match the manifest: {path}')
    return entry['rows']


def read_manifest(train_path, valid_path):
    train_manifest = train_path.with_name('split_manifest.json')
    valid_manifest = valid_path.with_name('split_manifest.json')
    if train_path.resolve() == valid_path.resolve():
        raise ValueError('Use separate train and validation files')
    if train_manifest.read_bytes() != valid_manifest.read_bytes():
        raise ValueError('Train and validation must use the same LongAlpaca split manifest')
    return json.loads(train_manifest.read_text(encoding='utf-8'))


def normalize(row):
    instruction, context, answer = (
        str(row.get(key) or '').replace('\r\n', '\n').replace('\r', '\n').strip()
        for key in ('instruction', 'input', 'output')
    )
    if not instruction or not answer:
        raise ValueError('Expected a normalized LongAlpaca record with instruction and output')
    if context:
        instruction = f'{instruction}\n\n{context}'
    return instruction, answer


def truncate_prompt(tokens, budget, tail_limit):
    if len(tokens) <= budget:
        return tokens
    tail = min(tail_limit, max(budget // 2, 1))
    head = budget - tail
    return tokens[:head] + tokens[-tail:]


def encode_row(row, encode, settings, *, validation=False):
    dataset = settings['dataset']
    instruction, answer = normalize(row)
    prompt = encode(f'### Instruction:\n{instruction}\n\n### Response:\n')
    if not prompt or prompt[0] != 128000:
        prompt.insert(0, 128000)
    if prompt.count(128000) != 1:
        raise ValueError('LongAlpaca prompt has more than one BOS token')
    answer_ids = encode(answer)[:settings['max_answer_tokens'] - 1] + [128001]
    if 128000 in answer_ids:
        raise ValueError('LongAlpaca answer contains a BOS token')
    length = len(prompt) + len(answer_ids)
    buckets = dataset['length_buckets']
    bucket = next((size for size in buckets if length <= size), buckets[-1])
    if min(length, bucket) < dataset['min_tokens']:
        return None
    tail_limit = min(dataset['prompt_tail_max_tokens'], max(bucket // 4, 1536))
    prompt = truncate_prompt(prompt, bucket - len(answer_ids), tail_limit)
    if validation:
        budget = dataset['validation_max_length'] - len(answer_ids)
        prompt = truncate_prompt(prompt, budget, dataset['prompt_tail_max_tokens'])
    return dict(input_ids=prompt + answer_ids, labels=[-100] * len(prompt) + answer_ids,
                prompt_tokens=len(prompt), bucket=bucket)


def scan_split(path, name, manifest, encode, settings):
    expected = verify_split(path, name, settings, manifest)
    index = []
    count = 0
    with path.open('rb') as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                continue
            count += 1
            row = json.loads(line)
            encoded = encode_row(row, encode, settings, validation=name == 'valid')
            if encoded is not None:
                index.append((offset, encoded['bucket'], hashlib.sha256(line).hexdigest()))
    if count != expected or not index:
        raise ValueError(f'Unexpected LongAlpaca {name} split size or no rows after filtering: {path}')
    return index


def read_example(path, record, encode, settings, *, validation=False):
    offset, bucket, digest = record
    with path.open('rb') as stream:
        stream.seek(offset)
        line = stream.readline()
    if hashlib.sha256(line).hexdigest() != digest:
        raise ValueError(f'LongAlpaca split changed after indexing: {path}')
    row = encode_row(json.loads(line), encode, settings, validation=validation)
    if row is None or row['bucket'] != bucket:
        raise ValueError(f'LongAlpaca tokenizer changed after indexing: {path}')
    return row


def bucket_order(rows, seed, epoch):
    groups = {}
    for index, (_, bucket, _) in enumerate(rows):
        groups.setdefault(bucket, []).append(index)
    rng = random.Random(seed + epoch)
    for bucket in sorted(groups):
        rng.shuffle(groups[bucket])
        yield from groups[bucket]


def save_checkpoint(torch, path, params, contract, step, validation_nll):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(trainable_state={name: value.detach().cpu() for name, value in params.items()},
                    contract=contract, step=step, validation_nll=validation_nll), temporary)
    temporary.replace(path)


def load_trained_state(torch, path, model, model_path, settings, method):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    contract = payload['contract']
    if contract['method'] != method:
        raise ValueError(f'Expected a matched {method} checkpoint')
    if contract['base_config_sha256'] != file_hash(model_path / 'config.json'):
        raise ValueError('Checkpoint was trained with a different base config')
    if contract['base_tokenizer_sha256'] != file_hash(model_path / 'tokenizer.json'):
        raise ValueError('Checkpoint was trained with a different tokenizer')
    for key in settings:
        if key in ('dataset', 'rank', 'alpha', 'rope', 'mem_freq', 'prefill_chunk', 'shifted_group_ratio'):
            if contract['settings'].get(key) != settings[key]:
                raise ValueError(f'Checkpoint training configuration differs for {key}')
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    trained = payload['trainable_state']
    if set(params) != set(trained):
        raise ValueError('Checkpoint parameter names do not match the baseline')
    with torch.no_grad():
        for name, value in params.items():
            if value.shape != trained[name].shape:
                raise ValueError(f'Unexpected checkpoint shape for {name}')
            value.copy_(trained[name])


def run_training(args, cfg, method):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import transformers
    if __package__:
        from . import run as evaluation
    else:
        import run as evaluation

    settings = dict(cfg['training'])
    if args.steps is not None:
        if args.steps <= 0:
            raise ValueError('--steps must be positive')
        settings['steps'] = args.steps
        settings['warmup_steps'] = min(settings['warmup_steps'], args.steps)
        settings['eval_every'] = args.steps
    if transformers.__version__ != settings['transformers_version']:
        raise RuntimeError(f"Use Transformers {settings['transformers_version']} for {method} training")
    if args.output_dir.exists():
        raise FileExistsError(f'Choose a new training output directory: {args.output_dir}')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU with BF16 support is required')
    torch.cuda.set_device(device if device.index is not None else torch.cuda.current_device())
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU with BF16 support is required')
    torch.manual_seed(settings['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    profile = cfg.get('llama3_fine_tuned', cfg['llama3'])
    evaluation.verify_model(args.model_path, profile)
    manifest = read_manifest(args.train_file, args.valid_file)
    if profile.get('separate_tokenizer') and args.tokenizer_python is None:
        raise ValueError('This backend requires --tokenizer-python from a Llama-3 tokenizer environment')
    with evaluation.llama3_tokenizer(args, profile) as tokenizer:
        train_with_tokenizer(args, settings, method, manifest, tokenizer, torch, device)


def train_with_tokenizer(args, settings, method, manifest, tokenizer, torch, device):
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint

    def encode(text):
        return tokenizer.call('encode', text)

    train = scan_split(args.train_file, 'train', manifest, encode, settings)
    valid = scan_split(args.valid_file, 'valid', manifest, encode, settings)
    train_buckets = dict(sorted(Counter(bucket for _, bucket, _ in train).items()))
    valid_buckets = dict(sorted(Counter(bucket for _, bucket, _ in valid).items()))
    model, hidden, project, prepare = training_model(args, settings)
    params = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if not params:
        raise RuntimeError('No trainable baseline parameters were created')
    optimizer = torch.optim.AdamW(params.values(), lr=settings['lr'], weight_decay=settings['weight_decay'])

    def schedule(step):
        warmup = settings['warmup_steps']
        if step < warmup:
            return (step + 1) / warmup
        ratio = (step - warmup) / max(1, settings['steps'] - warmup)
        return .5 * (1 + math.cos(math.pi * min(1, ratio)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    contract = dict(method=method, settings=settings, base_config_sha256=file_hash(args.model_path / 'config.json'),
                    base_tokenizer_sha256=file_hash(args.model_path / 'tokenizer.json'),
                    train_sha256=manifest['splits']['train']['sha256'],
                    validation_sha256=manifest['splits']['valid']['sha256'],
                    train_bucket_counts=train_buckets, validation_bucket_counts=valid_buckets)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'training_config.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')

    def sum_nll(row):
        ids, positions, targets = prepare(row, device)
        outputs = hidden(ids).index_select(1, positions)
        def nll_chunk(states, labels):
            logits = project(states)
            return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                   labels.reshape(-1), reduction='sum')
        total = outputs.new_zeros((), dtype=torch.float32)
        size = settings['logit_chunk']
        for begin in range(0, targets.shape[-1], size):
            end = begin + size
            x, y = outputs[:, begin:end], targets[:, begin:end]
            if torch.is_grad_enabled():
                total = total + checkpoint(nll_chunk, x, y, use_reentrant=False)
            else:
                total = total + nll_chunk(x, y)
        return total, targets.numel()

    best = math.inf
    epoch = 0
    order = iter(bucket_order(train, settings['seed'], epoch))
    for step in range(1, settings['steps'] + 1):
        model.train()
        row_index = next(order, None)
        if row_index is None:
            epoch += 1
            order = iter(bucket_order(train, settings['seed'], epoch))
            row_index = next(order)
        row = read_example(args.train_file, train[row_index], encode, settings)
        optimizer.zero_grad(set_to_none=True)
        numerator, count = sum_nll(row)
        loss = numerator / count
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f'Non-finite training loss at step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params.values(), settings['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        lr_scheduler.step()
        print(json.dumps(dict(step=step, bucket=row['bucket'], answer_nll=float(loss.detach()))), flush=True)

        validation_nll = None
        if step % settings['eval_every'] == 0 or step == settings['steps']:
            model.eval()
            total = tokens = 0
            with torch.inference_mode():
                for record in valid:
                    example = read_example(args.valid_file, record, encode, settings, validation=True)
                    subtotal, count = sum_nll(example)
                    total += subtotal.item()
                    tokens += count
            validation_nll = total / tokens
            if not math.isfinite(validation_nll):
                raise RuntimeError(f'Non-finite validation loss at step {step}')
            if validation_nll < best:
                best = validation_nll
                save_checkpoint(torch, args.output_dir / 'best.pt', params, contract, step, validation_nll)
            print(json.dumps(dict(step=step, validation_nll=validation_nll)), flush=True)
        if step % settings['save_every'] == 0 or step == settings['steps']:
            save_checkpoint(torch, args.output_dir / 'last.pt', params, contract, step, validation_nll)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--train-file', type=Path, required=True, help='Prepared LongAlpaca train.jsonl with split_manifest.json')
    parser.add_argument('--valid-file', type=Path, required=True, help='Prepared LongAlpaca valid.jsonl with split_manifest.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--tokenizer-python', type=Path,
                        help='Python with Llama-3 tokenizer support (required for the Landmark backend)')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8'))
    run_training(args, config, METHOD)


if __name__ == '__main__':
    main()
