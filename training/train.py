"""Train PAVE's Router and chunk representatives on prepared LongAlpaca-12k."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from functools import partial
import hashlib
from itertools import islice, zip_longest
import json
import math
from pathlib import Path
import random
import subprocess
from typing import TYPE_CHECKING

from dataset.longalpaca.prepare import DATASET_ID, MANIFEST, REVISION, SEED, SPLIT_ROWS


if TYPE_CHECKING:
    import torch
    from pave import model as core


TOKENIZER_WORKER = """
import json
import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
print(json.dumps([tokenizer.bos_token_id, tokenizer.eos_token_id]), flush=True)
for line in sys.stdin:
    texts = json.loads(line)
    print(json.dumps(tokenizer(texts, add_special_tokens=False)["input_ids"]), flush=True)
"""


@contextmanager
def tokenizer_encoder(args, profile):
    expected = [profile["bos_token_id"], profile["eos_token_id"]]
    if not profile["separate_tokenizer"]:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, local_files_only=True)
        if [tokenizer.bos_token_id, tokenizer.eos_token_id] != expected:
            raise ValueError("The local tokenizer does not match the backbone configuration")
        yield lambda texts: tokenizer(texts, add_special_tokens=False)["input_ids"]
        return

    command = [str(args.tokenizer_python.resolve()), "-u", "-c", TOKENIZER_WORKER, str(args.model_path.resolve())]
    with subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
    ) as process:
        def receive():
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("Llama-3 tokenizer stopped. Check --tokenizer-python and the local model directory.")
            return json.loads(line)

        def encode(texts):
            process.stdin.write(json.dumps(texts) + "\n")
            process.stdin.flush()
            return receive()

        try:
            if receive() != expected:
                raise ValueError("Expected the Meta-Llama-3-8B Base tokenizer")
            yield encode
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def truncate_prompt(tokens, budget, tail_limit):
    if len(tokens) <= budget:
        return tokens
    tail_length = min(tail_limit, budget // 2)
    return tokens[:budget - tail_length] + tokens[len(tokens) - tail_length:]


def encode_row(row, encode, profile):
    data = profile["data"]
    instruction, context, answer = (row[key] for key in ("instruction", "input", "output"))
    if not instruction or not answer:
        raise ValueError("Expected normalized LongAlpaca instructions and answers")
    if context:
        instruction = f"{instruction}\n\n{context}"
    prompt, answer_tokens = encode([f"### Instruction:\n{instruction}\n\n### Response:\n", answer])
    bos, eos = profile["bos_token_id"], profile["eos_token_id"]
    if not prompt or prompt[0] != bos:
        prompt.insert(0, bos)
    answer_tokens = answer_tokens[:data["max_answer_tokens"] - 1] + [eos]
    if prompt.count(bos) != 1 or bos in answer_tokens:
        raise ValueError("Expected a single BOS at the start of the prompt")
    total_length = len(prompt) + len(answer_tokens)
    if total_length < data["min_tokens"]:
        return None
    buckets = data["length_buckets"]
    bucket = next((limit for limit in buckets if total_length <= limit), buckets[-1])
    tail_limit = min(data["prompt_tail_max_tokens"], max(bucket // 4, 1536))
    prompt = truncate_prompt(prompt, bucket - len(answer_tokens), tail_limit)
    return prompt + answer_tokens, len(prompt), bucket


@dataclass(frozen=True)
class Row:
    offset: int
    bucket: int
    length: int
    sha256: str


def index_split(path, name, manifest, encode, profile):
    entry = manifest["splits"][name]
    if entry["file"] != path.name or entry["rows"] != SPLIT_ROWS[name]:
        raise ValueError(f"Unexpected LongAlpaca manifest entry for {name}")
    digest, rows, count = hashlib.sha256(), [], 0
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            digest.update(line)
            count += 1
            item = encode_row(json.loads(line), encode, profile)
            if item is not None:
                ids, _, bucket = item
                rows.append(Row(offset, bucket, len(ids), hashlib.sha256(line).hexdigest()))
    if count != entry["rows"] or digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"LongAlpaca split does not match its manifest: {path}")
    return rows


def read_example(stream, row, encode, profile):
    stream.seek(row.offset)
    line = stream.readline()
    if hashlib.sha256(line).hexdigest() != row.sha256:
        raise ValueError(f"LongAlpaca split changed after indexing: {stream.name}")
    return encode_row(json.loads(line), encode, profile)[:2]


def bucket_order(rows, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row.bucket].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    # Interleave buckets so early updates cover multiple lengths.
    for group in zip_longest(*(groups[bucket] for bucket in sorted(groups))):
        yield from (row for row in group if row is not None)


def training_order(rows, seed):
    epoch = 0
    while True:
        yield from bucket_order(rows, seed + epoch)
        epoch += 1


def split_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("dataset_id"), manifest.get("revision"), manifest.get("seed")) != (DATASET_ID, REVISION, SEED):
        raise ValueError("Use the pinned LongAlpaca-12k split manifest")
    if set(manifest["splits"]) != set(SPLIT_ROWS):
        raise ValueError("Expected train, valid, and test entries in the split manifest")
    return manifest


def load_model(args, profile):
    import torch
    import transformers
    from pave import model as core

    if transformers.__version__ != profile["transformers_version"]:
        raise RuntimeError(f"Use Transformers {profile['transformers_version']} for PAVE training")
    dca = profile["dca"]
    config = transformers.LlamaConfig.from_pretrained(args.model_path, local_files_only=True)
    mismatches = {
        key: (getattr(config, key, None), value)
        for key, value in profile["backbone"].items() if getattr(config, key, None) != value
    }
    if mismatches:
        raise ValueError(f"Local model configuration does not match {profile['model_id']}: {mismatches}")
    core.replace_with_chunkllama(**dca)
    model = transformers.LlamaForCausalLM.from_pretrained(
        args.model_path, config=config, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    ).to(args.device)
    model.requires_grad_(False)
    pave = core.PAVEConfig(**profile["pave"])
    router = core.attach_pave(model, pave)
    router_parameters = list(router.parameters())
    representative_parameters = [
        parameter
        for layer in model.model.layers
        for parameter in layer.self_attn.block_representative.parameters()
    ]
    actual = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    expected = {id(parameter) for parameter in router_parameters + representative_parameters}
    if actual != expected or not router_parameters or not representative_parameters:
        raise RuntimeError("Only the Router and chunk representatives may be trained")
    return model, pave, router_parameters, representative_parameters


def save_checkpoint(path, model, pave, profile, step, validation_ce):
    import torch
    from pave import model as core

    temporary = path.with_suffix(".tmp")
    payload = {
        "pave_config": asdict(pave),
        "trainable_state": core.trainable_state_dict(model),
        "backbone": profile["model_id"],
        "dca": profile["dca"],
        "step": step,
        "validation_ce": validation_ce,
    }
    torch.save(payload, temporary)
    temporary.replace(path)


@dataclass
class LayerSalience:
    seen: set = field(default_factory=set)
    sums: list = field(default_factory=list)
    rows: int = 0

    def add(self, scores, temperature, group, anchor):
        import torch
        from torch.nn import functional as F

        chunks = [(group, anchor + index) for index in range(scores.size(-2))]
        new = [index for index, key in enumerate(chunks) if key not in self.seen]
        if not new:
            return
        self.seen.update(chunks)
        logits = scores.float()
        if len(new) != len(chunks):
            logits = logits.index_select(-2, torch.tensor(new, device=scores.device))
        raw = F.log_softmax(logits, dim=-1)
        sharp = F.log_softmax(logits.detach() / temperature, dim=-1)
        self.sums.append((sharp.exp() * (sharp - raw)).sum())
        self.rows += logits.numel() // logits.size(-1)

    def mean(self, hidden):
        import torch

        if not self.rows:
            return hidden.new_zeros((), dtype=torch.float32)
        return torch.stack(self.sums).sum() / self.rows


@dataclass
class Selection:
    proposal: core.Proposal
    group: int
    scores: torch.Tensor | None = None
    top: torch.Tensor | None = None


class TrainingPrefill:
    def __init__(self, router_gradient_scale):
        self.router_gradient_scale = router_gradient_scale
        self.layer_salience = ContextVar("pave_layer_salience")
        self.selection = ContextVar("pave_training_selection")
        self.layer_losses = []

    @contextmanager
    def install(self, model):
        from pave import model as core

        def representative_forward(module, blocks):
            return self.representatives(module, blocks)

        with ExitStack() as stack:
            def patch(owner, name, replacement):
                original = getattr(owner, name)
                stack.callback(setattr, owner, name, original)
                setattr(owner, name, replacement)

            patch(
                core, "sparse_select_from_proposal",
                partial(self.with_selection, core.sparse_select_from_proposal),
            )
            patch(core, "prepare_selected_kv", partial(self.gated_values, core.prepare_selected_kv))
            patch(core.LearnableBlockRepresentative, "forward", representative_forward)
            for index, layer in enumerate(model.model.layers):
                patch(layer, "forward", partial(self.layer_forward, index, layer.forward))
            yield

    def with_selection(self, select, attn, q, k, v, proposal, state, selection_query=None):
        token = self.selection.set(Selection(proposal, state.current_group))
        try:
            return select(attn, q, k, v, proposal, state, selection_query=selection_query)
        finally:
            self.selection.reset(token)

    def remote_topk(self, scores, proposal):
        selection = self.selection.get()
        selection.scores = scores
        selection.top = scores.topk(proposal.keep, dim=-1).indices
        return selection.top

    def representatives(self, module, blocks):
        import torch
        from torch.nn import functional as F
        from pave import model as core

        # Use the same projections and temperature-scaled pool as PAVE inference.
        dtype = blocks.dtype
        normed = F.layer_norm(
            blocks, (blocks.size(-1),),
            module.norm.weight.to(dtype=dtype), module.norm.bias.to(dtype=dtype),
        )
        scores = F.linear(normed, module.salience.weight.to(dtype=dtype)).squeeze(-1)
        selection = self.selection.get()
        temperature = max(module.temperature, core.EPS)
        self.layer_salience.get().add(scores, temperature, selection.group, selection.proposal.sink)
        weights_fp32 = torch.softmax(scores.float() / temperature, dim=-1)
        weights = weights_fp32.to(dtype=dtype)
        representative = torch.sum(weights.unsqueeze(-1) * blocks, dim=-2)
        # The entropy is returned for the public representative cache.
        entropy = -(weights_fp32 * weights_fp32.clamp_min(core.EPS).log()).sum(-1).mean()
        return representative, weights, entropy

    def gated_values(self, prepare, attn, k, v, kb, vb, ids, proposal, state):
        import torch
        from pave import model as core

        sk, sv = prepare(attn, k, v, kb, vb, ids, proposal, state)
        if not torch.is_grad_enabled():
            return sk, sv

        # The public hard region membership and Top-K still decide which KV is read.
        # Both surrogates multiply selected values by exactly one in the forward pass.
        router = attn._shared_memory_router_ref
        soft = router.continuous_state_probabilities(
            proposal.ratios, proposal.history_blocks, attn._pave_config.boundary_temperature,
        )
        regions = (ids >= proposal.sink).long() + (ids >= proposal.sink + proposal.remote).long()
        membership = soft[ids, regions]
        boundary_gate = 1.0 + self.router_gradient_scale * (membership - membership.detach())
        sv = sv * self.token_gate(boundary_gate, sv.dtype)

        if proposal.remote:
            selection = self.selection.get()
            probability = selection.scores.float().softmax(-1).gather(1, selection.top)
            normalized = probability / probability.mean(-1, keepdim=True).detach().clamp_min(core.EPS)
            score_gate = 1.0 + (normalized - normalized.detach())
            all_gates = torch.ones_like(soft[:, 0]).unsqueeze(0)
            all_gates = all_gates.scatter(1, selection.top + proposal.sink, score_gate)
            sv = sv * self.token_gate(all_gates.gather(1, ids), sv.dtype)
        return sk, sv

    @staticmethod
    def token_gate(gate, dtype):
        from pave import model as core

        return gate.repeat_interleave(core.BLOCK_SIZE, dim=-1)[:, None, :, None].to(dtype=dtype)

    def layer_forward(self, index, forward, hidden_states, *args, **kwargs):
        import torch
        from torch.utils.checkpoint import checkpoint
        from pave import model as core

        parent = core._STATE.get()
        incoming_rates = parent.keep_rates.copy()

        def compute(hidden):
            # Each checkpoint replay gets its own cache, retention state, and KL accumulator.
            salience = LayerSalience()
            token = self.layer_salience.set(salience)
            try:
                with core.execution_context(remote_selector=self.remote_topk) as state:
                    state.keep_rates.update(incoming_rates)
                    outputs = forward(hidden, *args, **kwargs)
                    next_rates = {key: value for key, value in state.keep_rates.items() if key[0] == index + 1}
                return outputs, salience.mean(hidden), next_rates
            finally:
                self.layer_salience.reset(token)

        if torch.is_grad_enabled():
            outputs, sharp, next_rates = checkpoint(compute, hidden_states, use_reentrant=False)
        else:
            outputs, sharp, next_rates = compute(hidden_states)
        self.layer_losses.append(sharp)
        parent.keep_rates = next_rates
        return outputs

    def losses(self, model, input_ids, prompt_length, logit_chunk):
        import torch
        from torch.nn import functional as F
        from torch.utils.checkpoint import checkpoint

        self.layer_losses.clear()
        try:
            hidden = model.model(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state
            if len(self.layer_losses) != len(model.model.layers):
                raise RuntimeError("Expected one KL contribution per layer")
            sharp = torch.stack(self.layer_losses).mean()
        finally:
            self.layer_losses.clear()

        # Shift by one to include the first answer token; the prompt has no labels.
        answer = input_ids[0, prompt_length:]
        predictors = hidden[0, prompt_length - 1:-1]

        def chunk_loss(activations, labels):
            return F.cross_entropy(model.lm_head(activations).float(), labels, reduction="sum")

        task = hidden.new_zeros((), dtype=torch.float32)
        for activations, labels in zip(predictors.split(logit_chunk), answer.split(logit_chunk), strict=True):
            if torch.is_grad_enabled() and activations.requires_grad:
                task = task + checkpoint(chunk_loss, activations, labels, use_reentrant=False)
            else:
                task = task + chunk_loss(activations, labels)
        return task / answer.numel(), sharp

def run_training(args, profile, encode):
    import torch

    settings = profile["training"]
    data = profile["data"]
    manifest = split_manifest(args.data_dir / MANIFEST)
    paths = {name: args.data_dir / manifest["splits"][name]["file"] for name in ("train", "valid")}
    train = index_split(paths["train"], "train", manifest, encode, profile)
    valid = index_split(paths["valid"], "valid", manifest, encode, profile)
    dca = profile["dca"]
    effective_chunk = dca["pretraining_length"] * 3 // 4 - dca["local_window_size"]
    # The last input token cannot predict any of the reference answer tokens.
    eligible = [row for row in train if row.length > 2 * effective_chunk + 1]
    if not eligible or not valid:
        raise ValueError("Need training rows with a causal inter branch and an independent validation split")
    validation = list(islice(bucket_order(valid, settings["seed"]), data["validation_examples"]))
    recipe = {
        "profile": profile,
        "split_manifest": manifest,
        "train_post_filter_buckets": dict(Counter(row.bucket for row in train)),
        "train_update_buckets": dict(Counter(row.bucket for row in eligible)),
        "valid_buckets": dict(Counter(row.bucket for row in validation)),
        "train_rows_without_inter_for_answer": len(train) - len(eligible),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in recipe.items() if key not in ("profile", "split_manifest")}), flush=True)

    torch.manual_seed(settings["seed"])
    model, pave, router, representatives = load_model(args, profile)
    optimizer = torch.optim.AdamW(
        [{"params": router, "lr": settings["router_lr"]},
         {"params": representatives, "lr": settings["representative_lr"]}],
        weight_decay=settings["weight_decay"],
    )
    steps, warmup = settings["steps"], settings["warmup_steps"]

    def learning_rate_scale(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    runtime = TrainingPrefill(settings["router_st_gradient_scale"])

    def losses(stream, row):
        ids, prompt_length = read_example(stream, row, encode, profile)
        input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
        return runtime.losses(model, input_ids, prompt_length, settings["logit_chunk"])

    best_ce = math.inf
    iterator = training_order(eligible, settings["seed"])
    with runtime.install(model), paths["train"].open("rb") as train_stream, paths["valid"].open("rb") as valid_stream:
        model.train()
        for step, row in enumerate(islice(iterator, steps), start=1):
            optimizer.zero_grad(set_to_none=True)
            offload = torch.autograd.graph.save_on_cpu(pin_memory=True) if settings["cpu_activation_offload"] else nullcontext()
            with offload:
                task, sharp = losses(train_stream, row)
                total = task + settings["lambda_kl"] * sharp
            if not total.requires_grad or not bool(torch.isfinite(total)):
                raise RuntimeError("Expected a finite, differentiable answer CE + salience KL")
            total.backward()
            if step == 1:
                for label, parameters in (("Router", router), ("chunk representatives", representatives)):
                    if not any(parameter.grad is not None and bool(parameter.grad.abs().max() > 0) for parameter in parameters):
                        raise RuntimeError(f"No nonzero {label} gradient on the first update")
            torch.nn.utils.clip_grad_norm_(router + representatives, settings["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            print(json.dumps({
                "step": step, "bucket": row.bucket, "tokens": row.length,
                "answer_ce": float(task.detach()), "salience_kl": float(sharp.detach()), "loss": float(total.detach()),
            }), flush=True)

            validation_ce = None
            if step % settings["eval_every"] == 0 or step == steps:
                model.eval()
                with torch.no_grad():
                    metrics = [tuple(map(float, losses(valid_stream, item))) for item in validation]
                validation_ce = sum(ce for ce, _ in metrics) / len(metrics)
                validation_kl = sum(kl for _, kl in metrics) / len(metrics)
                if not math.isfinite(validation_ce) or not math.isfinite(validation_kl):
                    raise RuntimeError("Nonfinite validation loss")
                print(json.dumps({"step": step, "validation_ce": validation_ce,
                                  "validation_salience_kl": validation_kl}), flush=True)
                if validation_ce < best_ce:
                    save_checkpoint(args.output_dir / "best.pt", model, pave, profile, step, validation_ce)
                    best_ce = validation_ce
                model.train()
            if step % settings["save_every"] == 0 or step == steps:
                save_checkpoint(args.output_dir / "last.pt", model, pave, profile, step, validation_ce)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True, help="Local Base model and tokenizer directory")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with prepared splits and manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-python", type=Path, help="Llama-3 tokenizer environment's Python interpreter")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    profile = json.loads(args.config.read_text(encoding="utf-8"))
    if profile["separate_tokenizer"] and args.tokenizer_python is None:
        parser.error("Llama-3 requires --tokenizer-python from a newer Transformers environment")
    if args.output_dir.exists():
        parser.error(f"Choose a new output directory: {args.output_dir}")
    if not (args.model_path / "config.json").is_file():
        parser.error("--model-path must point to a local checkpoint directory")
    if not (args.data_dir / MANIFEST).is_file():
        parser.error(f"--data-dir must contain {MANIFEST}")

    import torch

    args.device = torch.device(args.device)
    if args.device.type != "cuda" or not torch.cuda.is_available():
        parser.error("PAVE training requires a CUDA GPU with BF16 support")
    torch.cuda.set_device(args.device.index if args.device.index is not None else torch.cuda.current_device())
    if not torch.cuda.is_bf16_supported():
        parser.error("PAVE training requires a CUDA GPU with BF16 support")
    with tokenizer_encoder(args, profile) as encode:
        run_training(args, profile, encode)


if __name__ == "__main__":
    main()
