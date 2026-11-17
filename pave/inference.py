"""Load PAVE inference modules on Llama-2-7B and Llama-3-8B Base."""
from dataclasses import fields
import hashlib
from pathlib import Path


LLAMA3_MODEL_ID = 'meta-llama/Meta-Llama-3-8B'
LLAMA3_BOS = 128000
LLAMA3_EOS = 128001
LLAMA3_DCA = {'pretraining_length': 8192, 'local_window_size': 1024}


def checkpoint_config(payload, config_class):
    config = payload.get('pave_config')
    if config is not None:
        return config_class(**config)
    # Match the configuration schema in archived checkpoints.
    expected = {field.name for field in fields(config_class)}
    candidates = [value for value in payload.values()
                  if isinstance(value, dict) and value.keys() == expected]
    if len(candidates) != 1:
        raise ValueError('PAVE configuration missing or ambiguous in checkpoint')
    return config_class(**candidates[0])


def verify_llama3_checkpoint(payload, model_path):
    if payload.get('backbone') == LLAMA3_MODEL_ID:
        if payload.get('dca') != LLAMA3_DCA:
            raise ValueError('Expected the Llama-3 DCA settings saved by the PAVE trainer')
        return
    # Older Llama-3 checkpoints identify their local backbone by file digests.
    identity = payload.get('backbone_identity', {})
    if identity.get('architecture', {}).get('num_key_value_heads') != 8:
        raise ValueError('Expected a Llama-3 PAVE checkpoint with backbone metadata')
    for filename, key in (('config.json', 'config_sha256'), ('tokenizer.json', 'tokenizer_sha256')):
        digest = hashlib.sha256((Path(model_path) / filename).read_bytes()).hexdigest()
        if identity.get(key) != digest:
            raise ValueError(f'Llama-3 checkpoint was recorded with a different {filename}')


def load(model_path, checkpoint_path, *, routing_interval=None, intervention=None, decode=True,
         backbone='llama2'):
    import torch
    from transformers import LlamaForCausalLM

    from . import model as core

    payload = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=True)
    if backbone == 'llama3':
        verify_llama3_checkpoint(payload, model_path)
        dca = LLAMA3_DCA
    elif backbone == 'llama2':
        dca = {}
        if routing_interval is None:
            routing_interval = 512
    else:
        raise ValueError(f'Unsupported PAVE backbone: {backbone}')
    config = checkpoint_config(payload, core.PAVEConfig)
    if routing_interval is not None:
        config.routing_interval = routing_interval
    core.replace_with_chunkllama(**dca)
    model = LlamaForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2',
        local_files_only=True,
    ).cuda().eval()
    core.attach_pave(model, config)
    weights = payload['trainable_state']
    if set(weights) != set(core.trainable_state_dict(model)):
        raise ValueError('Checkpoint does not contain the expected PAVE router and representative tensors')
    model.load_state_dict(weights, strict=False)
    model.requires_grad_(False)
    if intervention:
        intervention(model, core, config)
    if backbone == 'llama2' or config.routing_interval in (128, 512):
        core.install_indexed_prefill()
    reset_decode = core.install_sparse_decode(model, sparse=True) if decode else None
    return model, core, config, reset_decode


def generate_tokens(model, core, reset_decode, ids, cap, eos_token_id):
    """Generate greedily from an independent prompt with a full KV cache."""
    import torch
    from .model import FullKVCache

    def forward(tokens, cache):
        result = model.model(input_ids=tokens, past_key_values=cache, use_cache=True, return_dict=True)
        return model.lm_head(result.last_hidden_state[:, -1]), result.past_key_values

    reset_decode()
    core.reset_rotary_cache(model)
    prompt = torch.tensor([ids], device='cuda')
    cache = FullKVCache(capacity=prompt.shape[1] + cap + 1)
    with core.execution_context('sparse'):
        logits, cache = forward(prompt, cache)
        # The final Prefill logit supplies the first output token.
        token = logits.argmax(-1)
        generated = []
        for index in range(cap):
            value = int(token.item())
            generated.append(value)
            if value == eos_token_id or index + 1 == cap:
                break
            logits, cache = forward(token[:, None], cache)
            token = logits.argmax(-1)
    expected = prompt.shape[1] + len(generated) - 1
    if any(cache.get_seq_length(layer) != expected for layer in range(len(model.model.layers))):
        raise RuntimeError('Incomplete PAVE KV cache')
    return generated
