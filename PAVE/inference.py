"""Load the PAVE inference modules on Llama-2-7B Base."""
from dataclasses import fields
from pathlib import Path


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


def load(model_path, checkpoint_path, *, routing_interval=512, intervention=None, decode=True):
    import torch
    from transformers import LlamaForCausalLM

    from . import model as core

    payload = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=True)
    config = checkpoint_config(payload, core.PAVEConfig)
    config.routing_interval = routing_interval
    core.replace_with_chunkllama()
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
    core.install_indexed_prefill()
    reset_decode = core.install_sparse_decode(model, sparse=True) if decode else None
    return model, core, config, reset_decode
