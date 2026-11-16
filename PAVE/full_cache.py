"""Allocate and update the full KV arena for PAVE inference."""
import torch
from transformers.cache_utils import DynamicCache


class FullKVCache(DynamicCache):
    def __init__(self, capacity=17408):
        super().__init__()
        self.capacity = capacity
        self.arenas = {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if torch.is_grad_enabled():
            raise RuntimeError('Use functional DynamicCache during autograd, not mutable arena')
        start = self.get_seq_length(layer_idx)
        end = start + key_states.shape[-2]
        if end > self.capacity:
            raise RuntimeError(f'Full KV capacity exceeded: {end}>{self.capacity}; no eviction')
        if layer_idx == 0:
            if hasattr(self,'_seen_tokens'):self._seen_tokens += key_states.shape[-2]
            else:self.seen_tokens += key_states.shape[-2]
        if layer_idx not in self.arenas:
            shape = list(key_states.shape); shape[-2] = self.capacity
            self.arenas[layer_idx] = (torch.empty(shape,device=key_states.device,dtype=key_states.dtype),
                                     torch.empty(shape,device=value_states.device,dtype=value_states.dtype))
            self.key_cache.append(None); self.value_cache.append(None)
        k,v = self.arenas[layer_idx]
        k[:,:,start:end].copy_(key_states); v[:,:,start:end].copy_(value_states)
        self.key_cache[layer_idx] = k[:,:,:end]
        self.value_cache[layer_idx] = v[:,:,:end]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        if len(self.key_cache)<=layer_idx or self.key_cache[layer_idx] is None:return 0
        return self.key_cache[layer_idx].shape[-2]
