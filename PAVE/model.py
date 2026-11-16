"""PAVE routing, chunk summaries, and DCA-based Llama-2 attention.

DCA position handling and attention are adapted from HKUNLP/ChunkLlama,
Apache-2.0, commit 2add4d7c99d24dcc1ab03414cc602abb2e28cf2c.
"""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Callable
import math
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F
import transformers
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import rotate_half, repeat_kv
from flash_attn.flash_attn_interface import flash_attn_func

BLOCK_SIZE=128
EPS=1e-8
ROUTER_HIDDEN_SIZE=32
ROUTER_TEMPERATURE=1.0
ROUTER_INIT_RATIOS=(1/3,1/3,1/3)
ROUTER_OUTPUT_WEIGHT_INIT_STD=0.0
BOUNDARY_TEMPERATURE=0.1
SALIENCE_TEMPERATURE_DEFAULT=0.5
SCORE_TEMPERATURE=1.0
OBS_WINDOW_SIZE=128
ABLATION_MODE='full'
FIXED_ROUTER_RATIOS=None
MAX_NEW_TOKENS=512
chunk_size=None
local_window=None

class LegacyDistanceRouter(nn.Module):
    """Shared layer- and length-aware router over Sink, Remote, and Local."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int = ROUTER_HIDDEN_SIZE,
        temperature: float = ROUTER_TEMPERATURE,
        init_ratios: Sequence[float] = ROUTER_INIT_RATIOS,
    ):
        super().__init__()
        self.num_layers = max(int(num_layers), 1)
        self.hidden_size = int(hidden_size)
        self.temperature = float(temperature)
        self.layer_embedding = nn.Embedding(self.num_layers, self.hidden_size)
        self.router = nn.Sequential(
            nn.Linear(self.hidden_size + 1, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, 3),
        )
        self.reset_parameters(init_ratios=init_ratios)

    @torch.no_grad()
    def reset_parameters(
        self,
        init_ratios: Sequence[float] = ROUTER_INIT_RATIOS,
    ) -> None:
        self.layer_embedding.reset_parameters()
        self.router[0].reset_parameters()
        if ROUTER_OUTPUT_WEIGHT_INIT_STD == 0.0:
            nn.init.zeros_(self.router[2].weight)
        else:
            nn.init.normal_(
                self.router[2].weight,
                mean=0.0,
                std=ROUTER_OUTPUT_WEIGHT_INIT_STD,
            )

        init = torch.as_tensor(init_ratios, dtype=torch.float32).clamp_min(EPS)
        if init.numel() != 3:
            raise ValueError(f"init_ratios must contain exactly three elements, got {init_ratios}")
        init = init / init.sum()
        self.router[2].bias.copy_(
            torch.log(init).to(
                device=self.router[2].bias.device,
                dtype=self.router[2].bias.dtype,
            )
        )

    def predict_ratios(
        self,
        num_blocks: int,
        layer_idx: Optional[int],
        device=None,
        dtype=torch.float32,
    ) -> torch.Tensor:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        safe_layer_idx = 0 if layer_idx is None else int(layer_idx)
        safe_layer_idx = max(0, min(safe_layer_idx, self.num_layers - 1))
        param = next(self.parameters())
        device = param.device if device is None else device
        param_dtype = param.dtype

        if ABLATION_MODE == "fixed_router":
            if FIXED_ROUTER_RATIOS is None:
                raise RuntimeError(
                    "FIXED_ROUTER_RATIOS is not configured for fixed_router mode"
                )
            return torch.tensor(
                FIXED_ROUTER_RATIOS,
                device=device,
                dtype=dtype,
            )

        layer_id = torch.tensor([safe_layer_idx], device=device, dtype=torch.long)
        log_m = torch.log1p(
            torch.tensor([[float(num_blocks)]], device=device, dtype=param_dtype)
        )
        layer_emb = self.layer_embedding(layer_id)
        logits = self.router(torch.cat([log_m, layer_emb], dim=-1)).squeeze(0)
        logits = logits.float() / max(float(self.temperature), EPS)
        return torch.softmax(logits, dim=-1).to(dtype=dtype)

    def continuous_state_probabilities(
        self,
        ratios: torch.Tensor,
        num_blocks: int,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate soft Sink, Remote, and Local probabilities for each block."""
        if num_blocks <= 0:
            return torch.empty(0, 3, device=ratios.device, dtype=ratios.dtype)
        tau = BOUNDARY_TEMPERATURE if temperature is None else float(temperature)
        tau = max(tau, EPS)
        positions = (
            torch.arange(num_blocks, device=ratios.device, dtype=torch.float32) + 0.5
        ) / float(num_blocks)
        b_sink = ratios[0].float()
        b_local = 1.0 - ratios[2].float()

        sink = torch.sigmoid((b_sink - positions) / tau)
        local = torch.sigmoid((positions - b_local) / tau)
        remote = torch.sigmoid((positions - b_sink) / tau) * torch.sigmoid(
            (b_local - positions) / tau
        )
        probs = torch.stack([sink, remote, local], dim=-1)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(EPS)
        return probs.to(dtype=ratios.dtype)

    def forward(
        self,
        num_blocks: int,
        layer_idx: Optional[int],
        device,
        dtype=torch.float32,
    ):
        if num_blocks <= 0:
            empty_probs = torch.empty(0, 3, device=device, dtype=dtype)
            empty_ratios = torch.empty(3, device=device, dtype=dtype)
            return empty_probs, empty_ratios, (0, 0, 0)
        ratios = self.predict_ratios(num_blocks, layer_idx, device, dtype)
        soft_probs = self.continuous_state_probabilities(ratios, num_blocks)
        counts, _ = _hard_contiguous_state_assignment(ratios, num_blocks)
        return soft_probs, ratios, counts

class LearnableBlockRepresentative(nn.Module):
    """Learn token salience and pool each block with temperature sharpening."""

    def __init__(
        self,
        head_dim: int,
        temperature: float = SALIENCE_TEMPERATURE_DEFAULT,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(head_dim)
        self.salience = nn.Linear(head_dim, 1, bias=False)
        self.temperature = float(temperature)
        nn.init.normal_(self.salience.weight, mean=0.0, std=0.02)

    def set_temperature(self, temperature: float) -> None:
        value = float(temperature)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"temperature must be finite and positive, got {temperature}")
        self.temperature = value

    def forward(self, k_blocks: torch.Tensor):
        """Return block representatives, token weights, and monitoring entropy."""
        orig_dtype = k_blocks.dtype

        if ABLATION_MODE == "mean_representative":
            block_size = int(k_blocks.size(-2))
            if block_size <= 0:
                raise ValueError("k_blocks must have a positive block_size dimension")
            weights_fp32 = torch.full(
                k_blocks.shape[:-1],
                1.0 / float(block_size),
                device=k_blocks.device,
                dtype=torch.float32,
            )
            weights = weights_fp32.to(dtype=orig_dtype)
            block_repr = k_blocks.mean(dim=-2)
            entropy = torch.tensor(
                math.log(float(block_size)),
                device=k_blocks.device,
                dtype=torch.float32,
            )
            return block_repr, weights, entropy

        if ABLATION_MODE == "max_token_representative":
            # Deterministic non-learned heuristic: choose the Key token with
            # the largest L2 norm in each block.  This removes the learnable
            # LayerNorm + salience scorer and also removes token aggregation.
            # Squared norm is sufficient because sqrt is monotonic.
            token_scores = k_blocks.float().square().sum(dim=-1)
            max_index = token_scores.argmax(dim=-1, keepdim=True)
            gather_index = max_index.unsqueeze(-1).expand(
                *max_index.shape, k_blocks.size(-1)
            )
            block_repr = torch.gather(
                k_blocks, dim=-2, index=gather_index
            ).squeeze(-2)

            weights_fp32 = torch.zeros(
                k_blocks.shape[:-1],
                device=k_blocks.device,
                dtype=torch.float32,
            )
            weights_fp32.scatter_(-1, max_index, 1.0)
            weights = weights_fp32.to(dtype=orig_dtype)
            entropy = torch.zeros(
                (), device=k_blocks.device, dtype=torch.float32
            )
            return block_repr, weights, entropy

        normed = F.layer_norm(
            k_blocks,
            normalized_shape=(k_blocks.size(-1),),
            weight=self.norm.weight.to(dtype=orig_dtype),
            bias=self.norm.bias.to(dtype=orig_dtype),
        )
        scores = F.linear(
            normed, self.salience.weight.to(dtype=orig_dtype)
        ).squeeze(-1)
        # Temperature-scaled softmax implements normalized salience sharpening.
        weights_fp32 = torch.softmax(
            scores.float() / max(self.temperature, EPS), dim=-1
        )
        weights = weights_fp32.to(dtype=orig_dtype)
        block_repr = torch.sum(weights.unsqueeze(-1) * k_blocks, dim=-2)
        entropy = -(
            weights_fp32 * torch.log(weights_fp32.clamp_min(EPS))
        ).sum(dim=-1).mean()
        return block_repr, weights, entropy

def _largest_remainder_counts(
    ratios: torch.Tensor,
    total_blocks: int,
) -> Tuple[int, int, int]:
    """Convert continuous ratios into three integer region sizes summing to M."""
    total = int(total_blocks)
    if total < 0:
        raise ValueError(f"total_blocks must be non-negative, got {total}")
    if total == 0:
        return 0, 0, 0
    probs = ratios.detach().float().clamp_min(0)
    probs = probs / probs.sum().clamp_min(EPS)
    raw = probs * float(total)
    base = torch.floor(raw).to(torch.long)
    # Allocate on device, then transfer all three region sizes together.
    remainder=total-base.sum()
    order=torch.argsort(raw-base.float(),descending=True,stable=True)
    increments=(torch.arange(3,device=ratios.device)<remainder).to(torch.long)
    base=base+torch.zeros_like(base).scatter(0,order,increments)
    result=tuple(base.tolist())
    if sum(result)!=total or min(result)<0:raise RuntimeError('Invalid integer allocation')
    return result

def _hard_contiguous_state_assignment(
    ratios: torch.Tensor,
    num_blocks: int,
) -> Tuple[Tuple[int, int, int], torch.Tensor]:
    """Create the hard contiguous Sink-to-Remote-to-Local assignment."""
    counts = _largest_remainder_counts(ratios, num_blocks)
    sink_count, remote_count, local_count = counts
    state_ids = torch.empty(
        num_blocks, device=ratios.device, dtype=torch.long
    )
    state_ids[:sink_count] = 0
    state_ids[sink_count:sink_count + remote_count] = 1
    state_ids[sink_count + remote_count:] = 2
    return counts, state_ids

def compute_attention_concentration(
    remote_scores: torch.Tensor,
    temperature: float = SCORE_TEMPERATURE,
) -> torch.Tensor:
    """Compute concentration from normalized entropy of Remote block scores."""
    num_remote = remote_scores.size(-1)
    if num_remote <= 1:
        return torch.ones(
            remote_scores.size(0),
            device=remote_scores.device,
            dtype=torch.float32,
        )
    probs = torch.softmax(
        remote_scores.float() / max(float(temperature), EPS), dim=-1
    )
    entropy = -torch.sum(probs * torch.log(probs.clamp_min(EPS)), dim=-1)
    concentration = 1.0 - entropy / math.log(float(num_remote))
    return concentration.clamp(0.0, 1.0)

def _select_blocks_by_batch(blocks: torch.Tensor, indices: torch.Tensor):
    """Gather selected blocks independently for each batch element."""
    bsz, heads, _, block_size, head_dim = blocks.shape
    if bsz==1:
        # One index per whole contiguous token block, rather than expanding an
        # index to every head/token/channel element of a generic gather.
        selected=blocks.index_select(2,indices[0])
        return selected.view(bsz,heads,-1,head_dim)
    gather_idx = indices[:, None, :, None, None].expand(-1, heads, -1, block_size, head_dim)
    selected = torch.gather(blocks, dim=2, index=gather_idx)
    return selected.contiguous().view(bsz, heads, -1, head_dim)


class LayerAwareDistanceRouter(LegacyDistanceRouter):
    """Shared query-content + layer + length router; one request per batch.

    Pool strictly prior queries supplied by the causal routing loop. Query
    observation length is independent of KV block size and routing interval.
    """
    def __init__(self,num_layers,query_dim,hidden_size=ROUTER_HIDDEN_SIZE,observation_window=OBS_WINDOW_SIZE,**kwargs):
        super().__init__(num_layers,hidden_size=hidden_size,**kwargs)
        self.query_dim=int(query_dim)
        self.observation_window=int(observation_window)
        self.query_norm=nn.LayerNorm(self.query_dim)
        self.query_projection=nn.Linear(self.query_dim,hidden_size)
        self.query_output=nn.Linear(hidden_size,3,bias=False)
        nn.init.normal_(self.query_output.weight,std=.01)

    def predict_ratios(self,num_blocks,layer_idx,device=None,dtype=torch.float32,query=None):
        if query is None:raise ValueError('Query-aware router requires query; length-only fallback is forbidden')
        if query.shape[0]!=1:raise ValueError('Content-aware boundary allocation currently requires batch size 1')
        base=super().predict_ratios(num_blocks,layer_idx,device,dtype=torch.float32)
        if ABLATION_MODE == 'fixed_router':
            return base.to(dtype)
        pooled=query[:,:,-min(self.observation_window,query.shape[-2]):,:].float().mean(2).reshape(1,-1)
        if pooled.shape[-1]!=self.query_dim:raise ValueError('Query feature dimension mismatch')
        content=self.query_output(F.silu(self.query_projection(self.query_norm(pooled)))).squeeze(0)
        return (base.clamp_min(EPS).log()+content).softmax(-1).to(dtype)


def get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.05 * math.log(scale) + 1.0

class ChunkLlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000, scaling_factor=1.0, device=None):
        super().__init__()

        self.max_seq_len = max_position_embeddings
        self.dim = dim
        self.scaling_factor = scaling_factor
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=self.max_seq_len,
            device=device, dtype=torch.float32
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # employing yarn will lead to better performance but results reported in our paper did not use yarn.
        scale = seq_len / self.max_position_embeddings
        mscale = get_mscale(scale)
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        chunk_len = chunk_size - local_window
        q_t = torch.arange(chunk_len, device=device, dtype=self.inv_freq.dtype) / self.scaling_factor
        qc_t = (torch.arange(chunk_len, device=device, dtype=self.inv_freq.dtype) + chunk_len).clamp(
            max=chunk_size) / self.scaling_factor
        k_t = (torch.arange(seq_len + MAX_NEW_TOKENS, device=device,
                            dtype=self.inv_freq.dtype) % chunk_len) / self.scaling_factor

        q_freqs = torch.outer(q_t, self.inv_freq)  # seq_len x dim/2
        qc_freqs = torch.outer(qc_t, self.inv_freq)
        k_freqs = torch.outer(k_t, self.inv_freq)  # seq_len x dim/2

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        q_emb = torch.cat((q_freqs, q_freqs), dim=-1)  # seq_len x dim
        qc_emb = torch.cat((qc_freqs, qc_freqs), dim=-1)
        k_emb = torch.cat((k_freqs, k_freqs), dim=-1)  # seq_len x dim
        self.register_buffer("q_cos_cached", q_emb.cos().to(dtype) * mscale, persistent=False)
        self.register_buffer("q_sin_cached", q_emb.sin().to(dtype) * mscale, persistent=False)
        self.register_buffer("qc_cos_cached", qc_emb.cos().to(dtype) * mscale, persistent=False)
        self.register_buffer("qc_sin_cached", qc_emb.sin().to(dtype) * mscale, persistent=False)
        self.register_buffer("k_cos_cached", k_emb.cos().to(dtype) * mscale, persistent=False)
        self.register_buffer("k_sin_cached", k_emb.sin().to(dtype) * mscale, persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # no token will exceed chunk_size
        # chunk1_q,
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len=seq_len, device=self.inv_freq.device, dtype=torch.float32)
            self.max_seq_len = seq_len
        return (
            self.q_cos_cached[:seq_len].to(dtype=x.dtype),
            self.q_sin_cached[:seq_len].to(dtype=x.dtype),
            self.qc_cos_cached[:seq_len].to(dtype=x.dtype),
            self.qc_sin_cached[:seq_len].to(dtype=x.dtype),
            self.k_cos_cached[:seq_len].to(dtype=x.dtype),
            self.k_sin_cached[:seq_len].to(dtype=x.dtype),
        )

def apply_rotary_pos_emb(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_emb = (x * cos) + (rotate_half(x) * sin)
    return x_emb


def _cached_attention_values(q_states_intra,key_states,value_states,q_states_succ,q_states_inter,attention_mask,head_dim):
    """Pure original cached-attention math; no routing, cache writes or hooks."""
    chunk_len=chunk_size-local_window
    bsz,_,q_len,_=q_states_intra.shape
    kv_seq_len=key_states.shape[-2]
    chunk_num_curr = (kv_seq_len - 1) // chunk_len
    k_states_intra = key_states[:, :, chunk_len * chunk_num_curr:kv_seq_len, :]
    attn_weights = torch.matmul(q_states_intra, k_states_intra.transpose(2, 3)) / math.sqrt(
        head_dim)
    attn_scores = [attn_weights]

    if chunk_num_curr >= 1:
        k_states_succ = key_states[:, :, chunk_len * (chunk_num_curr - 1):chunk_len * chunk_num_curr, :]
        attn_weights = torch.matmul(q_states_succ, k_states_succ.transpose(2, 3)) / math.sqrt(
            head_dim)
        attn_scores = [attn_weights] + attn_scores

    if chunk_num_curr >= 2:
        k_states_inter = key_states[:, :, :chunk_len * (chunk_num_curr - 1), :]
        attn_weights = torch.matmul(q_states_inter, k_states_inter.transpose(2, 3)) / math.sqrt(
            head_dim)
        attn_scores = [attn_weights] + attn_scores

    attn_weights = torch.cat(attn_scores, dim=-1)
    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_states_intra.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output


def _dca_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    chunk_len = chunk_size - local_window

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # during inference
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    q_seq_len = query_states.shape[-2]
    has_kv_cache = q_seq_len != kv_seq_len
    # covert to b x head x len x h
    # need to chunk query states
    q_cos, q_sin, qc_cos, qc_sin, k_cos, k_sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    key_states = apply_rotary_pos_emb(key_states, k_cos, k_sin, position_ids)
    position_ids = position_ids % chunk_len

    if past_key_value is not None:
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs=None)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    flash_results = []
    if not has_kv_cache:
        q_states_intra = apply_rotary_pos_emb(query_states[:, :, :chunk_len, :], q_cos, q_sin,
                                             position_ids[:, :chunk_len])
        k_states_prev = key_states[:, :, :chunk_len, :]
        v_states_prev = value_states[:, :, :chunk_len, :]
        flash_results.append(do_flash_attn(q_states_intra, k_states_prev, v_states_prev))
        remain_len = kv_seq_len - chunk_len

        while remain_len > 0:
            flash_per_chunk = []
            begin = kv_seq_len - remain_len
            curr_chunk_len = min(chunk_len, remain_len)
            end = begin + curr_chunk_len

            q_states_intra = apply_rotary_pos_emb(query_states[:, :, begin:end, :], q_cos, q_sin,
                                                 position_ids[:, begin:end])

            k_states_intra = key_states[:, :, begin:end, :]
            v_states_intra = value_states[:, :, begin:end, :]
            flash_per_chunk.append(do_flash_attn(q_states_intra, k_states_intra, v_states_intra))

            q_states_succ = apply_rotary_pos_emb(query_states[:, :, begin:end, :], qc_cos, qc_sin,
                                                  position_ids[:, begin:end])
            flash_per_chunk.append(do_flash_attn(q_states_succ, k_states_prev, v_states_prev, False))

            if begin - (k_states_prev.size(-2)) > 0:
                prev_len = k_states_prev.size(-2)
                q_states_inter = apply_rotary_pos_emb(query_states[:, :, begin:end, :], qc_cos, qc_sin,
                                                    position_ids[:, chunk_len - 1][:, None].repeat(1, curr_chunk_len))
                k_states_inter = key_states[:, :, :begin - prev_len, :]
                v_states_inter = value_states[:, :, :begin - prev_len, :]
                flash_per_chunk.append(inter_attention_microchunks(
                    self,q_states_inter,k_states_inter,v_states_inter,query_states,
                    begin,position_ids[:,chunk_len-1],qc_cos,qc_sin))

            flash_results.append(flash_per_chunk)
            k_states_prev = k_states_intra
            v_states_prev = v_states_intra
            remain_len = remain_len - chunk_len

        attn_output = merge_attn_outputs(flash_results)
    else:
        chunk_num_curr=(kv_seq_len-1)//chunk_len
        q_states_intra=apply_rotary_pos_emb(query_states,q_cos,q_sin,position_ids)
        q_states_succ=apply_rotary_pos_emb(query_states,qc_cos,qc_sin,position_ids) if chunk_num_curr>=1 else None
        q_states_inter=apply_rotary_pos_emb(query_states,qc_cos,qc_sin,torch.tensor([[chunk_len-1]],device=query_states.device)) if chunk_num_curr>=2 else None
        inputs=(q_states_intra,key_states,value_states,q_states_succ,q_states_inter,attention_mask,self.head_dim)
        attn_output=_cached_attention_values(*inputs)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

@dataclass
class PAVEConfig:
    initial_remote_keep_ratio: float = 0.5
    min_remote_keep_ratio: float = 0.05
    sparsity_decay_strength: float = 3.0
    concentration_gamma: float = 1.0
    salience_temperature: float = 0.5
    boundary_temperature: float = 0.1
    routing_interval: int = 128
    query_observation_tokens: int = 128
    reuse_representatives: bool = True
    reuse_packed_kv: bool = False

    def validate(self):
        if self.routing_interval not in (128,256,512,2560):
            raise ValueError('Supported query routing intervals: 128,256,512,2560')
        if self.query_observation_tokens<=0:
            raise ValueError('query_observation_tokens must be positive')
        if not 0 < self.min_remote_keep_ratio <= self.initial_remote_keep_ratio <= 1:
            raise ValueError('Require 0 < min keep <= initial keep <= 1')
        if self.sparsity_decay_strength < 0 or self.concentration_gamma <= 0:
            raise ValueError('Invalid concentration recurrence')
        if self.salience_temperature <= 0 or self.boundary_temperature <= 0:
            raise ValueError('Temperatures must be positive')


@dataclass
class ExecutionState:
    mode: str = 'sparse'
    keep_rates: dict = field(default_factory=dict)
    proposal_hook: Optional[Callable] = None
    remote_selector: Optional[Callable] = None
    current_query: Optional[Tuple[int,int]] = None
    current_group: Optional[int] = None
    representative_cache: Optional[tuple] = None
    planned_allocations: Optional[dict] = None
    pending_keep_updates: Optional[list] = None
    packed_kv_cache: Optional[tuple] = None

_STATE=ContextVar('pave_execution_state',default=None)


@contextmanager
def execution_context(mode='sparse',proposal_hook=None,remote_selector=None):
    if mode not in ('sparse','direct'):
        raise ValueError(mode)
    state=ExecutionState(mode=mode,
                         proposal_hook=proposal_hook,
                         remote_selector=remote_selector)
    token=_STATE.set(state)
    try:
        yield state
    finally:
        _STATE.reset(token)


@dataclass
class Proposal:
    history_blocks: int
    sink: int
    remote: int
    local: int
    keep: int
    ratio: float
    ratios: torch.Tensor
    layer: int
    num_layers: int
    query_scope: Optional[Tuple[int,int]] = None


def make_proposal(attn,q,k,state,routing_query=None):
    config=attn._pave_config
    router=attn._shared_memory_router_ref
    h=k.shape[-2]//BLOCK_SIZE
    scope=state.current_query
    planned=state.planned_allocations.get(scope) if state.planned_allocations is not None else None
    if planned is not None:
        ratios,(s,m,l)=planned
    else:
        ratios=router.predict_ratios(h,attn.layer_idx,device=k.device,dtype=torch.float32,
                                     query=q if routing_query is None else routing_query)
        s,m,l=_largest_remainder_counts(ratios,h)
    rho=state.keep_rates.get((int(attn.layer_idx),h,scope),config.initial_remote_keep_ratio)
    keep=min(m,max(1,math.ceil(m*rho))) if m else 0
    return Proposal(h,s,m,l,keep,rho,ratios,int(attn.layer_idx),router.num_layers,scope)


def representative_signature(attn,k,p,state):
    # A single-entry, request-local cache; never reuse across requests, layers,
    # DCA groups, or different Remote boundaries.
    if not attn._pave_config.reuse_representatives or state.current_group is None or not p.remote:
        return None
    return (p.layer,state.current_group,p.sink,p.remote,k.data_ptr(),tuple(k.shape),tuple(k.stride()))


def update_keep_rate(attn,p,state,concentration=None,observed_value=None):
    # Direct branch has no concentration observation: explicitly HOLD, never fabricate one.
    rho=p.ratio
    if concentration is not None:
        if state.pending_keep_updates is not None and observed_value is None:
            state.pending_keep_updates.append((attn,p,concentration))
            return
        c=float(concentration.detach().float().mean().cpu()) if observed_value is None else observed_value
        cfg=attn._pave_config
        decay=math.exp(-cfg.sparsity_decay_strength*c**cfg.concentration_gamma/max(p.num_layers-1,1))
        rho=cfg.min_remote_keep_ratio+(rho-cfg.min_remote_keep_ratio)*decay
        rho=min(p.ratio,max(cfg.min_remote_keep_ratio,rho))
    if p.layer+1<p.num_layers:
        state.keep_rates[(p.layer+1,p.history_blocks,p.query_scope)]=rho


def flush_keep_updates(state):
    """Only next-layer scopes consume these states: one D2H transfer per group.

    Preserve the reference Python/double decay formula, not a float32 rewrite.
    Do not merge or share concentration observations between microqueries.
    """
    pending=state.pending_keep_updates
    if not pending:return
    observed=torch.stack([c.detach().float().mean() for _,_,c in pending]).cpu().tolist()
    state.pending_keep_updates=None
    try:
        for (attn,p,c),value in zip(pending,observed,strict=True):
            update_keep_rate(attn,p,state,c,observed_value=value)
    finally:
        state.pending_keep_updates=pending
        pending.clear()


def plan_microquery_allocations(attn,raw_queries,begin,length,step,history_blocks):
    """Batch the same causal Router computation; never pool future queries."""
    router=attn._shared_memory_router_ref;window=attn._pave_config.query_observation_tokens
    scopes=[(begin+offset,begin+min(offset+step,length)) for offset in range(0,length,step)]
    base=LegacyDistanceRouter.predict_ratios(router,history_blocks,attn.layer_idx,
                                             device=raw_queries.device,dtype=torch.float32)
    if ABLATION_MODE=='fixed_router':ratios=base.expand(len(scopes),-1)
    else:
        pooled=torch.stack([raw_queries[:,:,max(0,start-window):start].float().mean(2).reshape(-1)
                            for start,_ in scopes])
        content=router.query_output(F.silu(router.query_projection(router.query_norm(pooled))))
        ratios=(base.clamp_min(EPS).log()[None]+content).softmax(-1)
    probabilities=ratios.detach().float().clamp_min(0)
    probabilities=probabilities/probabilities.sum(-1,keepdim=True).clamp_min(EPS)
    raw=probabilities*float(history_blocks);counts=raw.floor().long()
    remainder=history_blocks-counts.sum(-1,keepdim=True)
    order=torch.argsort(raw-counts.float(),dim=-1,descending=True,stable=True)
    increments=(torch.arange(3,device=ratios.device)[None]<remainder).long()
    counts=counts+torch.zeros_like(counts).scatter(1,order,increments)
    host_counts=counts.cpu().tolist()
    if any(sum(row)!=history_blocks or min(row)<0 for row in host_counts):raise RuntimeError('Invalid batched allocation')
    return {scope:(ratios[i],tuple(host_counts[i])) for i,scope in enumerate(scopes)}


def prepare_selected_kv(attn,k,v,kb,vb,ids,p,state):
    """Reuse a gathered KV pair only for an identical selection in the same group."""
    enabled=attn._pave_config.reuse_packed_kv and state.current_group is not None
    signature=(p.layer,state.current_group,k.data_ptr(),v.data_ptr(),tuple(k.shape),
               tuple(v.shape),tuple(k.stride()),tuple(v.stride()),k.dtype,v.dtype,k.device)
    cached=state.packed_kv_cache if enabled else None
    if cached is not None and cached[0]==signature and cached[1].shape==ids.shape and torch.equal(cached[1],ids):
        return cached[2],cached[3]
    sk=_select_blocks_by_batch(kb,ids)
    sv=_select_blocks_by_batch(vb,ids)
    if enabled:
        state.packed_kv_cache=(signature,ids.detach(),sk,sv)
    return sk,sv


def sparse_select_from_proposal(attn,q,k,v,p,state,selection_query=None):
    b,heads,t,d=k.shape
    aligned=p.history_blocks*BLOCK_SIZE
    kb=k[:,:,:aligned,:].view(b,heads,p.history_blocks,BLOCK_SIZE,d)
    vb=v[:,:,:aligned,:].view(b,heads,p.history_blocks,BLOCK_SIZE,d)
    device=k.device
    remote_ids=torch.arange(p.sink,p.sink+p.remote,device=device)
    concentration=None
    scores=None
    hit=False
    if p.remote:
        signature=representative_signature(attn,k,p,state)
        hit=bool(signature is not None and state.representative_cache is not None
                 and state.representative_cache[0]==signature)
        if hit:
            _,representations,rep_entropy=state.representative_cache
        else:
            candidates=kb[:,:,p.sink:p.sink+p.remote]
            representations,_,rep_entropy=attn.block_representative(candidates)
            state.representative_cache=(signature,representations,rep_entropy) if signature is not None else None
        source=q if selection_query is None else selection_query
        window=attn._pave_config.query_observation_tokens
        probe=source[:,:,-min(window,source.shape[-2]):,:].mean(-2)
        scores=((probe.unsqueeze(2)@representations.transpose(-1,-2)).squeeze(2)/math.sqrt(d)).mean(1)
        concentration=compute_attention_concentration(scores)
        top=(scores.topk(p.keep,dim=-1).indices if state.remote_selector is None
             else state.remote_selector(scores,p))
        remote_selected=remote_ids[top]
    else:
        remote_selected=torch.empty(b,0,device=device,dtype=torch.long)
    parts=[torch.arange(p.sink,device=device)[None].expand(b,-1),remote_selected,
           torch.arange(p.sink+p.remote,p.history_blocks,device=device)[None].expand(b,-1)]
    ids=torch.cat(parts,dim=-1).sort(-1).values
    sk,sv=prepare_selected_kv(attn,k,v,kb,vb,ids,p,state)
    if aligned<t:
        sk=torch.cat([sk,k[:,:,aligned:,:]],-2);sv=torch.cat([sv,v[:,:,aligned:,:]],-2)
    update_keep_rate(attn,p,state,concentration)
    return sk,sv


def select_inter_memory(attn,q,k,v,routing_query=None,selection_query=None):
    state=_STATE.get()
    if state is None:
        raise RuntimeError('Missing request execution context; use installed model wrapper')
    if k.shape[-2]<BLOCK_SIZE:
        return k,v
    p=make_proposal(attn,q,k,state,routing_query=routing_query)
    if state.proposal_hook is not None:
        state.proposal_hook(attn,q,k,v,p)
    if state.mode=='direct':
        update_keep_rate(attn,p,state,None)
        return k,v
    return sparse_select_from_proposal(attn,q,k,v,p,state,selection_query=selection_query)


def inter_attention_microchunks(attn,q,k,v,raw_queries,begin,inter_position,cos,sin):
    """Split only inter attention; preserve DCA intra/successive branches and RoPE.

    A decision for [start,end) may only observe queries at positions < start.
    It is shared within that microchunk; same-microchunk question lag remains.
    """
    state=_STATE.get()
    if state is None:raise RuntimeError('Missing execution context')
    cfg=attn._pave_config
    step=q.shape[-2] if state.mode=='direct' else cfg.routing_interval
    outputs=[];normalizers=[]
    previous=(state.current_query,state.current_group,state.planned_allocations,state.pending_keep_updates,state.packed_kv_cache)
    try:
        state.current_group=begin
        state.packed_kv_cache=None
        state.planned_allocations=None;state.pending_keep_updates=None
        if state.mode!='direct' and k.shape[-2]>=BLOCK_SIZE:
            state.planned_allocations=plan_microquery_allocations(attn,raw_queries,begin,q.shape[-2],step,k.shape[-2]//BLOCK_SIZE)
        if state.mode!='direct':state.pending_keep_updates=[]
        for offset in range(0,q.shape[-2],step):
            stop=min(offset+step,q.shape[-2])
            start=begin+offset
            state.current_query=(start,begin+stop)
            prior=raw_queries[:,:,max(0,start-cfg.query_observation_tokens):start,:]
            if prior.shape[-2]==0:raise RuntimeError('Inter memory requires a nonempty prior query prefix')
            positions=inter_position[:,None].expand(-1,prior.shape[-2])
            selection=apply_rotary_pos_emb(prior,cos,sin,positions)
            sk,sv=select_inter_memory(attn,q[:,:,offset:stop,:],k,v,
                                      routing_query=prior,selection_query=selection)
            out,lse=do_flash_attn(q[:,:,offset:stop,:],sk,sv,False)
            outputs.append(out);normalizers.append(lse)
        if state.pending_keep_updates:
            flush_keep_updates(state)
        return torch.cat(outputs,dim=2),torch.cat(normalizers,dim=2)
    finally:
        state.current_query,state.current_group,state.planned_allocations,state.pending_keep_updates,state.packed_kv_cache=previous


def _native_flash_values(query_states,key_states,value_states,causal=True):
    # DCA native normalization data, NOT the old tiled FP32 recomputation.
    out,lse,_=flash_attn_func(query_states.transpose(1,2),key_states.transpose(1,2),
        value_states.transpose(1,2),dropout_p=0.0,causal=causal,return_attn_probs=True)
    return out.transpose(1,2),lse


def do_flash_attn(query_states,key_states,value_states,causal=True):
    return _native_flash_values(query_states,key_states,value_states,causal)


def _merge_attn_outputs(flash_results):
    # Same DCA detached branch normalization, without an in-place autograd mutation.
    results=[flash_results[0][0]]
    for branches in flash_results[1:]:
        outputs=torch.stack([x[0] for x in branches])
        lse=torch.stack([x[1] for x in branches]).float()
        weights=torch.softmax(lse,dim=0).detach().to(outputs.dtype)
        results.append((outputs*weights.unsqueeze(-1)).sum(0))
    return torch.cat(results,dim=2)


def merge_attn_outputs(flash_results):
    return _merge_attn_outputs(flash_results)


def forward(self,hidden_states,attention_mask=None,position_ids=None,past_key_value=None,
            output_attentions=False,use_cache=False,**kwargs):
    if output_attentions:
        raise ValueError('DCA flash path does not support output_attentions=True')
    if position_ids is None:
        raise ValueError('DCA requires explicit position_ids from LlamaModel')
    cached=0 if past_key_value is None else past_key_value.get_usable_length(hidden_states.shape[1],self.layer_idx)
    if cached and hidden_states.shape[1]!=1:
        raise ValueError('DCA cached path supports single-token decode only; chunked cached prefill is unsupported')
    if not cached and attention_mask is not None:
        # Dense causal 4-D masks generated by eager/SDPA are fine. Reject padding/additional masks.
        # Validation is done once at the backbone wrapper using the original 2-D mask.
        attention_mask=None
    return _dca_forward(self,hidden_states,attention_mask,position_ids,past_key_value,
                        output_attentions,use_cache,**kwargs)


def _model_forward_wrapper(original):
    def wrapped(self,*args,**kwargs):
        mask=kwargs.get('attention_mask',args[1] if len(args)>1 else None)
        if mask is not None and (mask.ndim!=2 or not bool(mask.eq(1).all())):
            raise ValueError('Unpadded batches only: DCA upstream ignores padding in Prefill')
        past=kwargs.get('past_key_values',args[3] if len(args)>3 else None)
        if past is None or (isinstance(past,(tuple,list)) and not past) or (hasattr(past,'get_seq_length') and past.get_seq_length()==0):
            reset_rotary_cache(self)
        if _STATE.get() is not None:
            return original(self,*args,**kwargs)
        mode=getattr(self,'_pave_default_mode','sparse')
        with execution_context(mode):
            return original(self,*args,**kwargs)
    return wrapped


def replace_with_chunkllama(pretraining_length=4096,local_window_size=None):
    global chunk_size,local_window
    chunk_size=int(pretraining_length)*3//4
    local_window=int(local_window_size) if local_window_size is not None else int(pretraining_length)//8
    if not 0<local_window<chunk_size:
        raise ValueError('Invalid DCA chunk/window')
    mod=transformers.models.llama.modeling_llama
    if not transformers.__version__.startswith('4.37.'):
        raise RuntimeError('This DCA integration targets transformers 4.37.x; other APIs require explicit validation')
    mod.LlamaRotaryEmbedding=ChunkLlamaRotaryEmbedding
    mod.LlamaLinearScalingRotaryEmbedding=ChunkLlamaRotaryEmbedding
    for name in ('LlamaAttention','LlamaFlashAttention2','LlamaSdpaAttention'):
        cls=getattr(mod,name,None)
        if cls is not None:cls.forward=forward
    if not getattr(mod.LlamaModel,'_pave_v2_wrapped',False):
        mod.LlamaModel.forward=_model_forward_wrapper(mod.LlamaModel.forward)
        mod.LlamaModel._pave_v2_wrapped=True


def attach_pave(model,config=None,default_mode='sparse'):
    if default_mode not in ('sparse','direct'):raise ValueError(default_mode)
    cfg=config or PAVEConfig();cfg.validate()
    backbone=model.model if hasattr(model,'model') else model
    device=next(backbone.parameters()).device
    if not hasattr(backbone,'shared_memory_router'):
        backbone.shared_memory_router=LayerAwareDistanceRouter(len(backbone.layers),backbone.config.num_attention_heads * (backbone.config.hidden_size//backbone.config.num_attention_heads),observation_window=cfg.query_observation_tokens).to(device=device,dtype=torch.float32)
    backbone.shared_memory_router.observation_window=cfg.query_observation_tokens
    for layer in backbone.layers:
        attn=layer.self_attn
        if not hasattr(attn,'block_representative'):
            attn.block_representative=LearnableBlockRepresentative(attn.head_dim,cfg.salience_temperature).to(device=device,dtype=torch.float32)
        attn.block_representative.set_temperature(cfg.salience_temperature)
        object.__setattr__(attn,'_shared_memory_router_ref',backbone.shared_memory_router)
        attn._pave_config=cfg
    backbone._pave_default_mode=default_mode
    return backbone.shared_memory_router


def reset_rotary_cache(model):
    # Independent requests must not inherit the previous long request's DCA mscale.
    backbone=model.model if hasattr(model,'model') else model
    for layer in backbone.layers:
        rotary=layer.self_attn.rotary_emb
        rotary.max_seq_len=rotary.max_position_embeddings
        rotary._set_cos_sin_cache(rotary.max_seq_len,rotary.inv_freq.device,torch.float32)


def trainable_state_dict(model):
    return {k:v.detach().cpu() for k,v in model.state_dict().items()
            if 'shared_memory_router' in k or 'block_representative' in k}


def install_indexed_prefill():
    """Use block indices to read the selected PAVE history during Prefill."""
    global prepare_selected_kv, inter_attention_microchunks
    from .prefill_kernel import attention

    original_prepare = prepare_selected_kv
    original_inter = inter_attention_microchunks
    state = dict(active=False, ids=None)

    def prepare(attn, k, v, kb, vb, ids, proposal, request):
        if not state['active']:
            return original_prepare(attn, k, v, kb, vb, ids, proposal, request)
        if attn.training or torch.is_grad_enabled():
            raise RuntimeError('Indexed Prefill is not a training path')
        assert k.shape[-2] % 128 == 0, 'DCA inter history must align to 128'
        state['ids'] = ids
        # These placeholders are never passed to dense attention or exposed
        # to the cache. Original scoring and recurrence run without packing.
        return k, v

    def inter(attn, q, k, v, raw_queries, begin, inter_position, cos, sin):
        request = _STATE.get()
        if request is None or request.mode != 'sparse':
            return original_inter(attn, q, k, v, raw_queries, begin, inter_position, cos, sin)
        cfg = attn._pave_config
        step = cfg.routing_interval
        assert step in (128, 512)
        previous = (request.current_query, request.current_group, request.planned_allocations,
                    request.pending_keep_updates, request.packed_kv_cache)
        selected = []
        try:
            request.current_group = begin
            request.packed_kv_cache = None
            request.planned_allocations = None
            request.pending_keep_updates = None
            if k.shape[-2] >= BLOCK_SIZE:
                request.planned_allocations = plan_microquery_allocations(
                    attn, raw_queries, begin, q.shape[-2], step, k.shape[-2] // 128)
            request.pending_keep_updates = []
            for offset in range(0, q.shape[-2], step):
                stop = min(offset + step, q.shape[-2])
                start = begin + offset
                request.current_query = (start, begin + stop)
                prior = raw_queries[:, :, max(0, start - cfg.query_observation_tokens):start, :]
                assert prior.shape[-2] > 0
                positions = inter_position[:, None].expand(-1, prior.shape[-2])
                selection = apply_rotary_pos_emb(prior, cos, sin, positions)
                state['ids'] = None
                state['active'] = True
                try:
                    select_inter_memory(attn, q[:, :, offset:stop], k, v,
                                             routing_query=prior, selection_query=selection)
                finally:
                    state['active'] = False
                ids = state['ids']
                assert ids is not None and ids.shape[1] > 0
                selected.append(ids[0])
            if request.pending_keep_updates:
                flush_keep_updates(request)
            width = max(x.numel() for x in selected)
            # Build compact block metadata, not a token-by-token attention mask.
            ids = torch.stack([torch.nn.functional.pad(x, (0, width-x.numel()), value=0)
                               for x in selected]).contiguous()
            counts = torch.tensor([x.numel() for x in selected], dtype=torch.int32, device=q.device)
            return attention(q, k, v, ids, counts, step)
        finally:
            state['active'] = False
            state['ids'] = None
            (request.current_query, request.current_group, request.planned_allocations,
             request.pending_keep_updates, request.packed_kv_cache) = previous

    prepare_selected_kv = prepare
    inter_attention_microchunks = inter


def install_sparse_decode(model, sparse=True):
    """Install PAVE Decode and return a reset function for request-local state."""
    global _cached_attention_values
    from .decode_kernel import attention

    active=ContextVar('pave_sparse_decode_layer',default=None)
    layers = {}

    def reset():
        layers.clear()

    def cached(qi,k,v,qs,qr,mask,dim):
        attn=active.get()
        if attn is None:
            raise RuntimeError('Missing Decode layer context')
        if attn.training or torch.is_grad_enabled():
            raise RuntimeError('Sparse Decode is inference-only; no training path')
        assert mask is None or bool(mask.eq(0).all()), 'Padding is unsupported'
        total=k.shape[-2]
        chunk=chunk_size-local_window
        group=(total-1)//chunk
        current=group*chunk
        inter=max(0,(group-1)*chunk)
        # Unused queries must still have valid addresses for the fused kernel.
        qs=qi if qs is None else qs
        qr=qi if qr is None else qr
        ids=None
        if sparse and inter>=128:
            layer=layers[attn.layer_idx]
            prior=layer['prior']
            assert prior.shape[-2]>0
            interval=attn._pave_config.routing_interval
            bucket=(current,(total-1)//interval)
            if layer.get('selection_bucket')==bucket:
                ids=layer['ids']
            else:
                ids=select(attn,qr,k,layer,prior,current,inter,bucket,interval)
                layer['ids']=ids;layer['selection_bucket']=bucket
        return attention(qi,qs,qr,k,v,ids,inter,current)

    def select(attn,qr,k,layer,prior,current,inter,bucket,interval):
        dim=k.shape[-1]
        chunk=chunk_size-local_window
        request=_STATE.get()
        request.current_query=(bucket[1]*interval,(bucket[1]+1)*interval)
        request.current_group=current
        request.planned_allocations=None
        # Propagate once across layers at a refresh. A fresh interval gets
        # its own scope, never inheriting the previous interval's decay.
        assert request.pending_keep_updates is None
        kr=k[:,:,:inter,:]
        p=make_proposal(attn,qr,kr,request,routing_query=prior)
        n=inter//128
        old_n=layer.get('n',0)
        if old_n<n:
            blocks=k[:,:,old_n*128:n*128,:].view(1,k.shape[1],n-old_n,128,dim)
            added,_,_=attn.block_representative(blocks)
            layer['reps']=torch.cat((layer['reps'],added),2) if old_n else added
            layer['n']=n
        concentration=None
        if p.remote:
            reps=layer['reps'][:,:,p.sink:p.sink+p.remote]
            pos=torch.zeros((1,prior.shape[-2]),device=k.device,dtype=torch.long)
            rotary=attn.rotary_emb
            cos=rotary.qc_cos_cached[...,chunk-1:chunk,:].to(k.dtype)
            sin=rotary.qc_sin_cached[...,chunk-1:chunk,:].to(k.dtype)
            selection=apply_rotary_pos_emb(prior,cos,sin,pos)
            probe=selection.mean(-2)
            scores=((probe.unsqueeze(2)@reps.transpose(-1,-2)).squeeze(2)/math.sqrt(dim)).mean(1)
            concentration=compute_attention_concentration(scores)
            remote=scores.topk(p.keep,-1).indices+p.sink
        else:
            remote=torch.empty((1,0),device=k.device,dtype=torch.long)
        ids=torch.cat((torch.arange(p.sink,device=k.device)[None],remote,
                       torch.arange(p.sink+p.remote,n,device=k.device)[None]),-1).sort(-1).values.contiguous()
        update_keep_rate(attn,p,request,concentration)
        return ids

    _cached_attention_values=cached
    for layer in model.model.layers:
        attn=layer.self_attn
        original_forward=attn.forward

        def capture(module,inputs,output,attn=attn):
            if not sparse:return
            length=output.shape[1]
            raw=output.view(1,length,attn.num_heads,attn.head_dim).transpose(1,2)
            entry=layers.setdefault(attn.layer_idx,{})
            if length>1:
                entry['prior']=raw[:,:,-attn._pave_config.query_observation_tokens:,:].detach().clone()
                entry['n']=0;entry.pop('reps',None)
            else:
                entry['pending']=raw.detach()
        attn.q_proj.register_forward_hook(capture)

        def wrapped(self,*args,_original=original_forward,**kwargs):
            marker=active.set(self)
            try:
                result=_original(*args,**kwargs)
                if sparse:
                    entry=layers[self.layer_idx]
                    if 'pending' in entry:
                        new=entry.pop('pending')
                        entry['prior']=torch.cat((entry['prior'],new),2)[:,:,-self._pave_config.query_observation_tokens:,:].contiguous()
                return result
            finally:
                active.reset(marker)
        attn.forward=MethodType(wrapped,attn)
    return reset
