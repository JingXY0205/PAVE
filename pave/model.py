"""PAVE routing, chunk summaries, and DCA-based Llama-2 attention.

DCA position handling and attention are adapted from HKUNLP/ChunkLlama,
Apache-2.0, commit 2add4d7c99d24dcc1ab03414cc602abb2e28cf2c.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Callable
import math
from types import MethodType
import torch
import triton
import triton.language as tl
from torch import nn
from torch.nn import functional as F
import transformers
from transformers.cache_utils import Cache, DynamicCache
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
            return _prefill_attention(q, k, v, ids, counts, step)
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
        return _decode_attention(qi,qs,qr,k,v,ids,inter,current)

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


# MInference-style indexed Prefill, runtime-shape version.
#
# Copyright (c) 2024 Microsoft. MIT; see the MIT license below.
# Rectangular DCA-inter attention, shared whole-block ID rows, FP32 online
# softmax and native-log LSE. Dynamic lengths/leading strides prevent a JIT
# variant for every prompt/head-allocation width. No training backward.
@triton.jit
def _indexed(Q, K, V, I, C, O, L,
             QS0, QS1: tl.constexpr, QS2: tl.constexpr,
             KS0, KS1: tl.constexpr, KS2: tl.constexpr,
             VS0, VS1: tl.constexpr, VS2: tl.constexpr,
             OS0, OS1: tl.constexpr, OS2: tl.constexpr,
             NQ, NK, D: tl.constexpr, WIDTH,
             INTERVAL: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
             SCALE: tl.constexpr):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    rows = tile * BM + tl.arange(0, BM)
    dims = tl.arange(0, D)
    cols = tl.arange(0, BN)
    decision = tile * BM // INTERVAL
    count = tl.load(C + decision)
    q = tl.load(Q + head * QS0 + rows[:, None] * QS1 + dims[None, :] * QS2,
                mask=rows[:, None] < NQ, other=0)
    maximum = tl.full((BM,), -float('inf'), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for slot in range(count):
        block = tl.load(I + decision * WIDTH + slot)
        positions = block * BN + cols
        k = tl.load(K + head * KS0 + dims[:, None] * KS2 + positions[None, :] * KS1,
                    mask=positions[None, :] < NK, other=0)
        v = tl.load(V + head * VS0 + positions[:, None] * VS1 + dims[None, :] * VS2,
                    mask=positions[:, None] < NK, other=0)
        scores = tl.dot(q, k) * SCALE
        scores = tl.where(positions[None, :] < NK, scores, -float('inf'))
        new_max = tl.maximum(maximum, tl.max(scores, 1))
        alpha = tl.exp2(maximum - new_max)
        probabilities = tl.exp2(scores - new_max[:, None])
        acc = acc * alpha[:, None]
        acc = tl.dot(probabilities.to(q.dtype), v, acc)
        denominator = denominator * alpha + tl.sum(probabilities, 1)
        maximum = new_max
    result = acc / denominator[:, None]
    tl.store(O + head * OS0 + rows[:, None] * OS1 + dims[None, :] * OS2,
             result, mask=rows[:, None] < NQ)
    tl.store(L + head * NQ + rows, (maximum + tl.log2(denominator)) * 0.6931471805599453,
             mask=rows < NQ)



def _prefill_attention(q, k, v, ids, counts, interval):
    if torch.is_grad_enabled():
        raise RuntimeError('Indexed Prefill is inference-only; no backward registered')
    assert q.shape[0] == k.shape[0] == v.shape[0] == 1
    assert q.shape[1] == k.shape[1] == v.shape[1]
    assert q.shape[-1] == k.shape[-1] == v.shape[-1] == 128
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert interval in (128, 512) and ids.is_contiguous()
    assert ids.shape[0] == triton.cdiv(q.shape[2], interval)
    assert counts.shape == (ids.shape[0],)
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:3], dtype=torch.float32, device=q.device)
    _indexed[(triton.cdiv(q.shape[2], 128), q.shape[1])](
        q, k, v, ids, counts, out, lse,
        *q.stride()[1:], *k.stride()[1:], *v.stride()[1:], *out.stride()[1:],
        NQ=q.shape[2], NK=k.shape[2], D=128, WIDTH=ids.shape[1],
        INTERVAL=interval, BM=128, BN=128,
        SCALE=math.log2(math.e) / math.sqrt(128), num_warps=8, num_stages=2)
    return out, lse


# Inference-only, whole-block sparse single-query attention for DCA.
#
# Adaptation of the indexed original-K/V access and online softmax design in
# Microsoft MInference's block_sparse_flash_attention.py (MIT, copyright 2024
# Microsoft). The original is a prefill kernel, NOT a drop-in decode kernel.
# This adaptation adds single-query split-K, original-position DCA branch queries,
# variable selection counts, tail masking and a reduction kernel. No token pruning.
# MInference source: https://github.com/microsoft/MInference/blob/main/minference/ops/block_sparse_flash_attention.py
@triton.jit
def _partials(QI, QS, QR, K, V, IDS, PO, PL,
              KH, KT, KD:tl.constexpr, VH, VT, VD:tl.constexpr,
              INTER, CURRENT, TOTAL, SELECTED, LOGICAL, PARTS,
              D:tl.constexpr, SCALE:tl.constexpr, SPARSE:tl.constexpr,
              TILE:tl.constexpr, PER_PART:tl.constexpr):
    h=tl.program_id(0)
    part=tl.program_id(1)
    dd=tl.arange(0,D)
    tt=tl.arange(0,TILE)
    qi=tl.load(QI+h*D+dd).to(tl.float32)
    qs=tl.load(QS+h*D+dd).to(tl.float32)
    qr=tl.load(QR+h*D+dd).to(tl.float32)
    m=tl.full((),-float('inf'),tl.float32)
    z=tl.full((),0.,tl.float32)
    acc=tl.zeros((D,),tl.float32)
    for offset in range(0,PER_PART,TILE):
        logical=part*PER_PART+offset+tt
        if SPARSE:
            block=tl.load(IDS+logical//128,mask=logical<SELECTED,other=0)
            physical=tl.where(logical<SELECTED,block*128+logical%128,
                              INTER+logical-SELECTED)
        else:
            physical=logical
        valid=(logical<LOGICAL)&(physical<TOTAL)
        key=tl.load(K+h*KH+physical[:,None]*KT+dd[None,:]*KD,
                    mask=valid[:,None],other=0).to(tl.float32)
        query=tl.where(physical[:,None]<INTER,qr[None,:],
                       tl.where(physical[:,None]<CURRENT,qs[None,:],qi[None,:]))
        scores=tl.sum(key*query,1)*SCALE
        scores=tl.where(valid,scores,-float('inf'))
        new_m=tl.maximum(m,tl.max(scores,0))
        # Empty final tiles have -inf scores; retain the previous state.
        safe_m=tl.where(new_m==-float('inf'),0.,new_m)
        alpha=tl.exp2((m-safe_m)*1.4426950408889634)
        p=tl.exp2((scores-safe_m)*1.4426950408889634)
        value=tl.load(V+h*VH+physical[:,None]*VT+dd[None,:]*VD,
                      mask=valid[:,None],other=0).to(tl.float32)
        acc=acc*alpha+tl.sum(p[:,None]*value,0)
        z=z*alpha+tl.sum(p,0)
        m=new_m
    out=acc/tl.maximum(z,1.e-30)
    tl.store(PO+(h*PARTS+part)*D+dd,out)
    tl.store(PL+h*PARTS+part,m+tl.log(tl.maximum(z,1.e-30)))


@triton.jit
def _merge(PO,PL,O,PARTS,D:tl.constexpr,PAD:tl.constexpr):
    h=tl.program_id(0)
    pp=tl.arange(0,PAD)
    dd=tl.arange(0,D)
    lse=tl.load(PL+h*PARTS+pp,mask=pp<PARTS,other=-float('inf'))
    weights=tl.exp2((lse-tl.max(lse,0))*1.4426950408889634)
    partial=tl.load(PO+(h*PARTS+pp[:,None])*D+dd[None,:],
                     mask=pp[:,None]<PARTS,other=0)
    result=tl.sum(partial*weights[:,None],0)/tl.sum(weights,0)
    tl.store(O+h*D+dd,result)


def _decode_attention(qi,qs,qr,k,v,ids,inter,current):
    assert qi.shape[0]==1 and qi.shape[2]==1 and qi.shape[-1]==128
    assert k.shape==v.shape and k.shape[0]==1
    assert not torch.is_grad_enabled(), 'Inference only'
    heads=k.shape[1]; total=k.shape[2]; dim=k.shape[3]
    sparse=ids is not None
    count=ids.numel()*128 if sparse else inter
    logical=count+total-inter
    parts=triton.cdiv(logical,2048)
    po=torch.empty((heads,parts,dim),device=k.device,dtype=torch.float32)
    pl=torch.empty((heads,parts),device=k.device,dtype=torch.float32)
    out=torch.empty_like(qi)
    _partials[(heads,parts)](qi,qs,qr,k,v,ids if sparse else k,po,pl,
        *k.stride()[1:],*v.stride()[1:],inter,current,total,count,logical,parts,
        dim,dim**-.5,sparse,TILE=128,PER_PART=2048,num_warps=4)
    _merge[(heads,)](po,pl,out,parts,dim,triton.next_power_of_2(parts),num_warps=4)
    return out


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


# MInference license
# MIT License
#
# Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# Apache license for the DCA adaptation and RULER metrics
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
#
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
#    1. Definitions.
#
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
#
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
#
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
#
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
#
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
#
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
#
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
#
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
#
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
#
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
#
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
#
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
#
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
#
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
#
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
#
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
#
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
#
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.
#
#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.
#
#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
#
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
#
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
#
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
#
#    END OF TERMS AND CONDITIONS
#
#    APPENDIX: How to apply the Apache License to your work.
#
#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.
#
#    Copyright [yyyy] [name of copyright owner]
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
