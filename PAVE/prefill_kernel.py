"""MInference-style indexed Prefill, runtime-shape version.

Copyright (c) 2024 Microsoft. MIT; see LICENSE-MInference.
Rectangular DCA-inter attention, shared whole-block ID rows, FP32 online
softmax and native-log LSE. Dynamic lengths/leading strides prevent a JIT
variant for every prompt/head-allocation width. No training backward.
"""
import math
import torch
import triton
import triton.language as tl


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



def attention(q, k, v, ids, counts, interval):
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
