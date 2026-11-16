"""Inference-only, whole-block sparse single-query attention for DCA.

Adaptation of the indexed original-K/V access and online softmax design in
Microsoft MInference's block_sparse_flash_attention.py (MIT, copyright 2024
Microsoft). The original is a prefill kernel, NOT a drop-in decode kernel.
This adaptation adds single-query split-K, original-position DCA branch queries,
variable selection counts, tail masking and a reduction kernel. No token pruning.
MInference source: https://github.com/microsoft/MInference/blob/main/minference/ops/block_sparse_flash_attention.py
"""
import torch
import triton
import triton.language as tl


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


def attention(qi,qs,qr,k,v,ids,inter,current):
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
