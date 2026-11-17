"""Landmark Attention (LCEG-512, Llama-2-7B): download assets and evaluate prepared RULER inputs."""

import argparse
from contextlib import contextmanager
import hashlib
import inspect
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


class LayerCache(list):
    def set_layer(self, index, values):
        if index == len(self):
            self.append(values)
        else:
            self[index] = values


def preserve_layer_cache(model):
    rewrite_forward(model.model, {
        'next_decoder_cache = () if use_cache else None':
            'next_decoder_cache = (past_key_values if isinstance(past_key_values, LayerCache) else LayerCache()) if use_cache else None',
        'next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)':
            'next_decoder_cache.set_layer(idx, layer_outputs[2 if output_attentions else 1])',
    }, LayerCache=LayerCache)


def greedy_generate(model, input_ids, max_tokens, **kwargs):
    from transformers import GenerationConfig

    generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_tokens,
                                  bos_token_id=1, eos_token_id=2, pad_token_id=0, use_cache=True)
    tokens = model.generate(input_ids=input_ids, attention_mask=input_ids.new_ones(input_ids.shape),
                            generation_config=generation, **kwargs)
    return tokens[0, input_ids.shape[1]:].tolist()


def load_generator(args, config):
    if config.get('backbone') == 'llama3':
        return load_llama3_landmark_generator(args, config)
    import torch
    from models.llama_landmark.llama_mem import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(
        args.model_path, local_files_only=True, torch_dtype=torch.bfloat16, use_safetensors=False,
    ).to(args.device).eval()
    settings = config['attention']
    model.set_mem_id(settings['mem_id'])
    preserve_layer_cache(model)
    # The upstream forward projects logits twice in its pretraining_tp == 1 path.
    original = textwrap.dedent(inspect.getsource(model.forward))
    projection = '        logits = self.lm_head(hidden_states)\n'
    if original.count('logits = self.lm_head(hidden_states)') != 2:
        raise RuntimeError('Unexpected Landmark logits implementation')
    rewrite_forward(model, {projection: ''})
    model.lm_head.register_forward_pre_hook(final_position)

    def generate(input_ids, max_tokens, _length):
        for layer in model.model.layers:
            rotary = layer.self_attn.rotary_emb
            layer.self_attn.rotary_emb = type(rotary)(
                model.config.hidden_size // model.config.num_attention_heads,
                max_position_embeddings=model.config.max_position_embeddings,
            ).to(input_ids.device)
        return greedy_generate(model, input_ids, max_tokens, mem_freq=settings['mem_freq'],
                               use_flash=settings['use_flash'], cache_top_k=settings['cache_top_k'],
                               max_chunk_length=settings['internal_chunk_length'],
                               offload_cache_to_cpu=settings['offload_cache_to_cpu'])

    return generate


def build_llama3_landmark(args, config):
    import math
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoModelForCausalLM

    # Grouped attention and sparse cache from the paired GQA + LoRA pilot.
    # Landmark attention semantics originate from epfml/landmark-attention (Apache-2.0).

    def weights(scores, start, block):
        # scores [batch, query_heads, query_chunk, prefix_keys]
        nq,nk=scores.shape[-2:];dev=scores.device
        q=torch.arange(start,start+nq,device=dev)[:,None]
        k=torch.arange(nk,device=dev)[None,:]
        kb=k//block;qb=q//block;mem=k%block==block-1
        local=(kb==qb)&~mem&(k<=q)
        remote_mem=mem&(kb<qb)
        valid=(k<=q)&~(mem&(kb==qb))
        # Dynamic group count; unlike upstream's fixed 64 this handles full samples.
        global_id=(nk+block-1)//block
        group=torch.where(local|remote_mem,global_id,kb).expand(nq,nk)
        idx=group[None,None].expand_as(scores)
        dtype=torch.float64 if scores.dtype==torch.float64 else torch.float32
        x=scores.to(dtype);masked=x.masked_fill(~valid,-torch.inf)
        maxima=torch.full((*x.shape[:-1],global_id+1),-torch.inf,device=dev,dtype=dtype)
        maxima.scatter_reduce_(-1,idx,masked.detach(),reduce='amax',include_self=True)
        mx=maxima.gather(-1,idx);mx=torch.where(torch.isfinite(mx),mx,0)
        ex=torch.exp((x-mx).masked_fill(~valid,-torch.inf))
        sums=torch.zeros_like(maxima).scatter_add(-1,idx,ex)
        prob=ex/sums.gather(-1,idx).clamp_min(torch.finfo(dtype).tiny)
        gates=torch.zeros_like(maxima).scatter_add(-1,kb.expand(nq,nk)[None,None].expand_as(x),prob*remote_mem)
        return torch.where(local,prob,torch.where(remote_mem,0,prob*gates.gather(-1,kb.expand(nq,nk)[None,None].expand_as(x))))


    def tile(q,k,v,start,block):
        s=(q@k.transpose(-1,-2))/math.sqrt(q.shape[-1])
        return weights(s,start,block).to(v.dtype)@v


    class LowRank(nn.Module):
        def __init__(self,base,rank=8,alpha=16):
            super().__init__();self.base=base;self.scale=alpha/rank
            self.A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
            self.B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
            nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
        def forward(self,x):
            return self.base(x)+(F.linear(F.linear(x.float(),self.A),self.B)*self.scale).to(x.dtype)


    def rotate(x):
        a,b=x.chunk(2,-1);return torch.cat((-b,a),-1)


    class Tiled(torch.autograd.Function):
        @staticmethod
        def forward(ctx,q,k,v,block,chunk):
            assert q.shape[0]==k.shape[0]==1 and q.shape[-2]==k.shape[-2]
            assert q.shape[1]%k.shape[1]==0
            ctx.save_for_backward(q,k,v);ctx.block=block;ctx.chunk=chunk
            b,h,n,d=q.shape;g=h//k.shape[1];out=torch.empty_like(q)
            for kh in range(k.shape[1]):
                hs=slice(kh*g,(kh+1)*g)
                for start in range(0,n,chunk):
                    end=min(n,start+chunk)
                    out[:,hs,start:end]=tile(q[:,hs,start:end],k[:,kh:kh+1,:end],v[:,kh:kh+1,:end],start,block)
            return out
        @staticmethod
        def backward(ctx,grad):
            q,k,v=ctx.saved_tensors;n=q.shape[-2];g=q.shape[1]//k.shape[1]
            accum=torch.float64 if q.dtype==torch.float64 else torch.float32
            dq=torch.empty_like(q);dk=torch.zeros_like(k,dtype=accum);dv=torch.zeros_like(v,dtype=accum)
            for kh in range(k.shape[1]):
                hs=slice(kh*g,(kh+1)*g)
                for start in range(0,n,ctx.chunk):
                    end=min(n,start+ctx.chunk)
                    with torch.enable_grad():
                        x=q[:,hs,start:end].detach().requires_grad_();y=k[:,kh:kh+1,:end].detach().requires_grad_();z=v[:,kh:kh+1,:end].detach().requires_grad_()
                        o=tile(x,y,z,start,ctx.block)
                        gx,gy,gz=torch.autograd.grad(o,(x,y,z),grad[:,hs,start:end])
                    dq[:,hs,start:end]=gx;dk[:,kh:kh+1,:end]+=gy;dv[:,kh:kh+1,:end]+=gz
            return dq,dk.to(k.dtype),dv.to(v.dtype),None,None

    def landmark_attention(q,k,v,block=64,chunk=128):
        return Tiled.apply(q,k,v,block,chunk)


    def retrieve(q,k,v,position,block,topk):
        # One query; absolute stored RoPE positions. Per-query-head retrieval,
        # while persistent KV storage remains grouped-query (not repeated).
        assert q.shape[-2]==1
        group=q.shape[1]//k.shape[1];count=position//block;outputs=[]
        for h in range(q.shape[1]):
            kh=h//group;qq=q[:,h:h+1]
            kk=k[:,kh:kh+1];vv=v[:,kh:kh+1]
            mem=torch.arange(count,device=q.device)*block+block-1
            if len(mem):
                rank=(qq@kk[:,:,mem].transpose(-1,-2)).float()/math.sqrt(q.shape[-1])
                if topk is not None and len(mem)>topk:
                    chosen=rank[0,0,0].topk(topk).indices.sort().values
                    mem=mem[chosen]
            local=torch.arange(count*block,min(position+1,count*block+block-1),device=q.device)
            top=torch.cat((mem,local))
            gate=((qq@kk[:,:,top].transpose(-1,-2)).float()/math.sqrt(q.shape[-1])).softmax(-1)
            # Match training's casting of probabilities to value dtype.
            probabilities=gate[...,len(mem):].to(v.dtype)
            selected_values=vv[:,:,local]
            if len(mem):
                indices=mem[:,None]-(block-1)+torch.arange(block-1,device=q.device)[None]
                keys=kk[:,:,indices];values=vv[:,:,indices]
                scores=(qq.unsqueeze(-3)@keys.transpose(-1,-2)).squeeze(-2).float()/math.sqrt(q.shape[-1])
                inner=scores.softmax(-1)
                remote=(inner*gate[...,:len(mem)].transpose(-1,-2)).to(v.dtype)
                probabilities=torch.cat((remote.flatten(-2).unsqueeze(-2),probabilities),-1)
                selected_values=torch.cat((values.flatten(-3,-2),selected_values),-2)
            value=probabilities@selected_values
            outputs.append(value)
        return torch.cat(outputs,dim=1)


    def augment(ids,mem):
        out=[]
        for i,t in enumerate(ids):
            out.append(t)
            if (i+1)%63==0 and i+1<len(ids):out.append(mem)
        return out


    class Engine:
        def __init__(self,model,topk=4):
            self.model=model;self.topk=topk;self.reset()
        def reset(self):self.cache=[None]*len(self.model.base.model.layers);self.position=0;self.content=0
        @torch.no_grad()
        def feed(self,ids,sparse=False):
            m=self.model;c=m.base.config;device=m.memory_embedding.device
            ids=torch.tensor([ids],device=device);n=ids.shape[1];start=self.position
            x=m.base.model.embed_tokens(ids.clamp_max(m.mem_id-1))
            x=torch.where((ids==m.mem_id)[...,None],m.memory_embedding.to(x.dtype),x)
            d=c.hidden_size//c.num_attention_heads
            inv=1/(c.rope_theta**(torch.arange(0,d,2,device=device,dtype=torch.float32)/d))
            freq=torch.outer(torch.arange(start,start+n,device=device,dtype=torch.float32),inv)
            angles=torch.cat((freq,freq),-1)[None,None];cos=angles.cos().to(x.dtype);sin=angles.sin().to(x.dtype)
            for i,layer in enumerate(m.base.model.layers):
                a=layer.self_attn;z=layer.input_layernorm(x)
                q=a.q_proj(z).view(1,n,c.num_attention_heads,d).transpose(1,2)
                k=a.k_proj(z).view(1,n,c.num_key_value_heads,d).transpose(1,2)
                v=a.v_proj(z).view(1,n,c.num_key_value_heads,d).transpose(1,2)
                q=q*cos+rotate(q)*sin;k=k*cos+rotate(k)*sin
                if self.cache[i] is not None:
                    oldk,oldv=self.cache[i];k=torch.cat((oldk,k),2);v=torch.cat((oldv,v),2)
                self.cache[i]=(k,v)
                if sparse:o=retrieve(q,k,v,start,m.block,self.topk)
                else:
                    g=c.num_attention_heads//c.num_key_value_heads;pieces=[]
                    for kh in range(c.num_key_value_heads):
                        pieces.append(tile(q[:,kh*g:(kh+1)*g],k[:,kh:kh+1],v[:,kh:kh+1],start,m.block))
                    o=torch.cat(pieces,1)
                x=x+a.o_proj(o.transpose(1,2).reshape(1,n,c.hidden_size))
                x=x+layer.mlp(layer.post_attention_layernorm(x))
            self.position+=n
            return m.base.model.norm(x)
        def logits(self,h):
            normal=self.model.base.lm_head(h)
            extra=F.linear(h.float(),self.model.memory_output[None]).to(normal.dtype)
            return torch.cat((normal,extra),-1).float()
        def prefill(self,ids):
            self.reset();self.content=len(ids);expanded=augment(ids,self.model.mem_id)
            for i in range(0,len(expanded),self.model.chunk):
                h=self.feed(expanded[i:i+self.model.chunk])
            # A landmark between prompt and answer is the answer's predecessor in training.
            if self.content%63==0:h=self.feed([self.model.mem_id],sparse=True)
            return self.logits(h[:,-1:])
        def append(self,token):
            h=self.feed([token],sparse=True);self.content+=1
            if self.content%63==0:h=self.feed([self.model.mem_id],sparse=True)
            return self.logits(h[:,-1:])
        def generate(self,ids,cap,eos=128001):
            assert cap>0;logits=self.prefill(ids);result=[]
            for i in range(cap):
                # Landmark is a scheduled internal token, never a generated answer token.
                token=int(logits[...,:self.model.mem_id].argmax())
                result.append(token)
                if token==eos or i+1==cap:break
                logits=self.append(token)
            return result


    class LandmarkModel(nn.Module):
        def __init__(self, base, settings):
            super().__init__()
            self.base = base
            self.block = settings['mem_freq'] + 1
            self.chunk = settings['prefill_chunk']
            self.mem_id = base.config.vocab_size
            for parameter in base.parameters():
                parameter.requires_grad_(False)
            self.memory_embedding = nn.Parameter(base.model.embed_tokens.weight.detach().float().mean(0))
            self.memory_output = nn.Parameter(base.lm_head.weight.detach().float().mean(0))
            for layer in base.model.layers:
                for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                    projection = getattr(layer.self_attn, name)
                    setattr(layer.self_attn, name, LowRank(projection, settings['rank'], settings['alpha']))

        def hidden(self, ids):
            config = self.base.config
            dim = config.hidden_size // config.num_attention_heads
            length = ids.shape[1]
            hidden = self.base.model.embed_tokens(ids.clamp_max(self.mem_id - 1))
            hidden = torch.where((ids == self.mem_id)[..., None], self.memory_embedding.to(hidden.dtype), hidden)
            inv = 1 / config.rope_theta ** (torch.arange(0, dim, 2, device=ids.device, dtype=torch.float32) / dim)
            phase = torch.outer(torch.arange(length, device=ids.device, dtype=torch.float32), inv)
            angles = torch.cat((phase, phase), dim=-1)[None, None]
            cos, sin = angles.cos().to(hidden.dtype), angles.sin().to(hidden.dtype)

            def apply_layer(x, layer):
                attention = layer.self_attn
                batch = x.shape[0]
                normed = layer.input_layernorm(x)
                q = attention.q_proj(normed).view(batch, length, config.num_attention_heads, dim).transpose(1, 2)
                k = attention.k_proj(normed).view(batch, length, config.num_key_value_heads, dim).transpose(1, 2)
                v = attention.v_proj(normed).view(batch, length, config.num_key_value_heads, dim).transpose(1, 2)
                q, k = q * cos + rotate(q) * sin, k * cos + rotate(k) * sin
                output = landmark_attention(q, k, v, self.block, self.chunk)
                x = x + attention.o_proj(output.transpose(1, 2).reshape(batch, length, config.hidden_size))
                return x + layer.mlp(layer.post_attention_layernorm(x))

            for layer in self.base.model.layers:
                if torch.is_grad_enabled():
                    hidden = checkpoint(apply_layer, hidden, layer, use_reentrant=False)
                else:
                    hidden = apply_layer(hidden, layer)
            return self.base.model.norm(hidden)

    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation='eager', use_safetensors=True,
    ).to(args.device).eval()
    return LandmarkModel(base, config['attention']), Engine


def load_llama3_landmark_generator(args, config):
    import torch
    from torch.torch_version import TorchVersion

    if config.get('adaptation') == 'pave_protocol_fine_tune':
        if __package__:
            from .train import load_trained_state
        else:
            from train import load_trained_state

        settings = read_json(Path(__file__).with_name('config.json'))['training']
        model, Engine = build_llama3_landmark(args, config)
        load_trained_state(torch, args.ft_checkpoint, model, args.model_path, settings, 'landmark_attention')
    else:
        archive_hash = hashlib.sha256()
        with args.adapter_path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                archive_hash.update(block)
        if archive_hash.hexdigest() != config['adapters'][args.adapter_epoch]:
            raise ValueError(f'Adapter does not match the archived {args.adapter_epoch} checkpoint')
        # The archived training metadata serializes torch.__version__ as TorchVersion.
        with torch.serialization.safe_globals([TorchVersion]):
            adapter = torch.load(args.adapter_path, map_location='cpu', weights_only=True)['trainable_state']
        model, Engine = build_llama3_landmark(args, config)
        trainable = {name: param for name, param in model.named_parameters() if param.requires_grad}
        if set(trainable) != set(adapter):
            raise ValueError('Unexpected Landmark adapter parameter names')
        with torch.no_grad():
            for name, parameter in trainable.items():
                if parameter.shape != adapter[name].shape:
                    raise ValueError(f'Unexpected Landmark parameter shape for {name}')
                parameter.copy_(adapter[name])
        del adapter
    model.eval()
    engine = Engine(model, config['attention']['topk'])

    def generate(input_ids, max_tokens, _length):
        return engine.generate(input_ids[0].tolist(), max_tokens, eos=config['eos_token_id'])

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

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, local_files_only=True)
    if tokenizer.convert_tokens_to_ids('<landmark>') != config['attention']['mem_id']:
        raise ValueError('Landmark tokenizer has the wrong memory token ID')
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
    if config.get('backbone') == 'llama3' and config.get('adaptation') != 'pave_protocol_fine_tune':
        result['adapter_epoch'] = args.adapter_epoch
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
        command.add_argument('--variant', choices=('archive', 'fine-tuned'), default='archive')
    run.add_argument('--tokenizer-python', type=Path,
                     help='Python executable in the Llama-3 tokenizer environment')
    run.add_argument('--adapter-path', type=Path, help='Local in-house Llama-3 Landmark adapter')
    run.add_argument('--adapter-epoch', choices=('epoch0', 'epoch1'), default='epoch1')
    run.add_argument('--ft-checkpoint', type=Path, help='Local best.pt from this baseline training entry point')
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
        overrides = config['llama3_fine_tuned'] if args.variant == 'fine-tuned' else config['llama3']
        config = dict(config, **overrides, backbone='llama3')
        if not config['lengths']:
            run.error(next(iter(config['unavailable_lengths'].values())))
    config.pop('llama3')
    config.pop('llama3_fine_tuned')
    if config.get('source') and args.source_dir is None:
        parser.error('--source-dir is required for this backbone')
    if args.command == 'download':
        if args.backbone == 'llama3' and args.model_output is None:
            fetch.error('--model-output is required for this Llama-3 profile')
        download(args, config)
    else:
        if config.get('separate_tokenizer') and args.tokenizer_python is None:
            run.error('This Llama-3 backend requires --tokenizer-python (a Transformers 4.40.2+ environment).')
        if config.get('adaptation') == 'pave_protocol_fine_tune' and args.ft_checkpoint is None:
            run.error('--ft-checkpoint is required with this local fine-tuned profile')
        if (config.get('backbone') == 'llama3' and config.get('adaptation') != 'pave_protocol_fine_tune'
                and args.adapter_path is None):
            run.error('--adapter-path is required for the archived Llama-3 Landmark profile')
        if args.samples < 1:
            run.error('--samples must be positive')
        for length in args.length or ():
            if length not in config['lengths']:
                reason = config.get('unavailable_lengths', {}).get(str(length), 'No configured RULER profile.')
                run.error(f'{length} tokens: {reason} Supported lengths: {config["lengths"]}.')
        evaluate(args, config)


if __name__ == '__main__':
    main()
