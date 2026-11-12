# PAVE

This repository accompanies the ICDE 2027 submission **PAVE: Adaptive Historical KV Management for Efficient Long-Context LLM Understanding**.

## Overview

PAVE reduces unnecessary attention over long histories by organizing historical key–value (KV) vectors into chunks and selecting which chunks to access. It adapts KV access at three levels:

- **Inter-chunk.** A learned boundary router divides history into directly accessed **Anchor** and **Recent** partitions and a **Remote** candidate partition.
- **Intra-chunk.** A semantic-aware summarizer weights each Remote chunk's original token keys by learned salience to produce a compact summary for relevance scoring.
- **Inter-layer.** PAVE ranks Remote chunks by query relevance and uses the concentration of the scores to update the next layer's retention ratio. A more concentrated distribution leads to a larger reduction.

Attention uses the original token-level KV vectors of the directly accessed and selected chunks.

## Reproducing the experiments

The experiments use **PyTorch**, **Hugging Face Transformers**, and **FlashAttention 2** on an **NVIDIA A100 80 GB** GPU. Use [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) or [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) with its original tokenizer.

To retrain PAVE, freeze the pretrained backbone and train the boundary router and chunk summarizer on [LongAlpaca](https://huggingface.co/datasets/Yukang/LongAlpaca-12k). Group the training examples into 8K, 16K, 24K, and 32K token buckets. The training objective combines next-token loss on reference answers with KL-based salience regularization.

| Workload | Backbone | Evaluation |
| --- | --- | --- |
| [RULER](https://github.com/NVIDIA/RULER) | Llama-2-7B and Llama-3-8B | Evaluate all 13 tasks at context lengths up to 128K tokens and report the macro-average of the official task scores. |
| [L-Eval](https://github.com/OpenLMLab/LEval) | Llama-2-7B | Report zero-shot accuracy on TOEFL, QuALITY, Coursera, and SFiction using the same prompts and answer extraction across methods. |
| [PG-19](https://github.com/google-deepmind/pg19) | Llama-2-7B and Llama-3-8B | Report next-token perplexity at context lengths up to 100K tokens. |
| [Proof-Pile-ArXiv](https://huggingface.co/datasets/EleutherAI/proof-pile-2) | Llama-2-7B | Report next-token perplexity on the arXiv subset at context lengths up to 100K tokens. |

For the efficiency experiments, use Llama-2-7B with BF16 precision, batch size 1, and identical 32K and 64K inputs across methods. Measure prefill and decoding separately. See Section V of the paper for the full evaluation protocol.
