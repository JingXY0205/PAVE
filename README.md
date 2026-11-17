# PAVE

This repository accompanies the ICDE 2027 submission **PAVE: Adaptive Historical KV Management for Efficient Long-Context LLM Understanding**.

## Framework

PAVE reduces unnecessary attention over long histories by organizing historical key–value (KV) vectors into chunks and selecting which chunks to access. It adapts KV access at three levels:

- **Inter-chunk.** A learned boundary router divides history into directly accessed **Anchor** and **Recent** partitions and a **Remote** candidate partition.
- **Intra-chunk.** A semantic-aware summarizer weights each Remote chunk's original token keys by learned salience to produce a compact summary for relevance scoring.
- **Inter-layer.** PAVE ranks Remote chunks by query relevance and uses the concentration of the scores to update the next layer's retention ratio. A more concentrated distribution leads to a larger reduction.

Attention uses the original token-level KV vectors of the directly accessed and selected chunks.

## Benchmarks

| Workload | Public PAVE evaluation |
| --- | --- |
| RULER | Llama-2 Base at 8K, 16K, and 32K. The full runner uses 500 prompts per task across all 13 tasks and reports per-task and macro-average scores. [RULER](https://github.com/NVIDIA/RULER) |
| L-Eval | Llama-2 Base on TOEFL, QuALITY, Coursera, and SFiction. The runner generates zero-shot predictions for the official scorer. [L-Eval](https://github.com/OpenLMLab/LEval) |
| PG-19 and Proof-Pile-ArXiv | Llama-2 Base next-token perplexity from binary token streams or prepared JSONL windows. [PG-19](https://github.com/google-deepmind/pg19), [Proof-Pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2) |

PAVE training and RULER prompt preparation support both Base backbones. The PAVE evaluation commands use Llama-2. Baseline RULER coverage follows the backbone and length settings in each method's config.

## Repository Structure

```text
PAVE/
├── PAVE/             # Core model, attention kernels, KV cache, and inference loader
├── backbone/         # Base model download helper
├── baselines/        # RULER runners, configs, and baseline training scripts
├── dataset/
│   ├── longalpaca/   # Shared training split preparation
│   ├── ruler/        # RULER prompt generation and validation
│   ├── l_eval/       # Four closed-ended evaluation inputs
│   └── ppl/          # PG-19 and Proof-Pile-ArXiv tokenization
├── evaluation/       # Quality evaluation, ablations, and sensitivity studies
├── training/         # PAVE trainer and two backbone configs
└── requirements.txt # PAVE runtime dependencies
```

## Requirements

Run the commands from the repository root in Bash. The PAVE runtime uses Python 3.10+, CUDA-enabled PyTorch with BF16 support, Transformers 4.37.2, Triton 3.4.0, and FlashAttention 2.8.3. Install the runtime dependencies after setting up a compatible CUDA build of PyTorch.

```bash
python -m pip install packaging ninja wheel
python -m pip install -r requirements.txt --no-build-isolation
```

Use a separate data-preparation environment with a Transformers version that reads the original Llama-3 tokenizer, such as 4.55.2. Install `datasets` for LongAlpaca-12k, `numpy` and `tqdm` for PPL preparation, and the dependencies of the pinned RULER checkout for prompt generation.

```bash
DATA_PYTHON=/absolute/path/to/data-env/bin/python
LLAMA2="$PWD/models/Llama-2-7b-hf"
LLAMA3="$PWD/models/Meta-Llama-3-8B"
```

After obtaining access to both gated model repositories and authenticating the Hugging Face CLI with `hf auth login`, run the helper below to download the Base models and their tokenizers. You can also set `LLAMA2` and `LLAMA3` to existing local snapshots. [Llama-2-7B Base](https://huggingface.co/meta-llama/Llama-2-7b-hf), [Llama-3-8B Base](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

```bash
MODELS_DIR="$PWD/models" python backbone/download_models.py
```

## Data Preparation

### LongAlpaca-12k

The preparation command writes fixed train, validation, and test splits of 10,798, 600, and 600 records before token-length filtering. PAVE and the fine-tuned baselines use the same dataset revision, split seed, and checksum manifest. [LongAlpaca-12k](https://huggingface.co/datasets/Yukang/LongAlpaca-12k)

```bash
"$DATA_PYTHON" dataset/longalpaca/prepare.py --output-dir data/longalpaca
```

### RULER

Check out the pinned RULER revision and prepare its synthetic-task assets in the data-preparation environment. [RULER setup](https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a#1-download-data)

```bash
mkdir -p data
git clone https://github.com/NVIDIA/RULER.git data/RULER
git -C data/RULER checkout c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
(
  cd data/RULER/scripts/data/synthetic/json
  "$DATA_PYTHON" download_paulgraham_essay.py
  bash download_qa_dataset.sh
)

"$DATA_PYTHON" -m dataset.ruler.prepare --backbone llama2 \
  --ruler-root data/RULER --tokenizer "$LLAMA2" --output-dir data/ruler

"$DATA_PYTHON" -m dataset.ruler.prepare --backbone llama3 \
  --ruler-root data/RULER --tokenizer "$LLAMA3" --output-dir data/ruler
```

Each task writes `data/ruler/<backbone>/<length>/<task>/validation.jsonl`. The default lengths are 8K, 16K, and 32K. Use repeated `--length` options to choose specific lengths. For a 64K ablation, generate the additional prompts with `--length 65536`. Use a fresh output directory to regenerate an existing backbone and length.

### L-Eval and Perplexity

L-Eval reads the four input files in [`dataset/l_eval/`](dataset/l_eval/). Use the matching tokenizer to prepare binary token streams and their manifests for perplexity. These commands prepare the Llama-2 streams.

```bash
"$DATA_PYTHON" dataset/ppl/rawtext_to_bin.py --model llama2 --model-path "$LLAMA2" \
  --output-path data/ppl/pg19_llama2.bin

"$DATA_PYTHON" dataset/ppl/prepare_proof_pile_arxiv.py --model-path "$LLAMA2" \
  --output-path data/ppl/arxiv_llama2.bin
```

## Model Training

The LongAlpaca-12k recipe freezes the backbone and trains the boundary router and chunk summarizer with answer-only cross-entropy and salience KL. The configs use 8K, 16K, 24K, and 32K token buckets and record the position mapping, optimizer, and loss settings. Training updates use examples that reach the historical routing branch.

```bash
python -m training.train --config training/configs/llama2_7b.json \
  --model-path "$LLAMA2" --data-dir data/longalpaca --output-dir runs/pave-llama2

python -m training.train --config training/configs/llama3_8b.json \
  --model-path "$LLAMA3" --data-dir data/longalpaca --output-dir runs/pave-llama3 \
  --tokenizer-python "$DATA_PYTHON"
```

These commands train new checkpoints. The trainer records the recipe and saves the router and summarizer parameters. It selects `best.pt` by validation answer cross-entropy.

## Evaluate PAVE

The archived evaluation runners verify the Llama-2 Base snapshot and PAVE checkpoint against recorded checksums. Set `PAVE_CHECKPOINT` to your local copy of the archived PAVE checkpoint. PAVE weights are kept outside this repository. To evaluate a locally retrained Llama-2 checkpoint on RULER, use its `best.pt` and add `--allow-different-checkpoint`. The backbone checks still apply.

```bash
PAVE_CHECKPOINT=/absolute/path/to/archived/pave-llama2.pt

python -m evaluation.ruler --model-path "$LLAMA2" --checkpoint "$PAVE_CHECKPOINT" \
  --data-dir data/ruler/llama2 --output-dir runs/ruler-llama2

python -m evaluation.l_eval --model-path "$LLAMA2" --checkpoint "$PAVE_CHECKPOINT" \
  --data-dir dataset/l_eval --output-dir runs/l-eval-llama2

python -m evaluation.ppl --model-path "$LLAMA2" --checkpoint "$PAVE_CHECKPOINT" \
  --inputs data/ppl/pg19_llama2.bin --dataset pg19 --length 32768 --stride 32768 \
  --max-windows 10 --output-dir runs/ppl-pg19-llama2
```

Pass the official `auto_eval.py` path via `--scorer` to score L-Eval predictions during the run. The PPL command illustrates a 32K window configuration. For Proof-Pile-ArXiv, select `arxiv` and its token stream. For prepared JSONL windows, pass `--inputs` and omit the binary-only arguments.

### Ablations and Additional Studies

The public ablation runner compares fixed boundaries, max-pooled summaries, fixed Remote retention, and full PAVE. It uses 500 prompts per task at the selected length, either 32K or 64K.

```bash
python -m evaluation.ablation --model-path "$LLAMA2" --checkpoint "$PAVE_CHECKPOINT" \
  --data-dir data/ruler/llama2 --length 32768 --output-dir runs/ablation-32k
```

The hyperparameter runner sweeps salience temperature and sparsity decay. The design-choice runner compares pooling and retrieval policies. Both take the archived 32K JSONL windows through `--inputs` and verify their checksum. [Hyperparameter runner](evaluation/hyper.py), [design-choice runner](evaluation/design_choice.py)

## Run Baselines

Each method has a `run.py` and a `config.json` with its upstream revision or checkpoint, dependencies, and supported RULER lengths. Use the method's own environment. The example below downloads the pinned DCA source and evaluates it at three lengths with the prepared Llama-2 prompts. [Baseline runners](baselines/)

```bash
python -m baselines.dca.run download --backbone llama2 --source-dir data/upstream/dca

python -m baselines.dca.run evaluate --backbone llama2 --source-dir data/upstream/dca \
  --model-path "$LLAMA2" --data-dir data/ruler/llama2 --length 8192 16384 32768 \
  --output-dir runs/dca-llama2
```

The other runners use the same `download` and `evaluate` subcommands. Check `evaluate --help` and each config for method-specific checkpoint, tokenizer, and source arguments. DCA on Llama-3 also takes `--tokenizer-python "$DATA_PYTHON"` and reads prompts under `data/ruler/llama3`.

YaRN, LongLoRA, and Landmark Attention provide Llama-3 LongAlpaca-12k training entry points. Pass `train.jsonl` and `valid.jsonl` through `--train-file` and `--valid-file`. Evaluate the resulting `best.pt` with `--ft-checkpoint`. For YaRN and Landmark Attention, also set `--variant fine-tuned`. The archived Llama-2 LongLoRA runner uses the official 32K full fine-tuned checkpoint. The Llama-3 training script produces a LoRA+ checkpoint.

## Efficiency Protocol

The paper measures Prefill and Decode separately on an NVIDIA A100 80 GB with Llama-2-7B, BF16, batch size one, and identical 32K and 64K inputs across methods. The public runners above cover quality evaluation. See Section V of the paper for the measurement protocol.
