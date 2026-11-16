# PAVE

Inference code and four experiment entry points for Llama-2-7B Base.

| File | Experiment |
| --- | --- |
| [full.py](full.py) | All 13 RULER tasks at 8K, 16K, and 32K with 500 examples per task and length |
| [ablation.py](ablation.py) | All 13 RULER tasks at 32K or 64K with 500 examples per task and length for each variant. Runs fixed regions, max pooling, fixed retention, and full PAVE in that order |
| [hyper.py](hyper.py) | Salience temperature and sparsity decay sweeps on 32K PG-19 and ArXiv windows |
| [design_choice.py](design_choice.py) | PG-19 PPL at 8K, 16K, and 32K for alternative chunk summaries and Remote selection rules |

The entry points share [model.py](model.py), which contains the PAVE
router, chunk summaries, and the Prefill and Decode paths.
[inference.py](inference.py) loads the weights and installs those paths.
The attention kernels and full KV cache are shared utilities. RULER scoring uses the
NVIDIA task metrics. PAVE uses DCA position handling adapted from
[ChunkLlama](https://github.com/HKUNLP/ChunkLlama) and indexed attention
adapted from [MInference](https://github.com/microsoft/MInference).

## Run from the parent of PAVE

The archived experiments use BF16 on an A800 80 GB with CUDA 12.8,
PyTorch 2.8.0, Transformers 4.37.2, Triton 3.4.0, and FlashAttention 2.8.3.
Use a FlashAttention build compatible with the installed PyTorch and CUDA.

~~~bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r PAVE/requirements.txt --no-build-isolation
~~~

Set `MODEL` to your local Llama-2-7B Base directory and `CKPT` to the
archived step110 inference checkpoint. Set `RULER_DATA` to the directory
with `<length>/<task>/validation.jsonl` files containing exactly 500
examples per file. Set `PPL_INPUTS` to the archived `paired_inputs.jsonl`
containing 10 token windows per
dataset, with fields `dataset`, `id`, `input_ids`, and `target_start`.
Set `OUT` to a writable directory for experiment outputs. Obtain all assets
separately and keep them outside the repository.

Generate the JSONL files using the
[RULER data preparation script](https://github.com/NVIDIA/RULER/blob/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/scripts/data/prepare.py)
and its synthetic data assets. Use the Llama-2 tokenizer, the base template,
and seed 42. Run the command for each task and length in the desired protocol.
Here is the 8K NIAH example. The generation budget reserves one position for
the tokenizer's beginning-of-sequence token.

~~~bash
python /path/to/RULER/scripts/data/prepare.py --benchmark synthetic \
  --task niah_single_1 --save_dir "$RULER_DATA/8192" \
  --tokenizer_path "$MODEL" --tokenizer_type hf --model_template_type base \
  --max_seq_length 8191 --num_samples 500 --random_seed 42
~~~

~~~bash
python -m PAVE.full --model-path "$MODEL" --checkpoint "$CKPT" \
  --data-dir "$RULER_DATA" --output-dir "$OUT/full"

python -m PAVE.ablation --length 32768 --model-path "$MODEL" \
  --checkpoint "$CKPT" --data-dir "$RULER_DATA" --output-dir "$OUT/ablation_32k"

python -m PAVE.hyper --model-path "$MODEL" --checkpoint "$CKPT" \
  --inputs "$PPL_INPUTS" --output-dir "$OUT/hyper"

python -m PAVE.design_choice --model-path "$MODEL" --checkpoint "$CKPT" \
  --inputs "$PPL_INPUTS" --output-dir "$OUT/design_choice"
~~~

Use `--length 65536` and a separate output directory for the other RULER
ablation. `--sweep bounds` uses the boundary settings of the PPL sweep.

The design-choice entry implements the manuscript's first-token, max-pooling,
mean-pooling, and fixed-stride alternatives. Uniform semantic and reverse-layer
selection follow the archived equal-budget protocol. Both reuse the full
model's region boundaries and the same total number of Remote accesses per
refresh, using a schedule calibrated on three separate ArXiv windows.
