#!/usr/bin/env python3
"""Convert pg19_raw.txt into a binary token file compatible with Llama-2, Llama-3, and Mistral."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize pg19_raw.txt and save token IDs to a binary .bin file."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=("llama2", "llama3", "mistral"),
        required=True,
        help="Model tag used in the default output filename (e.g., llama2, llama3, mistral).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the local Hugging Face model/tokenizer directory.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path(__file__).with_name("pg19_raw.txt"),
        help="Path to the raw PG19 text file (default: pg19_raw.txt).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output .bin path (default: ./pg19_{model}.validation.bin).",
    )
    parser.add_argument(
        "--dtype",
        choices=("uint32", "uint16", "auto"),
        default="uint32",
        help=(
            "Storage dtype for token IDs. Defaults to uint32 to match the original "
            "PG19 pipeline and support both Llama-2 and Llama-3."
        ),
    )
    parser.add_argument(
        "--no-add-special-tokens",
        action="store_true",
        help="Disable adding BOS/special tokens when encoding the raw text.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_arg: str, max_id: int) -> np.dtype:
    if dtype_arg == "auto":
        target_dtype = (
            np.dtype("<u2")
            if max_id <= np.iinfo(np.uint16).max
            else np.dtype("<u4")
        )
    elif dtype_arg == "uint16":
        target_dtype = np.dtype("<u2")
    else:
        target_dtype = np.dtype("<u4")

    if max_id > int(np.iinfo(target_dtype).max):
        raise ValueError(
            f"Max token ID ({max_id}) exceeds {target_dtype.name} limit "
            f"({np.iinfo(target_dtype).max}). Please use --dtype uint32 or --dtype auto."
        )
    return target_dtype


def main() -> None:
    args = parse_args()

    model_path = args.model_path.expanduser().resolve()
    input_path = args.input_path.expanduser().resolve()
    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path is not None
        else Path(f"./pg19_{args.model}.validation.bin").resolve()
    )

    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input text file not found: {input_path}")

    print(f"[Tokenizer] Loading from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    expected_bos = {"llama2": 1, "llama3": 128000}.get(args.model)
    if expected_bos is not None and tokenizer.bos_token_id != expected_bos:
        raise ValueError(f"Tokenizer BOS does not match {args.model}")

    print(f"[Input] Reading {input_path}")
    content = input_path.read_text(encoding="utf-8")
    if not content:
        raise ValueError(f"Input file is empty: {input_path}")

    add_special_tokens = not args.no_add_special_tokens

    if args.model.lower() == "mistral":
        batch_size = 100000
        batches = [
            content[i : i + batch_size] for i in range(0, len(content), batch_size)
        ]
        tokenized_batches = []
        with tqdm(total=len(batches), desc="Tokenizing (mistral)") as pbar:
            for batch in batches:
                tokenized = tokenizer(
                    batch,
                    return_tensors="np",
                    truncation=False,
                )
                tmp = tokenized["input_ids"][0][1:]
                tokenized_batches.append(tmp)
                pbar.update(1)
        ids = np.concatenate(tokenized_batches, axis=-1).astype(np.int64, copy=False)
    else:
        encoded = tokenizer(
            content,
            add_special_tokens=add_special_tokens,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        ids = np.asarray(encoded, dtype=np.int64)

    if ids.size == 0:
        raise ValueError("Tokenization produced 0 tokens.")

    min_id = int(ids.min())
    max_id = int(ids.max())
    if min_id < 0:
        raise ValueError(f"Negative token ID encountered: {min_id}")

    target_dtype = resolve_dtype(args.dtype, max_id)
    typed_ids = ids.astype(target_dtype, copy=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Write] Writing {typed_ids.size} tokens ({target_dtype.name}) to {output_path}")

    memmap_array = np.memmap(
        output_path,
        dtype=target_dtype,
        mode="w+",
        shape=(typed_ids.size,),
    )
    memmap_array[:] = typed_ids[:]
    memmap_array.flush()
    del memmap_array

    expected_bytes = int(typed_ids.size) * target_dtype.itemsize
    actual_bytes = output_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"File size mismatch: expected {expected_bytes} bytes, got {actual_bytes} bytes."
        )

    sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest = {
        "dataset": "PG19",
        "model_tag": args.model,
        "model_path": str(model_path),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "add_special_tokens": add_special_tokens,
        "tokens_written": int(typed_ids.size),
        "min_token_id": min_id,
        "max_token_id": max_id,
        "dtype": target_dtype.name,
        "byte_order": "little",
        "bos_token_id": tokenizer.bos_token_id,
        "output_file": output_path.name,
        "output_bytes": actual_bytes,
        "sha256": sha256,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("[Completed]")
    print(f"  tokens:   {typed_ids.size}")
    print(f"  id_range: [{min_id}, {max_id}]")
    print(f"  dtype:    {target_dtype.name}")
    print(f"  output:   {output_path}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()