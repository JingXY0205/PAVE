#!/usr/bin/env python3
"""Prepare the Proof-Pile-2 ArXiv validation split for PPL evaluation (Llama-2 & Llama-3 compatible)."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
from transformers import AutoTokenizer


DEFAULT_REPO_ID = "EleutherAI/proof-pile-2"
DEFAULT_REVISION = "96ced71b37196dad7ddea9bd7799424fa09c9a13"
DEFAULT_SUBSET_DIR = "redpajama-arxiv"
DEFAULT_SPLIT = "validation"
DEFAULT_EXPECTED_SHARDS = 100
DEFAULT_OUTPUT_NAME = "proof_pile_arxiv.validation.bin"
DEFAULT_DTYPE = "auto"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(
        description=(
            "Download the Proof-Pile-2 ArXiv validation split, tokenize it "
            "with a local tokenizer (e.g. Llama-2 or Llama-3), and write a binary token file."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=project_root / "models" / "Llama-2-7b-hf",
        help="Path to the local Hugging Face model/tokenizer directory.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=DEFAULT_REVISION,
    )
    parser.add_argument(
        "--subset-dir",
        type=str,
        default=DEFAULT_SUBSET_DIR,
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_SPLIT,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=script_dir / DEFAULT_OUTPUT_NAME,
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "uint16", "uint32"),
        default=DEFAULT_DTYPE,
        help=(
            "Integer storage dtype for token IDs. 'auto' selects uint16 when max token ID "
            "<= 65535 (e.g., Llama-2) and uint32 otherwise (e.g., Llama-3)."
        ),
    )
    parser.add_argument(
        "--use-slow-tokenizer",
        action="store_true",
        help="Force use_fast=False when loading the tokenizer (not supported by Llama-3).",
    )
    parser.add_argument(
        "--expected-shards",
        type=int,
        default=DEFAULT_EXPECTED_SHARDS,
        help="Set to 0 to disable the shard-count check.",
    )
    parser.add_argument(
        "--document-boundary",
        choices=("eos", "none"),
        default="eos",
        help="Append one EOS token after each document or concatenate directly.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional smoke-test limit. Omit for the full validation split.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def resolve_manifest_path(
    output_path: Path,
    manifest_path: Path | None,
) -> Path:
    if manifest_path is not None:
        return manifest_path.expanduser().resolve()
    return output_path.with_suffix(output_path.suffix + ".manifest.json")


def resolve_eos_token_id(tokenizer: Any) -> int | None:
    raw_eos = getattr(tokenizer, "eos_token_id", None)
    if raw_eos is None:
        return None
    if isinstance(raw_eos, (list, tuple)):
        if not raw_eos:
            return None
        return int(raw_eos[0])
    return int(raw_eos)


def resolve_target_dtype(
    dtype_arg: str,
    tokenizer: Any,
    eos_token_id: int | None,
) -> tuple[np.dtype, int]:
    vocab = tokenizer.get_vocab()
    max_token_id = max((int(token_id) for token_id in vocab.values()), default=-1)
    if max_token_id < 0 and len(tokenizer) > 0:
        max_token_id = len(tokenizer) - 1
    if eos_token_id is not None:
        max_token_id = max(max_token_id, int(eos_token_id))

    if dtype_arg == "auto":
        target_dtype = (
            np.dtype(np.uint16)
            if max_token_id <= np.iinfo(np.uint16).max
            else np.dtype(np.uint32)
        )
    elif dtype_arg == "uint16":
        target_dtype = np.dtype(np.uint16)
    elif dtype_arg == "uint32":
        target_dtype = np.dtype(np.uint32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype_arg}")

    max_allowed = int(np.iinfo(target_dtype).max)
    if max_token_id > max_allowed:
        raise ValueError(
            f"Tokenizer max token ID ({max_token_id}) exceeds {target_dtype.name} "
            f"maximum ({max_allowed}). Use --dtype auto or --dtype uint32."
        )

    return target_dtype, max_token_id


def list_validation_shards(
    repo_id: str,
    revision: str,
    subset_dir: str,
    split: str,
) -> list[str]:
    api = HfApi()
    prefix = f"{subset_dir}/{split}/"
    files = api.list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
    )
    shards = sorted(
        path
        for path in files
        if path.startswith(prefix) and path.endswith(".jsonl.gz")
    )
    return shards


def iter_jsonl_gz(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise TypeError(
                    f"Expected a JSON object in {path} at line {line_number}."
                )
            yield record


def write_token_ids(
    handle,
    token_ids: list[int],
    hasher: hashlib._Hash,
    target_dtype: np.dtype,
) -> int:
    if not token_ids:
        return 0

    array = np.asarray(token_ids, dtype=np.int64)
    dtype_info = np.iinfo(target_dtype)
    if int(array.min()) < int(dtype_info.min) or int(array.max()) > int(dtype_info.max):
        raise ValueError(
            f"Token IDs do not fit in {target_dtype.name}. "
            f"Observed range: [{int(array.min())}, {int(array.max())}], "
            f"allowed range: [{int(dtype_info.min)}, {int(dtype_info.max)}]"
        )

    encoded = array.astype(target_dtype, copy=False).tobytes(order="C")
    handle.write(encoded)
    hasher.update(encoded)
    return int(array.size)


def main() -> None:
    args = parse_args()

    model_path = args.model_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    manifest_path = resolve_manifest_path(
        output_path,
        args.manifest_path,
    )
    cache_dir = (
        None
        if args.cache_dir is None
        else args.cache_dir.expanduser().resolve()
    )

    if not model_path.is_dir():
        fail(f"Model directory not found: {model_path}")

    if output_path.exists() and not args.force:
        fail(
            f"Output file already exists: {output_path}. "
            "Use --force to overwrite it."
        )
    if manifest_path.exists() and not args.force:
        fail(
            f"Manifest file already exists: {manifest_path}. "
            "Use --force to overwrite it."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        use_fast=not args.use_slow_tokenizer,
        local_files_only=True,
    )
    eos_token_id = resolve_eos_token_id(tokenizer)
    if args.document_boundary == "eos" and eos_token_id is None:
        fail("The tokenizer does not define eos_token_id.")

    target_dtype, max_token_id = resolve_target_dtype(
        dtype_arg=args.dtype,
        tokenizer=tokenizer,
        eos_token_id=eos_token_id,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[Source] repo={args.repo_id}, revision={args.revision}, "
        f"subset={args.subset_dir}, split={args.split}"
    )
    print(
        f"[Tokenizer] {model_path} "
        f"(class={tokenizer.__class__.__name__}, vocab_size={len(tokenizer)}, "
        f"max_id={max_token_id}, eos_id={eos_token_id})"
    )
    print(f"[Output] {output_path} (dtype={target_dtype.name})")

    shards = list_validation_shards(
        repo_id=args.repo_id,
        revision=args.revision,
        subset_dir=args.subset_dir,
        split=args.split,
    )
    if not shards:
        fail(
            "No matching .jsonl.gz shards were found under "
            f"{args.subset_dir}/{args.split}/."
        )
    if args.expected_shards > 0 and len(shards) != args.expected_shards:
        fail(
            f"Unexpected shard count: expected={args.expected_shards}, "
            f"actual={len(shards)}."
        )

    total_documents = 0
    total_tokens = 0
    skipped_documents = 0
    completed_shards = 0
    hasher = hashlib.sha256()

    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary_path.exists():
        temporary_path.unlink()

    try:
        with temporary_path.open("wb") as output_handle:
            stop = False
            for shard_name in tqdm(shards, desc="Downloading/tokenizing shards"):
                local_path = Path(
                    hf_hub_download(
                        repo_id=args.repo_id,
                        filename=shard_name,
                        repo_type="dataset",
                        revision=args.revision,
                        cache_dir=None if cache_dir is None else str(cache_dir),
                        local_files_only=args.local_files_only,
                    )
                )

                for record in iter_jsonl_gz(local_path):
                    text = record.get("text")
                    if not isinstance(text, str) or not text.strip():
                        skipped_documents += 1
                        continue

                    token_ids = tokenizer(
                        text,
                        add_special_tokens=False,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                    )["input_ids"]

                    total_tokens += write_token_ids(
                        output_handle,
                        token_ids,
                        hasher,
                        target_dtype,
                    )
                    if args.document_boundary == "eos":
                        total_tokens += write_token_ids(
                            output_handle,
                            [int(eos_token_id)],
                            hasher,
                            target_dtype,
                        )

                    total_documents += 1
                    if (
                        args.max_documents is not None
                        and total_documents >= args.max_documents
                    ):
                        stop = True
                        break

                completed_shards += 1
                if stop:
                    break

        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    expected_bytes = total_tokens * target_dtype.itemsize
    actual_bytes = output_path.stat().st_size
    if actual_bytes != expected_bytes:
        fail(
            f"Unexpected output size: expected={expected_bytes}, "
            f"actual={actual_bytes}."
        )

    manifest = {
        "dataset": "Proof-Pile-2 ArXiv",
        "repo_id": args.repo_id,
        "revision": args.revision,
        "subset_dir": args.subset_dir,
        "split": args.split,
        "source_shards_total": len(shards),
        "source_shards_processed": completed_shards,
        "model_path": str(model_path),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "tokenizer_max_token_id": max_token_id,
        "add_special_tokens": False,
        "document_boundary": args.document_boundary,
        "eos_token_id": eos_token_id,
        "documents_written": total_documents,
        "documents_skipped": skipped_documents,
        "tokens_written": total_tokens,
        "dtype": target_dtype.name,
        "output_file": output_path.name,
        "output_bytes": actual_bytes,
        "sha256": hasher.hexdigest(),
        "max_documents": args.max_documents,
    }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("[Completed]")
    print(f"  shards:    {completed_shards}/{len(shards)}")
    print(f"  documents: {total_documents}")
    print(f"  tokens:    {total_tokens}")
    print(f"  dtype:     {target_dtype.name}")
    print(f"  output:    {output_path}")
    print(f"  manifest:  {manifest_path}")
    print(f"  sha256:    {hasher.hexdigest()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
