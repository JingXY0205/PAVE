#!/usr/bin/env python3
"""Download the base Llama 2 7B and Llama 3 8B models."""

import json
import os
import subprocess
import sys
from pathlib import Path


MODELS = (
    ("Llama-2-7B Base", "meta-llama/Llama-2-7b-hf", "Llama-2-7b-hf"),
    ("Llama-3-8B Base", "meta-llama/Meta-Llama-3-8B", "Meta-Llama-3-8B"),
)


def check_download(model_dir: Path) -> None:
    """Check required Transformers files and all indexed safetensors shards."""
    if not (model_dir / "config.json").is_file():
        raise RuntimeError("Missing config.json")

    if not any((model_dir / name).is_file() for name in ("tokenizer.model", "tokenizer.json")):
        raise RuntimeError("Missing tokenizer.model / tokenizer.json")

    index_file = model_dir / "model.safetensors.index.json"
    if index_file.is_file():
        with index_file.open(encoding="utf-8") as f:
            index = json.load(f)
        shard_names = set(index.get("weight_map", {}).values())
        if not shard_names:
            raise RuntimeError("Safetensors index contains no weight files")
        missing = [name for name in sorted(shard_names) if not (model_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"Missing weight shards: {', '.join(missing)}")
    elif not any(model_dir.glob("*.safetensors")):
        raise RuntimeError("Missing model safetensors weights")


def download_model(name: str, repo_id: str, model_dir: Path, revision: str) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "hf", "download", repo_id,
        "--local-dir", str(model_dir),
        # Both repos provide safetensors. Skip duplicate .bin weights and
        # original-format checkpoints, which are not needed by Transformers.
        "--exclude", "*.bin",
        "--exclude", "original/*",
        "--exclude", "*.pth",
    ]
    if revision:
        command.extend(["--revision", revision])

    # HF_TOKEN is picked up automatically from the environment by the CLI.
    # Do not put it on the command line or print it to the terminal.
    print(f"\n{'=' * 60}", flush=True)
    print(f"Model:       {name}", flush=True)
    print(f"Repository:  {repo_id}", flush=True)
    print(f"Destination: {model_dir}", flush=True)
    print(f"{'=' * 60}", flush=True)

    subprocess.run(command, check=True)
    check_download(model_dir)
    print(f"[OK] {name} -> {model_dir}", flush=True)


def main() -> None:
    base_dir = Path(os.environ.get("MODELS_DIR", str(Path(__file__).resolve().parent)))
    revision = os.environ.get("REVISION", "")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

    try:
        subprocess.run(
            ["hf", "--help"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Error: Hugging Face CLI unavailable. Install: pip install -U huggingface_hub", file=sys.stderr)
        sys.exit(1)

    failed = []
    for name, repo_id, folder_name in MODELS:
        try:
            download_model(name, repo_id, base_dir / folder_name, revision)
        except (subprocess.CalledProcessError, OSError, RuntimeError, ValueError) as exc:
            print(f"[FAILED] {name}: {exc}", file=sys.stderr, flush=True)
            failed.append(name)
            # Continue with the other model even if one download fails.

    if failed:
        print(f"\nDownload failed for: {', '.join(failed)}", file=sys.stderr)
        print("Check that your Hugging Face account has access to both gated model repositories.", file=sys.stderr)
        sys.exit(1)

    print("\nBoth base models downloaded and checked successfully.")


if __name__ == "__main__":
    main()
