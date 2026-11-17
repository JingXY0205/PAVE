"""Prepare the pinned LongAlpaca-12k splits shared by PAVE and FT baselines."""

import argparse
import hashlib
import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory


DATASET_ID = "Yukang/LongAlpaca-12k"
REVISION = "ce79abba17226d5b20a0ffe2aa04e78e56c71ebb"
SEED = 42
SPLIT_ROWS = {"train": 10798, "valid": 600, "test": 600}
MANIFEST = "split_manifest.json"


def normalize_text(value):
    return "" if value is None else str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def write_split(path, records):
    digest = hashlib.sha256()
    with path.open("wb") as stream:
        for record in records:
            line = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            stream.write(line)
            digest.update(line)
    return {"file": path.name, "rows": len(records), "sha256": digest.hexdigest()}


def prepare(output_dir, *, overwrite=False):
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    names = [f"{split}.jsonl" for split in SPLIT_ROWS] + [MANIFEST]
    if not overwrite and any((output_dir / name).exists() for name in names):
        raise FileExistsError(f"LongAlpaca files already exist in {output_dir}. Use --force to replace them.")

    source = load_dataset(DATASET_ID, revision=REVISION, split="train")
    records = []
    for row in source:
        item = {key: normalize_text(row.get(key)) for key in ("instruction", "input", "output")}
        if item["instruction"] and item["output"]:
            records.append(item)
    if len(records) != sum(SPLIT_ROWS.values()):
        raise ValueError(f"Expected {sum(SPLIT_ROWS.values())} normalized rows, got {len(records)}")

    # Keep the baseline split order and the JSONL serialization byte-for-byte.
    random.Random(SEED).shuffle(records)
    test_end = SPLIT_ROWS["test"]
    valid_end = test_end + SPLIT_ROWS["valid"]
    splits = {"train": records[valid_end:], "valid": records[test_end:valid_end], "test": records[:test_end]}

    with TemporaryDirectory(dir=output_dir) as staging:
        stage = Path(staging)
        manifest = {
            "dataset_id": DATASET_ID,
            "revision": REVISION,
            "seed": SEED,
            "source_rows": len(source),
            "invalid_rows_removed": len(source) - len(records),
            "normalized_rows": len(records),
            "splits": {name: write_split(stage / f"{name}.jsonl", rows) for name, rows in splits.items()},
        }
        (stage / MANIFEST).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        for name in names:
            (stage / name).replace(output_dir / name)
    print(f"Wrote LongAlpaca-12k splits and manifest to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare(args.output_dir, overwrite=args.force)


if __name__ == "__main__":
    main()
