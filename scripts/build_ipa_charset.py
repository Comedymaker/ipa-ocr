#!/usr/bin/env python3
"""
Extract the character set used in the IPA OCR labels file.

Example:
    python scripts/build_ipa_charset.py \
        --labels artifacts/full_dataset/labels.tsv \
        --output artifacts/full_dataset/charset.json \
        --freq artifacts/full_dataset/char_freq.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build IPA character set from labels.tsv.")
    parser.add_argument("--labels", type=Path, required=True, help="Path to labels.tsv file.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON file for the charset list.")
    parser.add_argument(
        "--freq",
        type=Path,
        default=None,
        help="Optional TSV output containing character frequencies.",
    )
    return parser.parse_args()


def load_text_column(label_path: Path) -> list[str]:
    with label_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if "text" not in reader.fieldnames:
            raise ValueError("labels.tsv missing 'text' column")
        return [row["text"] for row in reader]


def write_charset(charset: list[str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as fh:
        json.dump({"charset": charset, "size": len(charset)}, fh, ensure_ascii=False, indent=2)


def write_frequency_table(counter: Counter[str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["char", "frequency"])
        for char, freq in counter.most_common():
            writer.writerow([char, freq])


def main() -> None:
    args = parse_args()
    texts = load_text_column(args.labels)
    freq = Counter("".join(texts))
    charset = sorted(freq.keys())
    write_charset(charset, args.output)
    if args.freq:
        write_frequency_table(freq, args.freq)


if __name__ == "__main__":
    main()
