#!/usr/bin/env python3
"""
Sample a small validation subset from the generated IPA OCR dataset.

Example:
    python scripts/create_val_subset.py \
        --source-root artifacts/full_dataset \
        --output-dir artifacts/val_subset \
        --count 3000
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a smaller validation split from the full dataset.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("artifacts/full_dataset"),
        help="Root directory of the full dataset (must contain images/ and labels.tsv).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional explicit path to labels.tsv. Defaults to <source-root>/labels.tsv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/val_subset"),
        help="Directory where the sampled subset will be written.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=2000,
        help="Number of label rows to sample. Ignored if --fraction is set.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Optional fraction of the dataset to sample (between 0 and 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def read_labels(labels_path: Path) -> List[Tuple[str, str]]:
    with labels_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader, None)
        if header is None or header[:2] != ["image_path", "text"]:
            raise ValueError("labels.tsv is missing the expected header.")
        return [(row[0], row[1]) for row in reader if row]


def sample_rows(
    rows: Sequence[Tuple[str, str]],
    count: int,
    fraction: float | None,
    seed: int,
) -> List[Tuple[str, str]]:
    if fraction is not None:
        if not (0.0 < fraction <= 1.0):
            raise ValueError("--fraction must be within (0, 1].")
        count = max(1, int(len(rows) * fraction))
    count = min(count, len(rows))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), count))
    return [rows[i] for i in indices]


def prepare_output_dir(output_dir: Path, force: bool) -> Tuple[Path, Path]:
    images_dir = output_dir / "images"
    labels_path = output_dir / "labels.tsv"
    if output_dir.exists():
        if not force:
            existing = [p for p in output_dir.iterdir()]
            if existing:
                raise FileExistsError(f"Output directory {output_dir} is not empty. Use --force to override.")
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_path


def copy_subset(
    rows: Sequence[Tuple[str, str]],
    source_root: Path,
    images_dir: Path,
    labels_path: Path,
) -> None:
    with labels_path.open("w", encoding="utf-8", newline="") as label_file:
        writer = csv.writer(label_file, delimiter="\t")
        writer.writerow(["image_path", "text"])
        for rel_path, text in rows:
            src = source_root / rel_path
            dst = images_dir / Path(rel_path).name
            if not src.is_file():
                raise FileNotFoundError(f"Missing image referenced in labels: {src}")
            shutil.copy2(src, dst)
            writer.writerow([f"images/{dst.name}", text])


def main() -> None:
    args = parse_args()
    labels_path = args.labels or (args.source_root / "labels.tsv")
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels.tsv not found: {labels_path}")
    rows = read_labels(labels_path)
    if not rows:
        raise ValueError("No rows found in labels.tsv.")
    sampled_rows = sample_rows(rows, args.count, args.fraction, args.seed)
    images_dir, subset_labels = prepare_output_dir(args.output_dir, args.force)
    copy_subset(sampled_rows, args.source_root, images_dir, subset_labels)
    print(f"Wrote {len(sampled_rows)} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
