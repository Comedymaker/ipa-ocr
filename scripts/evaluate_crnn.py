#!/usr/bin/env python3
"""
Evaluate a trained CRNN model on the IPA OCR dataset.

Example:
    python scripts/evaluate_crnn.py \
        --dataset-root artifacts/full_dataset \
        --labels artifacts/full_dataset/labels.tsv \
        --charset artifacts/full_dataset/charset.json \
        --checkpoint artifacts/models/crnn/best.pt \
        --batch-size 32 \
        --device cuda:0 \
        --report-samples 5
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import Charset, IPADataset, collate_batch
from src.model import CRNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CRNN OCR checkpoints.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset folder containing images.")
    parser.add_argument("--labels", type=Path, required=True, help="labels.tsv path.")
    parser.add_argument("--charset", type=Path, required=True, help="charset.json path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint (best.pt or last.pt).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional subset size for quick evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sub-sampling.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    parser.add_argument("--report-samples", type=int, default=5, help="Number of decoded samples to print.")
    return parser.parse_args()


def load_charset(path: Path) -> Charset:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        symbols = payload.get("charset")
        if not isinstance(symbols, list):
            raise ValueError("Invalid charset.json format: missing list under 'charset'.")
    elif isinstance(payload, list):
        symbols = payload
    else:
        raise ValueError("Unsupported charset format.")
    return Charset.from_list(symbols)


def prepare_loader(
    dataset_root: Path,
    labels_path: Path,
    charset: Charset,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
    seed: int,
) -> DataLoader:
    dataset = IPADataset(dataset_root, labels_path, charset)
    if max_samples is not None and max_samples < len(dataset):
        rng = random.Random(seed)
        indices = rng.sample(range(len(dataset)), max_samples)
        dataset = Subset(dataset, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )


def load_model(checkpoint_path: Path, charset: Charset, device: torch.device) -> CRNN:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, bytes):
        raise ValueError("Checkpoint appears to be raw bytes. Regenerate with updated training script.")
    if "model_state" not in checkpoint:
        raise ValueError("Checkpoint missing 'model_state'.")
    model = CRNN(vocab_size=len(charset.symbols) + 1)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def greedy_decode(log_probs: Tensor, charset: Charset) -> List[str]:
    indices = log_probs.argmax(dim=-1).transpose(0, 1)  # (B, T)
    return [charset.decode(seq.tolist()) for seq in indices]


def split_targets(targets: Tensor, lengths: Tensor) -> List[Tensor]:
    offset = 0
    sequences: List[Tensor] = []
    for length in lengths.tolist():
        sequences.append(targets[offset : offset + length])
        offset += length
    return sequences


def character_error_rate(predictions: Sequence[str], references: Sequence[str]) -> float:
    total_distance = 0
    total_chars = 0
    for pred, ref in zip(predictions, references):
        total_distance += levenshtein_distance(pred, ref)
        total_chars += len(ref)
    return total_distance / max(total_chars, 1)


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = prev_row[j] + 1
            replace_cost = prev_row[j - 1] + (char_a != char_b)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        prev_row = current_row
    return prev_row[-1]


@torch.no_grad()
def evaluate(
    model: CRNN,
    loader: DataLoader,
    charset: Charset,
    device: torch.device,
) -> tuple[float, float, List[tuple[str, str]]]:
    criterion = nn.CTCLoss(blank=charset.blank_index, zero_infinity=True)
    total_loss = 0.0
    predictions: List[str] = []
    references: List[str] = []
    examples: List[tuple[str, str]] = []

    for images, targets, target_lengths in loader:
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        log_probs = model(images)
        input_lengths = torch.full(
            size=(log_probs.size(1),),
            fill_value=log_probs.size(0),
            dtype=torch.long,
            device=device,
        )
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        total_loss += loss.item() * images.size(0)
        batch_predictions = greedy_decode(log_probs.cpu(), charset)
        batch_references = [
            charset.decode(seq.tolist(), collapse_repeats=False)
            for seq in split_targets(targets.cpu(), target_lengths.cpu())
        ]
        predictions.extend(batch_predictions)
        references.extend(batch_references)
        examples.extend(zip(batch_predictions, batch_references))

    avg_loss = total_loss / len(loader.dataset)
    cer = character_error_rate(predictions, references)
    return avg_loss, cer, examples


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    charset = load_charset(args.charset)
    loader = prepare_loader(
        dataset_root=args.dataset_root,
        labels_path=args.labels,
        charset=charset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    model = load_model(args.checkpoint, charset, device)
    loss, cer, examples = evaluate(model, loader, charset, device)

    print(f"Evaluation results: loss={loss:.4f} CER={cer:.4f} on {len(loader.dataset)} samples")
    if args.report_samples > 0:
        print("\nSample predictions:")
        for pred, ref in examples[: args.report_samples]:
            print(f"  pred: {pred}\n  ref : {ref}\n")


if __name__ == "__main__":
    main()
