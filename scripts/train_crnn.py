#!/usr/bin/env python3
"""
Train a CRNN model for IPA OCR on the generated dataset.

Example:
    python scripts/train_crnn.py \
        --dataset-root artifacts/full_dataset \
        --labels artifacts/full_dataset/labels.tsv \
        --charset artifacts/full_dataset/charset.json \
        --epochs 10 \
        --batch-size 32 \
        --output-dir artifacts/models/crnn
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset, random_split

import sys
import shutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import Charset, IPADataset, collate_batch
from src.model import CRNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CRNN model for IPA OCR.")
    parser.add_argument("--dataset-root", type=Path, default=Path("artifacts/full_dataset"), help="Dataset folder.")
    parser.add_argument("--labels", type=Path, default=Path("artifacts/full_dataset/labels.tsv"), help="labels.tsv path.")
    parser.add_argument("--charset", type=Path, default=Path("artifacts/full_dataset/charset.json"), help="Charset JSON file.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="L2 weight decay.")
    parser.add_argument("--val-split", type=float, default=0.02, help="Fraction of data for validation.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/models/crnn"), help="Directory for checkpoints.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device.")
    return parser.parse_args()


def load_charset(path: Path) -> Charset:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        symbols = payload.get("charset")
        if not isinstance(symbols, list):
            raise ValueError("Invalid charset.json format: 'charset' list missing.")
    elif isinstance(payload, list):
        symbols = payload
    else:
        raise ValueError("Unsupported charset format.")
    return Charset.from_list(symbols)


def create_dataloaders(
    dataset_root: Path,
    labels_path: Path,
    charset: Charset,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    dataset = IPADataset(dataset_root, labels_path, charset)
    total = len(dataset)
    val_count = max(1, int(total * val_split))
    train_count = total - val_count
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_count, val_count], generator=generator)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    return train_loader, val_loader


def greedy_decode(log_probs: Tensor, charset: Charset) -> List[str]:
    # log_probs: (T, B, V)
    indices = log_probs.argmax(dim=-1).transpose(0, 1)  # (B, T)
    predictions: List[str] = []
    for sequence in indices:
        predictions.append(charset.decode(sequence.tolist()))
    return predictions


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


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CTCLoss,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for images, targets, target_lengths in loader:
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad(set_to_none=True)
        log_probs = model(images)
        input_lengths = torch.full(
            size=(log_probs.size(1),),
            fill_value=log_probs.size(0),
            dtype=torch.long,
            device=device,
        )
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    charset: Charset,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    predictions: List[str] = []
    references: List[str] = []
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
        predictions.extend(greedy_decode(log_probs.cpu(), charset))
        for seq in split_targets(targets, target_lengths):
            references.append(charset.decode(seq.tolist(), collapse_repeats=False))
    cer = character_error_rate(predictions, references)
    return total_loss / len(loader.dataset), cer


def split_targets(targets: Tensor, lengths: Tensor) -> List[Tensor]:
    offset = 0
    sequences: List[Tensor] = []
    for length in lengths.tolist():
        sequences.append(targets[offset : offset + length])
        offset += length
    return sequences


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    charset = load_charset(args.charset)

    train_loader, val_loader = create_dataloaders(
        dataset_root=args.dataset_root,
        labels_path=args.labels,
        charset=charset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )

    model = CRNN(vocab_size=len(charset.symbols) + 1).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CTCLoss(blank=charset.blank_index, zero_infinity=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_cer = math.inf
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_cer = evaluate(model, val_loader, criterion, charset, device)
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_cer={val_cer:.4f}")
        checkpoint_path = args.output_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_cer": val_cer,
                "charset": charset.symbols,
            },
            checkpoint_path,
        )
        if val_cer < best_cer:
            best_cer = val_cer
            shutil.copy2(checkpoint_path, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
