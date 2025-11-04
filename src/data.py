from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Charset:
    symbols: Sequence[str]
    blank_index: int
    token_to_index: Dict[str, int]
    index_to_token: Sequence[str]

    @classmethod
    def from_list(cls, symbols: Sequence[str]) -> "Charset":
        token_to_index = {token: idx for idx, token in enumerate(symbols)}
        blank_index = len(symbols)
        return cls(symbols=symbols, blank_index=blank_index, token_to_index=token_to_index, index_to_token=list(symbols))

    def encode(self, text: str) -> List[int]:
        try:
            return [self.token_to_index[ch] for ch in text]
        except KeyError as exc:
            raise ValueError(f"Character {exc.args[0]!r} not found in charset.") from exc

    def decode(self, indices: Sequence[int], collapse_repeats: bool = True) -> str:
        blank = self.blank_index
        result: List[str] = []
        previous = None
        for idx in indices:
            if idx == blank:
                previous = None
                continue
            if collapse_repeats and previous == idx:
                continue
            result.append(self.index_to_token[idx])
            previous = idx
        return "".join(result)


class IPADataset(Dataset):
    """Dataset that loads IPA OCR samples using paths from a TSV label file."""

    def __init__(
        self,
        root: Path,
        labels_path: Path,
        charset: Charset,
        image_height: int = 128,
        image_width: int = 384,
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self.root = root
        self.labels_path = labels_path
        self.entries = self._load_labels(labels_path)
        self.charset = charset
        self.image_height = image_height
        self.image_width = image_width
        self.transform = transform

    @staticmethod
    def _load_labels(labels_path: Path) -> List[Tuple[str, str]]:
        with labels_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            if "image_path" not in reader.fieldnames or "text" not in reader.fieldnames:
                raise ValueError("labels.tsv must contain image_path and text columns.")
            return [(row["image_path"], row["text"]) for row in reader]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Tuple[Tensor, torch.Tensor]:
        rel_path, text = self.entries[index]
        image = Image.open(self.root / rel_path).convert("L")
        if image.size != (self.image_width, self.image_height):
            image = image.resize((self.image_width, self.image_height))
        np_img = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np_img).unsqueeze(0)  # (1, H, W)
        if self.transform is not None:
            tensor = self.transform(tensor)
        target = torch.tensor(self.charset.encode(text), dtype=torch.long)
        return tensor, target


def collate_batch(batch: Iterable[Tuple[Tensor, torch.Tensor]]) -> Tuple[Tensor, torch.Tensor, torch.Tensor]:
    tensors: List[Tensor] = []
    targets: List[torch.Tensor] = []
    target_lengths: List[int] = []
    for image, encoded in batch:
        tensors.append(image)
        targets.append(encoded)
        target_lengths.append(len(encoded))
    images = torch.stack(tensors, dim=0)
    flat_targets = torch.cat(targets)
    target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long)
    return images, flat_targets, target_lengths_tensor
