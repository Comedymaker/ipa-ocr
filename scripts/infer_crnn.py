#!/usr/bin/env python3
"""
Run OCR inference on IPA images with a trained CRNN model.

Example:
    python scripts/infer_crnn.py \
        --image artifacts/sample_dataset/images/00000_00.png \
        --charset artifacts/full_dataset/charset.json \
        --checkpoint artifacts/models/crnn/best.pt \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import Charset
from src.model import CRNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image IPA OCR inference.")
    parser.add_argument("--image", type=Path, required=True, help="Path to input PNG image.")
    parser.add_argument("--charset", type=Path, required=True, help="Path to charset.json.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained checkpoint (best.pt/last.pt).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use.")
    parser.add_argument("--image-width", type=int, default=384, help="Resize width expected by the model.")
    parser.add_argument("--image-height", type=int, default=128, help="Resize height expected by the model.")
    return parser.parse_args()


def load_charset(path: Path) -> Charset:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        symbols = payload.get("charset")
    else:
        symbols = payload
    if not isinstance(symbols, list):
        raise ValueError("Invalid charset.json format: expecting list under 'charset'.")
    return Charset.from_list(symbols)


def load_model(checkpoint_path: Path, charset: Charset, device: torch.device) -> CRNN:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CRNN(vocab_size=len(charset.symbols) + 1)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def preprocess(image_path: Path, width: int, height: int) -> torch.Tensor:
    image = Image.open(image_path).convert("L").resize((width, height))
    np_img = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np_img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return tensor


def decode_prediction(log_probs: torch.Tensor, charset: Charset) -> str:
    indices = log_probs.argmax(dim=-1)  # (T,)
    return charset.decode(indices.tolist())


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    charset = load_charset(args.charset)
    model = load_model(args.checkpoint, charset, device)

    tensor = preprocess(args.image, args.image_width, args.image_height).to(device)
    with torch.no_grad():
        log_probs = model(tensor).squeeze(1)  # (T, vocab)
    prediction = decode_prediction(log_probs.cpu(), charset)

    print(f"Image : {args.image}")
    print(f"Output: {prediction}")


if __name__ == "__main__":
    main()
