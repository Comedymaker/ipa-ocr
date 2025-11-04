#!/usr/bin/env python3
"""
Generate a labeled IPA OCR dataset by rendering text samples to images with augmentations.

Example usage:
    python scripts/generate_ipa_dataset.py \\
        --csv data/IPA/ipa_list.csv \\
        --output-dir artifacts/ipa_dataset \\
        --samples-per-text 3 \\
        --max-samples 1000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


DEFAULT_FONTS = [
    "data/IPA/CharisSIL-Regular.ttf",
    "data/IPA/CharisSIL-Bold.ttf",
    "data/IPA/CharisSIL-Italic.ttf",
    "data/IPA/IpaPanNew.ttf",
]


@dataclass(frozen=True)
class RenderConfig:
    canvas_width: int
    canvas_height: int
    margin: int
    background: int = 255
    foreground: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate IPA OCR training images from a word list.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to ipa_list.csv source file.")
    parser.add_argument(
        "--fonts",
        type=str,
        nargs="*",
        default=DEFAULT_FONTS,
        help="List of font paths used for rendering. Defaults to bundled IPA fonts.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ipa_dataset"), help="Output base directory.")
    parser.add_argument("--canvas-width", type=int, default=512, help="Image width in pixels.")
    parser.add_argument("--canvas-height", type=int, default=128, help="Image height in pixels.")
    parser.add_argument("--margin", type=int, default=12, help="Minimum blank space around the rendered text.")
    parser.add_argument("--samples-per-text", type=int, default=3, help="How many augmented images to create per line.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional maximum number of source lines to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Write metadata.json with generation settings alongside labels.tsv.",
    )
    return parser.parse_args()


def read_ipa_lines(csv_path: Path, limit: int | None) -> List[str]:
    with csv_path.open("r", encoding="utf-8") as fh:
        rows = [line.strip() for line in fh if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    return rows


def resolve_fonts(font_paths: Sequence[str]) -> List[Path]:
    resolved = []
    for font in font_paths:
        path = Path(font)
        if not path.is_file():
            raise FileNotFoundError(f"Font not found: {path}")
        resolved.append(path)
    return resolved


def load_font(font_path: Path, size: int) -> ImageFont.FreeTypeFont:
    """Load a font, preferring RAQM layout when available."""
    layout_engines: List[int | None] = []
    layout_class = getattr(ImageFont, "Layout", None)
    if layout_class is not None:
        raqm_layout = getattr(layout_class, "RAQM", None)
        if raqm_layout is not None:
            layout_engines.append(raqm_layout)
    raqm_constant = getattr(ImageFont, "LAYOUT_RAQM", None)
    if raqm_constant is not None and raqm_constant not in layout_engines:
        layout_engines.append(raqm_constant)
    layout_engines.append(None)

    for layout_engine in layout_engines:
        kwargs = {"size": size}
        if layout_engine is not None:
            kwargs["layout_engine"] = layout_engine
        try:
            return ImageFont.truetype(str(font_path), **kwargs)
        except (AttributeError, TypeError, OSError):
            continue
    raise RuntimeError(f"Could not load font {font_path}")


def pick_font_size(text: str, font_path: Path, cfg: RenderConfig, max_size: int | None = None) -> ImageFont.FreeTypeFont:
    """Pick the largest font size that fits the canvas."""
    max_width = cfg.canvas_width - 2 * cfg.margin
    max_height = cfg.canvas_height - 2 * cfg.margin
    size_upper = max_size or min(cfg.canvas_height, cfg.canvas_width)
    size = size_upper
    while size > 4:  # 4px is effectively unreadable, so bail out at that point
        font = load_font(font_path, size)
        bbox = font.getbbox(text)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= max_width and height <= max_height:
            return font
        size -= 1
    raise ValueError(f"Unable to fit text within canvas bounds: {text}")


def render_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    cfg: RenderConfig,
) -> Image.Image:
    image = Image.new("L", (cfg.canvas_width, cfg.canvas_height), color=cfg.background)
    draw = ImageDraw.Draw(image)

    bbox = font.getbbox(text)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    pos_x = (cfg.canvas_width - width) // 2 - bbox[0]
    pos_y = (cfg.canvas_height - height) // 2 - bbox[1]

    draw.text((pos_x, pos_y), text, fill=cfg.foreground, font=font)
    return image


def random_affine(image: Image.Image) -> Image.Image:
    angle = random.uniform(-5.0, 5.0)
    shear = random.uniform(-5.0, 5.0)
    translate_x = random.uniform(-5.0, 5.0)
    translate_y = random.uniform(-5.0, 5.0)
    return image.transform(
        image.size,
        Image.AFFINE,
        (
            math.cos(math.radians(angle)),
            math.tan(math.radians(shear)),
            translate_x,
            math.tan(math.radians(-shear)),
            math.cos(math.radians(angle)),
            translate_y,
        ),
        resample=Image.BILINEAR,
        fillcolor=255,
    )


def add_noise(image: Image.Image, sigma: float = 8.0) -> Image.Image:
    np_img = np.array(image).astype(np.float32)
    noise = np.random.normal(0, sigma, np_img.shape)
    noisy = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="L")


def apply_random_augmentations(image: Image.Image) -> Image.Image:
    if random.random() < 0.6:
        image = random_affine(image)
    if random.random() < 0.5:
        sigma = random.uniform(4.0, 10.0)
        image = add_noise(image, sigma=sigma)
    if random.random() < 0.3:
        radius = random.uniform(0.5, 1.2)
        image = image.filter(ImageFilter.GaussianBlur(radius))
    return image


def ensure_output_dirs(base_dir: Path) -> Tuple[Path, Path]:
    image_dir = base_dir / "images"
    label_path = base_dir / "labels.tsv"
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir, label_path


def write_metadata(output_dir: Path, args: argparse.Namespace, samples: int, fonts: Sequence[Path]) -> None:
    metadata = {
        "source_csv": str(args.csv),
        "fonts": [str(font) for font in fonts],
        "canvas_size": [args.canvas_width, args.canvas_height],
        "margin": args.margin,
        "samples_per_text": args.samples_per_text,
        "max_samples": args.max_samples,
        "total_texts": samples,
    }
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, ensure_ascii=False, indent=2)


def generate_dataset(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    ipa_lines = read_ipa_lines(args.csv, args.max_samples)
    fonts = resolve_fonts(args.fonts)
    cfg = RenderConfig(
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        margin=args.margin,
    )

    images_dir, labels_path = ensure_output_dirs(args.output_dir)

    with labels_path.open("w", encoding="utf-8", newline="") as label_file:
        tsv_writer = csv.writer(label_file, delimiter="\t")
        tsv_writer.writerow(["image_path", "text"])

        for idx, text in enumerate(ipa_lines):
            for sample_idx in range(args.samples_per_text):
                font_path = random.choice(fonts)
                font = pick_font_size(text, font_path, cfg)
                image = render_text(text, font, cfg)
                image = apply_random_augmentations(image)

                filename = f"{idx:05d}_{sample_idx:02d}.png"
                filepath = images_dir / filename
                image.save(filepath)
                tsv_writer.writerow([str(filepath.relative_to(args.output_dir)), text])

    if args.metadata:
        write_metadata(args.output_dir, args, samples=len(ipa_lines), fonts=fonts)


def main() -> None:
    args = parse_args()
    generate_dataset(args)


if __name__ == "__main__":
    main()
