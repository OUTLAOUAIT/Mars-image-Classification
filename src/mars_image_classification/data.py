"""HiRISE archive validation and image input pipelines."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = {
    0: "other",
    1: "crater",
    2: "dark_dune",
    3: "streak",
    4: "bright_dune",
    5: "impact",
    6: "edge",
    7: "spider",
}


@dataclass(frozen=True)
class Sample:
    path: Path
    source_label: int
    split: str


def find_archive_files(data_dir: str | Path) -> tuple[Path, Path]:
    """Find the image directory and official train/validation/test manifest."""
    root = Path(data_dir)
    candidates = list(root.rglob("labels-map-proj-v3_2_train_val_test.txt"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one labels-map-proj-v3_2_train_val_test.txt under {root}; "
            "download and extract the Zenodo v3.2 archive first."
        )
    manifest = candidates[0]
    image_dirs = list(manifest.parent.rglob("map-proj-v3_2"))
    image_dir = image_dirs[0] if image_dirs else manifest.parent / "map-proj-v3_2"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    return image_dir, manifest


def load_manifest(data_dir: str | Path, include_other: bool = False) -> list[Sample]:
    """Read Zenodo's official manifest and resolve images under the archive."""
    image_dir, manifest = find_archive_files(data_dir)
    samples: list[Sample] = []
    with manifest.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            fields = raw.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) != 3:
                raise ValueError(f"Expected filename, label, split at {manifest}:{line_number}")
            filename, raw_label, split = fields
            try:
                label = int(raw_label)
            except ValueError as exc:
                raise ValueError(f"Invalid class id at {manifest}:{line_number}") from exc
            split = split.lower()
            if split not in {"train", "training", "val", "validation", "test", "testing"}:
                raise ValueError(f"Unknown split '{split}' at {manifest}:{line_number}")
            if label not in CLASS_NAMES:
                raise ValueError(f"Unknown class id {label} at {manifest}:{line_number}")
            if label == 0 and not include_other:
                continue
            path = image_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Manifest image missing: {path}")
            normalized_split = {"training": "train", "validation": "val", "testing": "test"}.get(split, split)
            samples.append(Sample(path, label, normalized_split))
    if not samples:
        raise ValueError("No samples found after applying class selection")
    return samples


def class_mapping(include_other: bool = False) -> dict[str, int]:
    """Return a stable, zero-based model index mapping by source class id."""
    ids = sorted(CLASS_NAMES if include_other else (key for key in CLASS_NAMES if key != 0))
    return {CLASS_NAMES[source_id]: index for index, source_id in enumerate(ids)}


def source_to_index(include_other: bool = False) -> dict[int, int]:
    names = class_mapping(include_other)
    return {source_id: names[name] for source_id, name in CLASS_NAMES.items() if name in names}


def preprocess_image(path: str | Path, image_size: int = 227) -> np.ndarray:
    """Read grayscale pixels, resize, and replicate channels for EfficientNet."""
    with Image.open(path) as image:
        gray = image.convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(gray, dtype=np.float32)
    return np.repeat(array[..., np.newaxis], 3, axis=-1)


def split_samples(samples: Iterable[Sample], split: str) -> list[Sample]:
    return [sample for sample in samples if sample.split == split]
