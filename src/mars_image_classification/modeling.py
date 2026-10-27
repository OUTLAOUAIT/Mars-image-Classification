"""TensorFlow model construction and checkpoint utilities."""

from __future__ import annotations

import json
from pathlib import Path


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("Training and neural inference require the optional dependency: pip install -e '.[train]'") from exc
    return tf


def build_model(num_classes: int, image_size: int = 227, weights: str | None = "imagenet", dropout: float = 0.8):
    tf = _tensorflow()
    backbone = tf.keras.applications.EfficientNetB0(
        weights=weights, include_top=False, input_shape=(image_size, image_size, 3)
    )
    x = tf.keras.layers.Dropout(dropout)(backbone.output)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(backbone.input, outputs)


def load_checkpoint(checkpoint: str | Path):
    return _tensorflow().keras.models.load_model(checkpoint)


def save_metadata(output_dir: str | Path, config: dict, mapping: dict[str, int], image_size: int):
    path = Path(output_dir) / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": config, "class_to_index": mapping, "image_size": image_size}, indent=2), encoding="utf-8")
    return path
