"""Train, evaluate, and predict workflows."""

from __future__ import annotations

import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .data import class_mapping, load_manifest, preprocess_image, source_to_index, split_samples
from .metrics import classification_metrics
from .modeling import build_model, load_checkpoint, save_metadata, _tensorflow


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(seed)


def _dataset(samples, mapping, image_size, batch_size, training=False, seed=42):
    tf = _tensorflow()
    lookup = source_to_index("other" in mapping)
    paths = [str(s.path) for s in samples]
    labels = np.array([lookup[s.source_label] for s in samples], dtype=np.int32)
    if training:
        counts = np.bincount(labels, minlength=len(mapping))
        class_weights = len(labels) / (len(mapping) * np.maximum(counts, 1))
        weights = class_weights[labels].astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels, weights))
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(path, label, weight=None):
        def read_image(p):
            return preprocess_image(p.decode("utf-8"), image_size)
        image = tf.numpy_function(read_image, [path], tf.float32)
        image.set_shape((image_size, image_size, 3))
        return (image, label, weight) if weight is not None else (image, label)

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(min(len(samples), 4096), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def prepare(data_dir: str, include_other: bool = False) -> dict:
    samples = load_manifest(data_dir, include_other)
    counts = {key: len(split_samples(samples, key)) for key in ("train", "val", "test")}
    if any(value == 0 for value in counts.values()):
        raise ValueError(f"All official splits must contain samples; got {counts}")
    return {"samples": len(samples), "splits": counts, "class_to_index": class_mapping(include_other)}


def train(config: dict) -> dict:
    seed_everything(int(config["seed"]))
    samples = load_manifest(config["data_dir"], config.get("include_other", False))
    train_rows = split_samples(samples, "train")
    val_rows = split_samples(samples, "val")
    mapping = class_mapping(config.get("include_other", False))
    train_ds = _dataset(train_rows, mapping, config["image_size"], config["batch_size"], True, config["seed"])
    val_ds = _dataset(val_rows, mapping, config["image_size"], config["batch_size"])
    if config.get("model", "efficientnet_b0") != "efficientnet_b0":
        raise ValueError("Only model='efficientnet_b0' is currently supported")
    model = build_model(len(mapping), config["image_size"], config.get("weights", "imagenet"), config.get("dropout", 0.8))
    model.compile(optimizer=_tensorflow().keras.optimizers.Adam(config["learning_rate"]), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    resolved = dict(config)
    (out / "config.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    save_metadata(out, resolved, mapping, config["image_size"])
    best = out / "best.keras"
    callbacks = [
        _tensorflow().keras.callbacks.ModelCheckpoint(best, monitor="val_accuracy", mode="max", save_best_only=True),
        _tensorflow().keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True),
        _tensorflow().keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.3, patience=2, mode="max"),
    ]
    tf = _tensorflow()
    device = config.get("device", "auto")
    device_name = "/CPU:0" if device == "cpu" else ("/GPU:0" if device in {"gpu", "cuda"} else None)
    if device in {"gpu", "cuda"} and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("GPU requested but TensorFlow reports no available GPU")
    with tf.device(device_name) if device_name else nullcontext():
        history = model.fit(train_ds, validation_data=val_ds, epochs=config["epochs"], callbacks=callbacks, verbose=1)
    history_json = {key: [float(x) for x in values] for key, values in history.history.items()}
    (out / "history.json").write_text(json.dumps(history_json, indent=2), encoding="utf-8")
    model = load_checkpoint(best)
    y_true, y_pred = _predict_samples(model, val_rows, mapping, config["image_size"], config["batch_size"])
    metrics = classification_metrics(y_true, y_pred, len(mapping))
    (out / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"checkpoint": str(best), "validation": metrics}


def _predict_samples(model, samples, mapping, image_size, batch_size):
    ds = _dataset(samples, mapping, image_size, batch_size)
    predictions = model.predict(ds, verbose=0)
    lookup = source_to_index("other" in mapping)
    labels = [lookup[row.source_label] for row in samples]
    return labels, np.argmax(predictions, axis=1).tolist()


def evaluate(checkpoint: str, data_dir: str, split: str = "test", batch_size: int = 32) -> dict:
    metadata = json.loads((Path(checkpoint).parent / "metadata.json").read_text(encoding="utf-8"))
    mapping = metadata["class_to_index"]
    samples = split_samples(load_manifest(data_dir, "other" in mapping), split)
    if not samples:
        raise ValueError(f"No samples found for split '{split}'")
    model = load_checkpoint(checkpoint)
    y_true, y_pred = _predict_samples(model, samples, mapping, metadata["image_size"], batch_size)
    result = classification_metrics(y_true, y_pred, len(mapping))
    result["split"] = split
    return result


def predict(checkpoint: str, image_path: str, batch_size: int = 32) -> list[dict]:
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint.parent / "metadata.json").read_text(encoding="utf-8"))
    mapping = metadata["class_to_index"]
    target = Path(image_path)
    files = sorted(p for p in target.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}) if target.is_dir() else [target]
    if not files or any(not p.is_file() for p in files):
        raise FileNotFoundError(f"No readable image files found at {target}")
    model = load_checkpoint(checkpoint)
    scores = []
    for start in range(0, len(files), batch_size):
        images = np.stack([preprocess_image(p, metadata["image_size"]) for p in files[start:start + batch_size]])
        scores.extend(model.predict(images, verbose=0))
    names = {index: name for name, index in mapping.items()}
    return [{"path": str(path), "label": names[int(np.argmax(row))], "scores": {names[i]: float(v) for i, v in enumerate(row)}} for path, row in zip(files, scores)]
