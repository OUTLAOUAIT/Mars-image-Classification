"""Command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workflows import evaluate, predict, prepare, train


def _config(path: str, overrides: dict) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    config.update({key: value for key, value in overrides.items() if value is not None})
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mars-classify")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="validate the extracted Zenodo archive")
    prep.add_argument("--data-dir", default="data/hirise-map-proj-v3_2")
    prep.add_argument("--include-other", action="store_true")
    fitting = sub.add_parser("train", help="train and select a checkpoint by validation accuracy")
    fitting.add_argument("--config", default="configs/default.json")
    for arg, typ in (("--data-dir", str), ("--model", str), ("--image-size", int), ("--batch-size", int), ("--epochs", int), ("--learning-rate", float), ("--seed", int), ("--device", str), ("--output-dir", str), ("--weights", str)):
        fitting.add_argument(arg, type=typ)
    ev = sub.add_parser("evaluate", help="evaluate a checkpoint on one official split")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--data-dir", required=True)
    ev.add_argument("--split", choices=("train", "val", "test"), default="test")
    ev.add_argument("--batch-size", type=int, default=32)
    pred = sub.add_parser("predict", help="predict one image or all images in a directory")
    pred.add_argument("--checkpoint", required=True)
    pred.add_argument("--input", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.data_dir, args.include_other)
    elif args.command == "train":
        overrides = {name: getattr(args, name) for name in ("data_dir", "model", "image_size", "batch_size", "epochs", "learning_rate", "seed", "device", "output_dir", "weights")}
        if overrides["weights"] and overrides["weights"].lower() == "none":
            overrides["weights"] = None
        result = train(_config(args.config, overrides))
    elif args.command == "evaluate":
        result = evaluate(args.checkpoint, args.data_dir, args.split, args.batch_size)
    else:
        result = predict(args.checkpoint, args.input)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
