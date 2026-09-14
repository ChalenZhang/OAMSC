#!/usr/bin/env python3
"""Combine separate YOLO train and validation YAML files into one config.

Input entries are resolved deliberately so the generated file is unambiguous.
"""

from __future__ import print_function

import argparse
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required. Run this helper in the same environment as Ultralytics."
    ) from exc


def load_yaml(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dataset_root(config, yaml_path):
    root = Path(config.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = yaml_path.parent / root
    return root.resolve()


def absolute_entries(value, root):
    if isinstance(value, list):
        return [
            str((root / item).resolve())
            if not Path(item).is_absolute()
            else str(Path(item))
            for item in value
        ]
    path = Path(value)
    return str(path if path.is_absolute() else (root / path).resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-yaml", type=Path, required=True)
    parser.add_argument("--val-yaml", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_config = load_yaml(args.train_yaml)
    val_config = load_yaml(args.val_yaml)
    if train_config.get("names") != val_config.get("names"):
        raise SystemExit(
            "Class-name mismatch between {} and {}".format(
                args.train_yaml, args.val_yaml
            )
        )

    result = {
        "path": "/",
        "train": absolute_entries(
            train_config["train"], dataset_root(train_config, args.train_yaml)
        ),
        "val": absolute_entries(
            val_config["val"], dataset_root(val_config, args.val_yaml)
        ),
        "names": train_config["names"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result, handle, sort_keys=False, allow_unicode=True)
    print("Prepared YOLO train/validation yaml: {}".format(args.out))


if __name__ == "__main__":
    main()
