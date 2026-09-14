"""Export only detector tensors and numeric training metadata for distribution.

Optimizer state, training arguments, paths, and logging metadata are deliberately
excluded. The output can be read by the existing evaluators with weights='ema'.
"""

import argparse
import json
from pathlib import Path

import torch


def export_checkpoint(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    ema = checkpoint.get("model_ema")
    if not isinstance(ema, dict) or not isinstance(ema.get("shadow"), dict):
        raise ValueError("An EMA checkpoint with a shadow state dictionary is required.")
    state = {}
    for name, value in ema["shadow"].items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("The detector state must contain only named tensors.")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("The detector contains non-finite parameters.")
        # Clone each tensor so no unrelated backing storage is serialized.
        state[name] = value.detach().cpu().contiguous().clone()
    if not state:
        raise ValueError("The EMA detector state is empty.")
    output = {
        "epoch": int(checkpoint["epoch"]),
        "model_ema": {
            "shadow": state,
            "decay": float(ema["decay"]),
            "updates": int(ema["updates"]),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, destination)
    restored = torch.load(destination, map_location="cpu", weights_only=True)
    actual = restored["model_ema"]["shadow"]
    if state.keys() != actual.keys() or any(
        not torch.equal(state[name], actual[name]) for name in state
    ):
        raise RuntimeError("Export verification failed.")
    return {
        "epoch": output["epoch"],
        "ema_updates": output["model_ema"]["updates"],
        "tensor_count": len(state),
        "tensor_equality": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(export_checkpoint(args.checkpoint, args.out), indent=2))


if __name__ == "__main__":
    main()
