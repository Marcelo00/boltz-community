"""Export the EMA weights of a training checkpoint as a new checkpoint.

The checkpoint of a training run holds the raw weights in its state dict and the
EMA weights in the state of the EMA callback. The exported checkpoint has the EMA
weights as its state dict, and nothing from the run (optimizer, scheduler, EMA, loops),
so that it can be used as ``pretrained`` to start a new run, or for inference.
"""

import argparse
from pathlib import Path

import torch

_RUN_STATE = ("optimizer_states", "lr_schedulers", "callbacks", "ema", "loops")


def export_ema(src: Path, dst: Path) -> None:
    """Write a copy of the checkpoint `src` whose state dict is its EMA weights."""
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    ema_weights = checkpoint.get("callbacks", {}).get("EMA", {}).get("ema_weights")
    if ema_weights is None:
        msg = f"No EMA weights in {src}, was the run trained with ema: true?"
        raise ValueError(msg)

    state_dict = checkpoint["state_dict"]
    if ema_weights.keys() != state_dict.keys():
        msg = "The EMA weights and the state dict have different keys"
        raise ValueError(msg)

    difference = max(
        (ema_weights[k].float() - v.float()).abs().max().item()
        for k, v in state_dict.items()
        if v.is_floating_point()
    )
    print(f"Largest difference between the EMA and the raw weights: {difference:.3g}")

    checkpoint["state_dict"] = {
        key: ema_weights[key].to(value.dtype) for key, value in state_dict.items()
    }
    for key in _RUN_STATE:
        checkpoint.pop(key, None)
    torch.save(checkpoint, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="The checkpoint.")
    parser.add_argument("--dst", type=Path, required=True, help="The exported checkpoint.")
    args = parser.parse_args()
    export_ema(args.src, args.dst)
