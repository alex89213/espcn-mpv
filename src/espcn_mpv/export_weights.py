"""Dump a trained checkpoint's conv weights to a flat .npz."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .model import ESPCN


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="espcn-mpv export",
        description="Export a trained ESPCN checkpoint to an .npz weight file.",
    )
    parser.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    parser.add_argument("--out", required=True, help="output .npz path")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    scale = int(checkpoint["scale"])
    d1 = int(checkpoint["d1"])
    d2 = int(checkpoint["d2"])

    model = ESPCN(scale, d1, d2)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Conv2d weight is already (out_channels, in_channels, kh, kw).
    np.savez(
        args.out,
        scale=np.array(scale),
        d1=np.array(d1),
        d2=np.array(d2),
        conv1_w=model.conv1.weight.detach().numpy(),
        conv1_b=model.conv1.bias.detach().numpy(),
        conv2_w=model.conv2.weight.detach().numpy(),
        conv2_b=model.conv2.bias.detach().numpy(),
        conv3_w=model.conv3.weight.detach().numpy(),
        conv3_b=model.conv3.bias.detach().numpy(),
    )
    print(f"wrote {args.out}  (scale={scale}, d1={d1}, d2={d2})")


if __name__ == "__main__":
    main()
