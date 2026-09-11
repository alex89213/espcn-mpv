"""Run a trained checkpoint on an image, the reference the shader must match.

The shader is supposed to be this function compiled to GLSL. Running both on
the same input and comparing the results is what separates "the generator has
a bug" from "the model itself does not beat the scaler it is up against".
"""

import argparse
from pathlib import Path

import torch

from .metrics import load_luma, save_luma
from .model import ESPCN


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="espcn-mpv infer",
        description="Upscale an image's luma with a trained checkpoint.",
    )
    parser.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    parser.add_argument("--input", required=True, help="low-resolution image")
    parser.add_argument("--out", required=True, help="output .png path")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    scale = int(checkpoint["scale"])
    d1, d2 = int(checkpoint["d1"]), int(checkpoint["d2"])

    model = ESPCN(scale, d1, d2)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)

    lr = load_luma(args.input).to(device)
    with torch.no_grad():
        hr = model(lr).cpu()

    save_luma(hr, args.out)
    h, w = lr.shape[-2:]
    print(
        f"wrote {Path(args.out)}  ({w}x{h} -> {w * scale}x{h * scale}, "
        f"scale={scale}, d1={d1}, d2={d2})"
    )


if __name__ == "__main__":
    main()
