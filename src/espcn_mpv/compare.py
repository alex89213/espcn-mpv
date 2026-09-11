"""Score one or more images against a reference, on luma, with PSNR and SSIM.

PSNR alone is a poor guide for an upscaler. It rewards conservative, slightly
blurry output, because blur keeps squared error low while destroying exactly
the detail the network is supposed to recover. SSIM disagrees with it often
enough to be worth printing next to it.
"""

import argparse
from pathlib import Path

from .metrics import compare as compare_pair
from .metrics import load_luma


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="espcn-mpv compare",
        description="Compare images against a reference on luma (PSNR, SSIM).",
    )
    parser.add_argument("--reference", required=True, help="ground-truth image")
    parser.add_argument(
        "--crop",
        type=int,
        default=0,
        help="drop this many pixels from each edge before scoring. Useful when "
        "comparing implementations that pad differently: torch's Conv2d pads "
        "with zeros, while a texture sampler clamps to the edge, so the outer "
        "few pixels disagree for reasons that say nothing about the interior",
    )
    parser.add_argument("images", nargs="+", help="images to score")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    def read(path: str):
        image = load_luma(path)
        if args.crop:
            c = args.crop
            image = image[..., c:-c, c:-c]
        return image

    reference = read(args.reference)
    width = max(len(Path(p).name) for p in args.images)

    print(f"reference: {args.reference}")
    print(f"{'image'.ljust(width)}   {'PSNR':>9}   {'SSIM':>8}")

    best_psnr = best_ssim = None
    rows = []
    for path in args.images:
        try:
            value_psnr, value_ssim = compare_pair(reference, read(path))
        except ValueError as error:
            print(f"{Path(path).name.ljust(width)}   {error}")
            continue
        rows.append((path, value_psnr, value_ssim))
        best_psnr = value_psnr if best_psnr is None else max(best_psnr, value_psnr)
        best_ssim = value_ssim if best_ssim is None else max(best_ssim, value_ssim)

    for path, value_psnr, value_ssim in rows:
        # Mark the winner on each metric, since they do not always agree.
        mark_p = "*" if value_psnr == best_psnr else " "
        mark_s = "*" if value_ssim == best_ssim else " "
        print(
            f"{Path(path).name.ljust(width)}   "
            f"{value_psnr:8.3f}{mark_p}   {value_ssim:7.5f}{mark_s}"
        )


if __name__ == "__main__":
    main()
