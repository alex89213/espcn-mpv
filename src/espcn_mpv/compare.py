"""Score images against a reference, on luma, with several metrics.

Accepts single files or directories. Given directories, files are matched by
name and the reported figure is the mean over every matching frame, which is
how a clip gets scored: one number per upscaler rather than one per frame.

PSNR alone is a poor guide for an upscaler. It rewards conservative, slightly
blurry output, because blur keeps squared error low while destroying exactly
the detail the network is supposed to recover. SSIM and MS-SSIM disagree with
it often enough to be worth printing alongside.
"""

import argparse
from pathlib import Path

from .metrics import METRICS
from .metrics import compare as compare_pair
from .metrics import load_luma

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

FORMATS = {"MAE": "{:.5f}", "PSNR": "{:.3f}", "SSIM": "{:.5f}", "MS-SSIM": "{:.5f}"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="espcn-mpv compare",
        description="Compare images or frame directories against a reference.",
    )
    parser.add_argument(
        "--reference", required=True, help="ground-truth image or directory"
    )
    parser.add_argument(
        "--crop",
        type=int,
        default=0,
        help="drop this many pixels from each edge before scoring. Useful when "
        "comparing implementations that pad differently: torch's Conv2d pads "
        "with zeros, while a texture sampler clamps to the edge, so the outer "
        "few pixels disagree for reasons that say nothing about the interior",
    )
    parser.add_argument("images", nargs="+", help="images or directories to score")
    return parser.parse_args(argv)


def frames(path: str | Path) -> list[Path]:
    """One path for a file, or every image inside a directory, sorted."""
    path = Path(path)
    if path.is_dir():
        found = sorted(
            p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not found:
            raise FileNotFoundError(f"no images in {path}")
        return found
    return [path]


def score(reference: list[Path], other: list[Path], crop: int) -> dict[str, float]:
    """Mean of every metric over matching frames.

    Directories are matched by filename rather than by position, so a missing
    or extra frame on one side is an error instead of a silent misalignment
    that quietly compares frame 5 against frame 6.
    """
    if len(reference) != len(other):
        raise ValueError(f"{len(reference)} reference frames vs {len(other)}")
    if len(reference) > 1:
        mismatched = [
            (r.name, o.name) for r, o in zip(reference, other) if r.name != o.name
        ]
        if mismatched:
            first = mismatched[0]
            raise ValueError(f"frame names do not line up: {first[0]} vs {first[1]}")

    totals = {name: 0.0 for name in METRICS}
    for reference_path, other_path in zip(reference, other):
        a, b = load_luma(reference_path), load_luma(other_path)
        if crop:
            a, b = a[..., crop:-crop, crop:-crop], b[..., crop:-crop, crop:-crop]
        for name, value in compare_pair(a, b).items():
            totals[name] += value
    return {name: total / len(reference) for name, total in totals.items()}


def report(rows: list[tuple[str, dict[str, float]]], count: int) -> None:
    """Print one line per entry, marking the best value in each column."""
    if not rows:
        return

    best = {}
    for name, (_, higher_is_better) in METRICS.items():
        values = [row[1][name] for row in rows]
        best[name] = max(values) if higher_is_better else min(values)

    label_width = max(len(label) for label, _ in rows)
    header = "".join(f"{name:>12}" for name in METRICS)
    suffix = f" (mean of {count} frames)" if count > 1 else ""
    print(f"{'image'.ljust(label_width)}{header}{suffix}")

    for label, values in rows:
        line = label.ljust(label_width)
        for name in METRICS:
            text = FORMATS[name].format(values[name])
            line += f"{text:>11}" + ("*" if values[name] == best[name] else " ")
        print(line)


def main() -> None:
    args = parse_args()

    reference = frames(args.reference)
    print(f"reference: {args.reference}")

    rows = []
    for target in args.images:
        label = Path(target).name
        try:
            rows.append((label, score(reference, frames(target), args.crop)))
        except (ValueError, FileNotFoundError) as error:
            print(f"{label}: {error}")

    report(rows, len(reference))


if __name__ == "__main__":
    main()
