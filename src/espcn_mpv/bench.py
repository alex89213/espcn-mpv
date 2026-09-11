"""Score a checkpoint over frames pulled from clips, not a single image.

The numbers in the README used to come from one frame of one clip, which is
not enough to tell a real difference from a lucky one. This walks a set of
clips, samples frames from each, builds a ground-truth pair per frame, and
reports the mean per clip and overall.

Downscaling happens in gamma light, on the encoded values, the same way
dataset.py builds training pairs. Doing it in linear light is arguably more
correct optically but produces a different kind of artifact from the one the
model was trained to invert, so it would measure the wrong thing.

What this scores is the PyTorch model, not the compiled shader. Driving mpv
once per frame is slow, and `espcn-mpv compare` has already shown the shader
tracks the model to roughly the capture path's noise floor, so the model
stands in for it here. Section 5 of the README covers that check.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import torch
from PIL import Image

from .metrics import METRICS, compare, load_luma, save_luma
from .model import ESPCN

KERNELS = {
    "lanczos": Image.LANCZOS,
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "box": Image.BOX,
}

# Baselines the model has to beat. Bicubic is the reference point the README
# quotes; lanczos is what mpv reaches for by default and is the harder target.
BASELINES = {"bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="espcn-mpv bench",
        description="Score a checkpoint over frames sampled from clips.",
    )
    parser.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    parser.add_argument("clips", nargs="+", help="video files to sample from")
    parser.add_argument(
        "--frames", type=int, default=12, help="frames to sample per clip"
    )
    parser.add_argument(
        "--kernel",
        default="lanczos",
        choices=sorted(KERNELS),
        help="kernel used to build the low-resolution side of each pair",
    )
    parser.add_argument(
        "--workdir", default="eval/bench", help="where extracted frames are written"
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="reuse frames already in workdir instead of re-extracting",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args(argv)


def probe_frame_count(clip: Path) -> int:
    """Rough frame count from duration and frame rate.

    Counting packets exactly means decoding the whole file, which is minutes
    on an uncompressed clip. Only the sampling stride depends on this, so an
    estimate is fine.
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate:format=duration",
            "-of", "default=nw=1:nk=1",
            str(clip),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rate, duration = result.stdout.split()
    numerator, denominator = rate.split("/")
    return int(float(duration) * float(numerator) / float(denominator))


def extract(clip: Path, out: Path, count: int) -> list[Path]:
    """Sample `count` frames spread across the clip as PNGs.

    `setsar=1` forces square pixels. Anamorphic sources such as the NTSC test
    clips would otherwise be measured through a display-aspect correction that
    has nothing to do with the upscaler.
    """
    out.mkdir(parents=True, exist_ok=True)
    stride = max(1, probe_frame_count(clip) // count)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(clip),
            "-vf", f"setsar=1,select=not(mod(n\\,{stride}))",
            # -fps_mode replaced -vsync, which ffmpeg 9 removed outright.
            "-fps_mode", "vfr",
            "-frames:v", str(count),
            str(out / "frame_%03d.png"),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg failed on {clip.name}:\n{result.stderr.strip()}")
    return sorted(out.glob("*.png"))


def build_pair(path: Path, scale: int, kernel: int) -> tuple[Image.Image, Image.Image]:
    """(HR luma, LR luma) for one frame, cropped to a multiple of `scale`."""
    image = Image.open(path).convert("RGB").convert("YCbCr").getchannel("Y")
    width = image.width - image.width % scale
    height = image.height - image.height % scale
    hr = image.crop((0, 0, width, height))
    lr = hr.resize((width // scale, height // scale), kernel)
    return hr, lr


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("ffmpeg and ffprobe must be on PATH")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    scale = int(checkpoint["scale"])
    d1, d2 = int(checkpoint["d1"]), int(checkpoint["d2"])
    model = ESPCN(scale, d1, d2)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)

    print(
        f"{Path(args.checkpoint).name}: scale={scale}, d1={d1}, d2={d2}, "
        f"{args.frames} frames per clip, {args.kernel} downscale"
    )

    methods = ["espcn", *BASELINES]
    running = {method: {name: 0.0 for name in METRICS} for method in methods}
    total_frames = 0

    for clip_path in args.clips:
        clip = Path(clip_path)
        root = Path(args.workdir) / clip.stem
        source = root / "source"

        if args.keep and source.is_dir() and any(source.glob("*.png")):
            extracted = sorted(source.glob("*.png"))
        else:
            extracted = extract(clip, source, args.frames)

        if not extracted:
            print(f"{clip.name}: no frames extracted, skipping")
            continue

        for directory in ("hr", "lr", *methods):
            (root / directory).mkdir(parents=True, exist_ok=True)

        totals = {method: {name: 0.0 for name in METRICS} for method in methods}
        size = ""

        for frame in extracted:
            hr, lr = build_pair(frame, scale, KERNELS[args.kernel])
            hr.save(root / "hr" / frame.name)
            lr.save(root / "lr" / frame.name)
            size = f"{hr.width}x{hr.height}"

            reference = load_luma(root / "hr" / frame.name)
            with torch.no_grad():
                upscaled = model(load_luma(root / "lr" / frame.name).to(device))
            save_luma(upscaled.cpu(), root / "espcn" / frame.name)

            for name, kernel in BASELINES.items():
                lr.resize(hr.size, kernel).save(root / name / frame.name)

            for method in methods:
                other = load_luma(root / method / frame.name)
                for metric, value in compare(reference, other).items():
                    totals[method][metric] += value
                    running[method][metric] += value

        total_frames += len(extracted)
        print(f"\n{clip.name}  ({len(extracted)} frames, {size})")
        show({m: {k: v / len(extracted) for k, v in t.items()} for m, t in totals.items()})

    if len(args.clips) > 1:
        print(f"\noverall  ({total_frames} frames, {len(args.clips)} clips)")
        show({m: {k: v / total_frames for k, v in t.items()} for m, t in running.items()})


def show(rows: dict[str, dict[str, float]]) -> None:
    formats = {"MAE": "{:.5f}", "PSNR": "{:.3f}", "SSIM": "{:.5f}", "MS-SSIM": "{:.5f}"}
    best = {
        name: (max if higher else min)(row[name] for row in rows.values())
        for name, (_, higher) in METRICS.items()
    }
    width = max(len(name) for name in rows)
    print(f"{''.ljust(width)}" + "".join(f"{name:>12}" for name in METRICS))
    for method, values in rows.items():
        line = method.ljust(width)
        for name in METRICS:
            line += f"{formats[name].format(values[name]):>11}"
            line += "*" if values[name] == best[name] else " "
        print(line)


if __name__ == "__main__":
    main()
