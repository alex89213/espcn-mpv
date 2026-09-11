"""Compile exported ESPCN weights into an mpv custom shader (.glsl).

Layout follows the conventions used by the existing mpv neural upscalers
(igv's FSRCNNX, bjin's mpv-prescalers):

- Each layer's feature channels are packed four-per-RGBA-texture. A layer
  wider than 4 is split across several textures (conv1_1, conv1_2, ...),
  one //!SAVE pass each, and the next layer binds all of them. This is how
  FSRCNNX carries 8 or 16 feature maps.
- Every pass carries a //!WHEN guard so the shader does nothing unless the
  video is actually being upscaled.
- The final pass fuses conv3 with the sub-pixel shuffle and picks its output
  channel from texture coordinates rather than gl_FragCoord.
"""

import argparse
from pathlib import Path

import numpy as np

COMPONENTS = "xyzw"
PER_TEXTURE = 4

# Only run when the output is meaningfully larger than the source, matching
# FSRCNNX's threshold. RPN: OUTPUT.w / LUMA.w > 1.3 AND same for height.
WHEN_UPSCALING = "//!WHEN OUTPUT.w LUMA.w / 1.300 > OUTPUT.h LUMA.h / 1.300 > *"


def fmt(x: float) -> str:
    """Format a float as a GLSL literal with at least 8 significant digits."""
    s = f"{float(x):.8g}"
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s


def textures(prefix: str, channels: int) -> list[str]:
    """Texture names holding `channels` feature maps, four per texture."""
    count = (channels + PER_TEXTURE - 1) // PER_TEXTURE
    return [f"{prefix}_{i + 1}" for i in range(count)]


def channel_ref(names: list[str], channel: int) -> tuple[str, str]:
    """Feature channel index -> (texture name, component letter)."""
    return names[channel // PER_TEXTURE], COMPONENTS[channel % PER_TEXTURE]


def slice_bounds(channels: int, index: int) -> tuple[int, int]:
    """Channel range carried by texture `index`."""
    lo = index * PER_TEXTURE
    return lo, min(channels, lo + PER_TEXTURE)


def bias_vec4(bias: np.ndarray, lo: int, hi: int) -> str:
    values = [fmt(bias[i]) for i in range(lo, hi)]
    values += ["0.0"] * (PER_TEXTURE - len(values))
    return f"vec4({', '.join(values)})"


def sample_block(names: list[str], indent: str = "    ") -> list[str]:
    """Cache the 3x3 neighborhood of every input texture as s{t}_{ky}{kx}."""
    lines = []
    for t, tex in enumerate(names):
        for ky in range(3):
            for kx in range(3):
                lines.append(
                    f"{indent}vec4 s{t}_{ky}{kx} = {tex}_texOff("
                    f"vec2({fmt(kx - 1)}, {fmt(ky - 1)}));"
                )
    return lines


def sample_name(names: list[str], channel: int, ky: int, kx: int) -> str:
    """The cached sample expression for one input channel at one tap."""
    t, comp = channel // PER_TEXTURE, COMPONENTS[channel % PER_TEXTURE]
    return f"s{t}_{ky}{kx}.{comp}"


def gen_conv1_passes(w, b, d1: int, name: str) -> list[str]:
    """5x5 conv over the hooked luma plane, 1 -> d1 channels, ReLU."""
    names = textures("conv1", d1)
    passes = []
    for index, tex in enumerate(names):
        lo, hi = slice_bounds(d1, index)
        lines = [
            "//!HOOK LUMA",
            WHEN_UPSCALING,
            "//!BIND HOOKED",
            f"//!SAVE {tex}",
            "//!WIDTH HOOKED.w",
            "//!HEIGHT HOOKED.h",
            f"//!COMPONENTS {hi - lo}",
            f"//!DESC {name} conv1 (5x5, 1->{d1}) [{index + 1}/{len(names)}]",
            "",
            "vec4 hook() {",
            f"    vec4 res = {bias_vec4(b, lo, hi)};",
        ]
        for oc in range(lo, hi):
            comp = COMPONENTS[oc - lo]
            lines.append("")
            for ky in range(5):
                for kx in range(5):
                    weight = float(w[oc, 0, ky, kx])
                    if weight == 0.0:
                        continue
                    lines.append(
                        f"    res.{comp} += {fmt(weight)} * "
                        f"HOOKED_texOff(vec2({fmt(kx - 2)}, {fmt(ky - 2)})).x;"
                    )
        lines += ["", "    return max(res, 0.0);", "}"]
        passes.append("\n".join(lines))
    return passes


def gen_conv2_passes(w, b, d1: int, d2: int, name: str) -> list[str]:
    """3x3 conv over the conv1 textures, d1 -> d2 channels, ReLU."""
    in_names = textures("conv1", d1)
    out_names = textures("conv2", d2)
    passes = []
    for index, tex in enumerate(out_names):
        lo, hi = slice_bounds(d2, index)
        lines = ["//!HOOK LUMA", WHEN_UPSCALING]
        lines += [f"//!BIND {n}" for n in in_names]
        lines += [
            f"//!SAVE {tex}",
            f"//!WIDTH {in_names[0]}.w",
            f"//!HEIGHT {in_names[0]}.h",
            f"//!COMPONENTS {hi - lo}",
            f"//!DESC {name} conv2 (3x3, {d1}->{d2}) [{index + 1}/{len(out_names)}]",
            "",
            "vec4 hook() {",
        ]
        lines += sample_block(in_names)
        lines += ["", f"    vec4 res = {bias_vec4(b, lo, hi)};"]
        for oc in range(lo, hi):
            comp = COMPONENTS[oc - lo]
            lines.append("")
            for ky in range(3):
                for kx in range(3):
                    for ic in range(d1):
                        weight = float(w[oc, ic, ky, kx])
                        if weight == 0.0:
                            continue
                        lines.append(
                            f"    res.{comp} += {fmt(weight)} * "
                            f"{sample_name(in_names, ic, ky, kx)};"
                        )
        lines += ["", "    return max(res, 0.0);", "}"]
        passes.append("\n".join(lines))
    return passes


# The final pass fuses conv3 with the sub-pixel shuffle. Which of the
# scale*scale conv3 outputs a given output pixel needs depends on where that
# pixel falls inside its source texel, computed entirely in texture space:
#
#     fcoord = fract(conv2_pos * conv2_size)   // position within the texel
#     sub    = ivec2(fcoord * scale)           // sub-pixel row/column
#     idx    = sub.y * scale + sub.x           // nn.PixelShuffle channel order
#
# This follows FSRCNNX's aggregation pass rather than reading gl_FragCoord.
# gl_FragCoord lives in framebuffer space, whose origin convention need not
# agree with the texture space that texOff samples in, so mixing the two can
# tile or scramble the output. Taking both the taps and the sub-pixel index
# from the same texture coordinates avoids that.
#
# `base` snaps the sample position to the center of the source texel before
# the neighborhood offsets are applied, so each tap lands on exactly one texel
# whether the texture is sampled linear or nearest.
#
# Still unverified, and it needs a real GPU to settle: if mpv's texture y axis
# runs opposite to PyTorch's row order, the network runs with a vertically
# mirrored kernel. Training augments with vertical flips, so that costs a
# little accuracy rather than producing visible corruption. If the output
# looks soft in a way the PSNR numbers do not predict, flip the sign of the y
# offsets here and in gen_conv2_passes and regenerate.
def gen_final_pass(w, b, d2: int, scale: int, name: str) -> str:
    """3x3 conv (d2 -> scale*scale) fused with the pixel shuffle."""
    in_names = textures("conv2", d2)
    anchor = in_names[0]
    lines = ["//!HOOK LUMA", WHEN_UPSCALING]
    lines += [f"//!BIND {n}" for n in in_names]
    lines += [
        f"//!WIDTH {anchor}.w {scale} *",
        f"//!HEIGHT {anchor}.h {scale} *",
        "//!COMPONENTS 1",
        f"//!DESC {name} conv3 + pixel shuffle (3x3, {d2}->{scale * scale})",
        "",
        "vec4 hook() {",
        f"    vec2 fcoord = fract({anchor}_pos * {anchor}_size);",
        f"    ivec2 sub = ivec2(fcoord * {fmt(scale)});",
        f"    int idx = sub.y * {scale} + sub.x;",
        "",
    ]
    # Snap to the source texel center, then sample the 3x3 neighborhood.
    for t, tex in enumerate(in_names):
        lines.append(
            f"    vec2 base{t} = {tex}_pos + (vec2(0.5) - fcoord) * {tex}_pt;"
        )
    lines.append("")
    for t, tex in enumerate(in_names):
        for ky in range(3):
            for kx in range(3):
                lines.append(
                    f"    vec4 s{t}_{ky}{kx} = {tex}_tex(base{t} + {tex}_pt * "
                    f"vec2({fmt(kx - 1)}, {fmt(ky - 1)}));"
                )
    lines.append("")

    for oc in range(scale * scale):
        keyword = "if" if oc == 0 else "} else if"
        lines.append(f"    {keyword} (idx == {oc}) {{")
        lines.append(f"        float res = {fmt(b[oc])};")
        for ky in range(3):
            for kx in range(3):
                for ic in range(d2):
                    weight = float(w[oc, ic, ky, kx])
                    if weight == 0.0:
                        continue
                    lines.append(
                        f"        res += {fmt(weight)} * "
                        f"{sample_name(in_names, ic, ky, kx)};"
                    )
        lines.append("        return vec4(res, 0.0, 0.0, 1.0);")
    lines += ["    }", "", "    return vec4(0.0, 0.0, 0.0, 1.0);", "}"]
    return "\n".join(lines)


def gen_header(name: str, scale: int, d1: int, d2: int, out: str) -> str:
    n_tex = len(textures("conv1", d1)) + len(textures("conv2", d2))
    return "\n".join(
        [
            f"// {name}: ESPCN luma upscaler for mpv",
            f"// scale={scale}, d1={d1}, d2={d2} "
            f"({n_tex} feature textures, {n_tex + 1} passes)",
            "//",
            "// Generated by espcn-mpv (gen_glsl.py) - do not edit by hand;",
            "// retrain / re-export and regenerate instead.",
            "//",
            f"// Usage: mpv --glsl-shaders={out} video.mkv",
            "//",
            "// Intermediate feature maps are unbounded ReLU outputs, so mpv",
            "// must keep them in a float framebuffer. Use --fbo-format=rgba16f",
            "// if your mpv defaults to an 8-bit format (vo=gpu-next already",
            "// picks a float format); an 8-bit FBO clamps activations at 1.0",
            "// and the result will look wrong.",
        ]
    )


def generate(weights, name: str, out: str) -> str:
    scale = int(weights["scale"])
    d1 = int(weights["d1"])
    d2 = int(weights["d2"])

    passes = gen_conv1_passes(weights["conv1_w"], weights["conv1_b"], d1, name)
    passes += gen_conv2_passes(
        weights["conv2_w"], weights["conv2_b"], d1, d2, name
    )
    passes.append(
        gen_final_pass(weights["conv3_w"], weights["conv3_b"], d2, scale, name)
    )
    return (
        gen_header(name, scale, d1, d2, out) + "\n\n" + "\n\n".join(passes) + "\n"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="espcn-mpv gen-glsl",
        description="Generate an mpv custom shader from exported ESPCN weights.",
    )
    parser.add_argument("--weights", required=True, help="path to an .npz weight file")
    parser.add_argument("--out", required=True, help="output .glsl path")
    parser.add_argument(
        "--name", default="espcn", help="name used in the shader pass descriptions"
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    weights = np.load(args.weights)
    source = generate(weights, args.name, args.out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(source)

    d1, d2 = int(weights["d1"]), int(weights["d2"])
    n_tex = len(textures("conv1", d1)) + len(textures("conv2", d2))
    print(
        f"wrote {args.out}  (scale={int(weights['scale'])}, d1={d1}, d2={d2}, "
        f"{n_tex + 1} passes)"
    )


if __name__ == "__main__":
    main()
