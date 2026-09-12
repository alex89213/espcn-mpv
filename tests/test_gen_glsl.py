"""Tests for the GLSL generator.

The generator's real risk is silent index errors: a transposed kernel or an
off-by-one in the channel packing still produces a shader that compiles and
runs, just one that computes the wrong thing. `test_conv1_weights_round_trip`
is the important test here, because it parses the emitted literals back out
and checks them against the weights that went in.
"""

import re

import numpy as np
import pytest

from espcn_mpv import gen_glsl


def weights(scale: int = 2, d1: int = 8, d2: int = 8, seed: int = 0) -> dict:
    """A weight set shaped like export_weights.py's .npz output."""
    rng = np.random.default_rng(seed)
    return {
        "scale": np.array(scale),
        "d1": np.array(d1),
        "d2": np.array(d2),
        "conv1_w": rng.normal(size=(d1, 1, 5, 5)).astype(np.float32),
        "conv1_b": rng.normal(size=(d1,)).astype(np.float32),
        "conv2_w": rng.normal(size=(d2, d1, 3, 3)).astype(np.float32),
        "conv2_b": rng.normal(size=(d2,)).astype(np.float32),
        "conv3_w": rng.normal(size=(scale * scale, d2, 3, 3)).astype(np.float32),
        "conv3_b": rng.normal(size=(scale * scale,)).astype(np.float32),
    }


@pytest.mark.parametrize(
    "d1,d2,expected",
    [(4, 4, 3), (8, 8, 5), (16, 16, 9), (32, 16, 13), (5, 5, 5), (1, 1, 3)],
)
def test_pass_count(d1: int, d2: int, expected: int) -> None:
    """ceil(d1/4) + ceil(d2/4) + 1, the figure section 6 of the README quotes."""
    source = gen_glsl.generate(weights(d1=d1, d2=d2), "t", "out.glsl")
    assert source.count("//!HOOK LUMA") == expected


def test_every_pass_is_guarded() -> None:
    """A pass without //!WHEN would run even when not upscaling."""
    source = gen_glsl.generate(weights(), "t", "out.glsl")
    assert source.count("//!WHEN") == source.count("//!HOOK LUMA")


def test_wide_layers_split_across_textures() -> None:
    source = gen_glsl.generate(weights(d1=16, d2=4), "t", "out.glsl")
    for index in range(1, 5):
        assert f"//!SAVE conv1_{index}" in source
    # conv2 has to bind every conv1 texture to reassemble the activation.
    for index in range(1, 5):
        assert f"//!BIND conv1_{index}" in source


def test_final_pass_upscales_by_scale() -> None:
    source = gen_glsl.generate(weights(scale=3), "t", "out.glsl")
    assert "//!WIDTH conv2_1.w 3 *" in source
    assert "//!HEIGHT conv2_1.h 3 *" in source
    # scale*scale branches, one per sub-pixel position.
    assert source.count("if (idx == ") == 9


def test_final_pass_is_the_only_unsaved_one() -> None:
    """Exactly one pass writes to the output instead of a //!SAVE texture."""
    source = gen_glsl.generate(weights(), "t", "out.glsl")
    assert source.count("//!HOOK LUMA") - source.count("//!SAVE") == 1


@pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 1e-9, 1e12, 0.1, 2.5e-7])
def test_fmt_is_always_a_valid_glsl_literal(value: float) -> None:
    """An integer-valued float written as "1" is an int in GLSL, not a float."""
    text = gen_glsl.fmt(value)
    assert "." in text or "e" in text or "E" in text
    assert float(text) == pytest.approx(value, rel=1e-7)


def test_channel_ref_packs_four_per_texture() -> None:
    names = gen_glsl.textures("conv1", 10)
    assert names == ["conv1_1", "conv1_2", "conv1_3"]
    assert gen_glsl.channel_ref(names, 0) == ("conv1_1", "x")
    assert gen_glsl.channel_ref(names, 3) == ("conv1_1", "w")
    assert gen_glsl.channel_ref(names, 4) == ("conv1_2", "x")
    assert gen_glsl.channel_ref(names, 9) == ("conv1_3", "y")


def test_conv1_weights_round_trip() -> None:
    """Parse the emitted conv1 taps back out and compare against the input.

    Catches transposed kernels and swapped offsets, which a shader that merely
    compiles would not reveal.
    """
    data = weights(d1=4, d2=4)
    source = gen_glsl.generate(data, "t", "out.glsl")

    conv1 = source.split("//!HOOK LUMA")[1]
    pattern = re.compile(
        r"res\.([xyzw]) \+= (-?[\d.e+-]+) \* HOOKED_texOff"
        r"\(vec2\((-?[\d.]+), (-?[\d.]+)\)\)\.x;"
    )

    recovered = np.zeros((4, 1, 5, 5), dtype=np.float64)
    for component, weight, dx, dy in pattern.findall(conv1):
        channel = "xyzw".index(component)
        # Offsets are centred on the tap, so undo the kx-2 / ky-2 shift.
        kx, ky = int(float(dx)) + 2, int(float(dy)) + 2
        recovered[channel, 0, ky, kx] = float(weight)

    assert np.allclose(recovered, data["conv1_w"], atol=1e-6)


def test_zero_weights_are_skipped() -> None:
    """Emitting `+= 0.0 * sample` would be dead work in every fragment."""
    data = weights(d1=4, d2=4)
    data["conv1_w"][:] = 0.0
    source = gen_glsl.generate(data, "t", "out.glsl")
    assert "HOOKED_texOff" not in source
