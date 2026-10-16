"""Tests for the metric implementations.

These are hand-rolled rather than taken from scikit-image, so the properties
every metric is supposed to have are worth pinning down: perfect scores on
identical input, monotonicity as error grows, and no crash on inputs too small
for MS-SSIM's five scales.
"""

import math
from pathlib import Path

import pytest
import torch

from espcn_mpv import metrics


def image(height: int = 64, width: int = 64, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(1, 1, height, width, generator=generator)


def test_identical_images_score_perfectly() -> None:
    a = image()
    assert metrics.mae(a, a) == 0.0
    assert metrics.psnr(a, a) == math.inf
    assert metrics.ssim(a, a) == pytest.approx(1.0, abs=1e-6)
    assert metrics.ms_ssim(a, a) == pytest.approx(1.0, abs=1e-6)


def test_metrics_degrade_as_error_grows() -> None:
    a = image()
    small = (a + 0.01).clamp(0, 1)
    large = (a + 0.20).clamp(0, 1)

    assert metrics.mae(a, small) < metrics.mae(a, large)
    assert metrics.psnr(a, small) > metrics.psnr(a, large)
    assert metrics.ssim(a, small) > metrics.ssim(a, large)
    assert metrics.ms_ssim(a, small) > metrics.ms_ssim(a, large)


def test_psnr_matches_its_definition() -> None:
    a = torch.zeros(1, 1, 8, 8)
    b = torch.full((1, 1, 8, 8), 0.1)
    assert metrics.psnr(a, b) == pytest.approx(10 * math.log10(1 / 0.01))


def test_ssim_is_bounded() -> None:
    a, b = image(seed=1), image(seed=2)
    assert -1.0 <= metrics.ssim(a, b) <= 1.0
    assert 0.0 <= metrics.ms_ssim(a, b) <= 1.0


def test_ms_ssim_survives_images_too_small_for_five_scales() -> None:
    """Five scales need ~176px. Smaller inputs drop scales instead of raising."""
    for size in (16, 24, 48, 100):
        a, b = image(size, size, seed=1), image(size, size, seed=2)
        value = metrics.ms_ssim(a, b)
        assert 0.0 <= value <= 1.0
        assert not math.isnan(value)


def test_ms_ssim_on_the_smallest_workable_image() -> None:
    a = image(11, 11)
    assert metrics.ms_ssim(a, a) == pytest.approx(1.0, abs=1e-6)


def test_compare_returns_every_metric() -> None:
    a, b = image(seed=1), image(seed=2)
    result = metrics.compare(a, b)
    assert set(result) == set(metrics.METRICS)
    assert all(isinstance(v, float) for v in result.values())


def test_compare_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        metrics.compare(image(64, 64), image(32, 32))


def test_luma_round_trip_through_disk(tmp_path: Path) -> None:
    """save_luma then load_luma should be lossless to 8-bit quantization."""
    original = image(32, 32)
    path = tmp_path / "frame.png"
    metrics.save_luma(original, path)
    assert metrics.psnr(original, metrics.load_luma(path)) > 48.0


def test_save_luma_clamps_out_of_range_values(tmp_path: Path) -> None:
    """ReLU output can exceed 1.0; writing it must not wrap around."""
    path = tmp_path / "frame.png"
    metrics.save_luma(torch.tensor([[[[-0.5, 0.5, 1.5]]]]), path)
    loaded = metrics.load_luma(path)
    assert loaded.min() >= 0.0 and loaded.max() <= 1.0
    assert loaded.flatten().tolist() == pytest.approx([0.0, 0.5, 1.0], abs=1 / 255)
