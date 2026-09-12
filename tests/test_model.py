"""Tests for the network and the dataset's input validation."""

import pytest
import torch

from espcn_mpv.dataset import PatchDataset
from espcn_mpv.model import ESPCN


@pytest.mark.parametrize("scale,d1,d2", [(2, 4, 4), (2, 8, 8), (3, 16, 8), (4, 8, 4)])
def test_forward_upscales_by_scale(scale: int, d1: int, d2: int) -> None:
    model = ESPCN(scale, d1, d2)
    out = model(torch.rand(2, 1, 16, 24))
    assert out.shape == (2, 1, 16 * scale, 24 * scale)


def test_output_layer_starts_near_zero() -> None:
    """The deliberate init from model.py.

    A kaiming-scaled conv3 pushes the initial output far outside [0, 1], and
    Adam's correction kills most conv2 units through the ReLU. Guards against
    someone "fixing" the init back to kaiming for consistency.
    """
    model = ESPCN(2, 8, 8)
    assert model.conv3.weight.std() < 0.01
    assert model.conv1.weight.std() > 0.05
    assert torch.all(model.conv3.bias == 0)


def test_untrained_output_is_in_a_sane_range() -> None:
    """With the near-zero output init, a fresh model starts near mid-grey."""
    model = ESPCN(2, 8, 8)
    out = model(torch.rand(1, 1, 32, 32))
    assert out.abs().max() < 1.0


def test_patch_size_must_divide_by_scale(tmp_path) -> None:
    with pytest.raises(ValueError, match="divisible"):
        PatchDataset(tmp_path, scale=2, patch_size=97)


def test_empty_directory_is_an_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no images"):
        PatchDataset(tmp_path, scale=2, patch_size=96)
