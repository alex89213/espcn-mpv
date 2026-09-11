"""PSNR and SSIM for single-channel images.

Kept dependency-free beyond torch and pillow, which the project already needs.
Everything works on (1, 1, h, w) tensors in [0, 1].
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_luma(path: str | Path) -> torch.Tensor:
    """Load an image and return its luma channel as (1, 1, h, w) in [0, 1].

    Grayscale files are taken as-is. Color files go through pillow's YCbCr
    conversion, which is the same one dataset.py uses to build training pairs,
    so a comparison never mixes two different definitions of Y.

    Pillow reads 16-bit PNGs down to 8 bits. That is fine here because the
    reference frames ffmpeg writes are 8-bit as well, so 8 bits is the real
    ceiling on any comparison rather than a limit this loader introduces.
    """
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("RGB").convert("YCbCr").getchannel("Y")
    array = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None, None]


def save_luma(tensor: torch.Tensor, path: str | Path) -> None:
    """Write a (1, 1, h, w) tensor in [0, 1] as an 8-bit grayscale PNG."""
    array = tensor.detach().squeeze().clamp(0.0, 1.0).numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((array * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def mae(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean absolute error between two tensors in [0, 1]."""
    return float((a - b).abs().mean())


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR in dB between two tensors in [0, 1]."""
    mse = float(((a - b) ** 2).mean())
    if mse <= 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def _gaussian(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    weights = torch.exp(-(coords**2) / (2 * sigma**2))
    return weights / weights.sum()


def _ssim_parts(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """(mean SSIM, mean contrast-structure term) for two tensors in [0, 1].

    Standard parameters: 11x11 Gaussian window, sigma 1.5, K1=0.01, K2=0.03.
    The window is applied separably and without padding, so the outer 5 pixels
    are dropped rather than contributing an edge artifact to the mean.

    MS-SSIM needs the contrast-structure term on its own at every scale but
    the full SSIM only at the coarsest, which is why both come back here.
    """
    window = _gaussian()
    size = window.numel()
    horizontal = window.view(1, 1, 1, size)
    vertical = window.view(1, 1, size, 1)

    def blur(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.conv2d(t, horizontal), vertical)

    c1, c2 = 0.01**2, 0.03**2

    mu_a, mu_b = blur(a), blur(b)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = blur(a * a) - mu_a2
    var_b = blur(b * b) - mu_b2
    cov = blur(a * b) - mu_ab

    contrast = (2 * cov + c2) / (var_a + var_b + c2)
    luminance = (2 * mu_ab + c1) / (mu_a2 + mu_b2 + c1)
    return float((luminance * contrast).mean()), float(contrast.mean())


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean SSIM between two tensors in [0, 1]."""
    return _ssim_parts(a, b)[0]


# Scale weights from Wang et al. 2003, "Multiscale structural similarity for
# image quality assessment".
MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def ms_ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Multi-scale SSIM between two tensors in [0, 1].

    SSIM at a single scale depends on viewing distance, since the same window
    covers different amounts of image content depending on resolution. MS-SSIM
    halves the image repeatedly and combines the per-scale results, which is
    why it tends to track subjective ranking better for upscalers.

    Scales are dropped if the image is too small to survive the halvings, and
    the weights renormalized, so small crops degrade to plain SSIM instead of
    raising.
    """
    levels = len(MS_SSIM_WEIGHTS)
    while levels > 1 and min(a.shape[-2:]) // (2 ** (levels - 1)) < 11:
        levels -= 1

    weights = MS_SSIM_WEIGHTS[:levels]
    total = sum(weights)

    result = 1.0
    for level in range(levels):
        full, contrast = _ssim_parts(a, b)
        value = full if level == levels - 1 else contrast
        result *= max(value, 1e-8) ** (weights[level] / total)
        if level < levels - 1:
            a = F.avg_pool2d(a, 2)
            b = F.avg_pool2d(b, 2)
    return result


#: Metric name -> (function, "higher is better"). Order is the report order.
METRICS = {
    "MAE": (mae, False),
    "PSNR": (psnr, True),
    "SSIM": (ssim, True),
    "MS-SSIM": (ms_ssim, True),
}


def compare(reference: torch.Tensor, other: torch.Tensor) -> dict[str, float]:
    """Every metric in METRICS for `other` against `reference`."""
    if reference.shape != other.shape:
        raise ValueError(
            f"shape mismatch: reference is {tuple(reference.shape[-2:])}, "
            f"image is {tuple(other.shape[-2:])}"
        )
    return {name: fn(reference, other) for name, (fn, _) in METRICS.items()}
