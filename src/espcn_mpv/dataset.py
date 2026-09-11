"""LR/HR luma patch pairs built on the fly from a folder of images."""

import io
import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# Downscale kernels drawn from when `degrade` is on. A model trained only on
# bicubic-downscaled pairs learns to invert bicubic specifically, and measures
# well only when the test input was made the same way; on video resized by
# some other resampler the advantage over plain lanczos disappears. Sampling
# the kernel keeps the model from specialising on one of them.
DOWNSCALE_FILTERS = (Image.BICUBIC, Image.BILINEAR, Image.BOX, Image.LANCZOS)
DOWNSCALE_WEIGHTS = (0.4, 0.2, 0.2, 0.2)


class PatchDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        scale: int = 2,
        patch_size: int = 96,
        patches_per_image: int = 20,
        augment: bool = True,
        degrade: bool = False,
    ):
        if patch_size % scale != 0:
            raise ValueError(
                f"patch_size ({patch_size}) must be divisible by scale ({scale})"
            )

        self.image_dir = Path(image_dir)
        self.scale = scale
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.augment = augment
        self.degrade = degrade

        self.paths = sorted(
            p
            for p in self.image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(
                f"no images found under {self.image_dir} "
                f"(looked for {', '.join(sorted(IMAGE_EXTENSIONS))})"
            )

    def __len__(self) -> int:
        return len(self.paths) * self.patches_per_image

    def _load(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        # Upscale first if the image is too small to crop a full patch from.
        w, h = img.size
        if w < self.patch_size or h < self.patch_size:
            ratio = max(self.patch_size / w, self.patch_size / h)
            new_size = (max(self.patch_size, round(w * ratio)),
                        max(self.patch_size, round(h * ratio)))
            img = img.resize(new_size, Image.BICUBIC)
        return img

    def _augment(self, patch: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            patch = patch.transpose(Image.FLIP_TOP_BOTTOM)
        angle = random.choice((0, 90, 180, 270))
        if angle:
            patch = patch.rotate(angle, expand=True)
            # A 90/270 rotation of a non-square patch swaps the sides; put it
            # back to patch_size x patch_size so batching stays uniform.
            if patch.size != (self.patch_size, self.patch_size):
                patch = patch.resize(
                    (self.patch_size, self.patch_size), Image.BICUBIC
                )
        return patch

    def _make_lr(self, hr: Image.Image) -> Image.Image:
        """Produce the LR counterpart of an HR patch.

        Without `degrade` this is a plain bicubic downscale. With it, the LR
        side goes through a randomized approximation of what happens to real
        video on its way to your screen: a little source softness, some
        resampler that isn't necessarily bicubic, and lossy compression.
        The HR target is always left clean.
        """
        size = (self.patch_size // self.scale,) * 2
        if not self.degrade:
            return hr.resize(size, Image.BICUBIC)

        lr = hr
        if random.random() < 0.35:
            lr = lr.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 0.8)))

        kernel = random.choices(DOWNSCALE_FILTERS, DOWNSCALE_WEIGHTS)[0]
        lr = lr.resize(size, kernel)

        if random.random() < 0.5:
            buf = io.BytesIO()
            lr.save(buf, "JPEG", quality=random.randint(55, 95))
            buf.seek(0)
            lr = Image.open(buf).convert("RGB")

        return lr

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self._load(self.paths[index // self.patches_per_image])

        w, h = img.size
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        hr = img.crop((x, y, x + self.patch_size, y + self.patch_size))

        if self.augment:
            hr = self._augment(hr)

        lr = self._make_lr(hr)

        hr_t = TF.to_tensor(hr.convert("YCbCr").getchannel("Y"))
        lr_t = TF.to_tensor(lr.convert("YCbCr").getchannel("Y"))

        if self.degrade and random.random() < 0.3:
            sigma = random.uniform(0.002, 0.012)
            lr_t = (lr_t + torch.randn_like(lr_t) * sigma).clamp(0.0, 1.0)

        return lr_t, hr_t
