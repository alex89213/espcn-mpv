"""ESPCN network (Shi et al. 2016), luma-only.

"Real-Time Single Image and Video Super-Resolution Using an Efficient
Sub-Pixel Convolutional Neural Network".

The paper uses d1=64, d2=32. The GLSL generator packs feature channels four
per RGBA texture and splits wider layers across several textures, so any
width works. Cost in the shader grows with ceil(d/4) textures per layer, plus
d1*d2 multiply-adds per output pixel in conv2. d1=8, d2=8 is a reasonable
middle ground for real-time playback.
"""

from typing import override

import torch
from torch import nn


class ESPCN(nn.Module):
    def __init__(self, scale: int = 2, d1: int = 4, d2: int = 4):
        super().__init__()
        self.scale: int = scale
        self.d1: int = d1
        self.d2: int = d2

        self.conv1: nn.Conv2d = nn.Conv2d(1, d1, kernel_size=5, padding=2)
        self.relu1: nn.ReLU = nn.ReLU(inplace=True)
        self.conv2: nn.Conv2d = nn.Conv2d(d1, d2, kernel_size=3, padding=1)
        self.relu2: nn.ReLU = nn.ReLU(inplace=True)
        self.conv3: nn.Conv2d = nn.Conv2d(d2, scale * scale, kernel_size=3, padding=1)
        self.pixel_shuffle: nn.PixelShuffle = nn.PixelShuffle(scale)

        self._init_weights()

    def _init_weights(self) -> None:
        for conv in (self.conv1, self.conv2, self.conv3):
            assert conv.bias is not None
            nn.init.zeros_(conv.bias)

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")

        # The output layer starts near zero rather than at kaiming scale. With
        # only 4 channels per hidden layer there is no redundancy to spare. A
        # kaiming-scaled conv3 makes the initial output about 2x the target
        # range, and Adam's first steps then shrink the net hard enough to push
        # most conv2 units permanently below zero. Measured over 600 steps on
        # DIV2K: kaiming throughout left 2 of 4 conv2 channels dead and reached
        # 19.6 dB, versus 26.1 dB with all 4 alive using the init below.
        nn.init.normal_(self.conv3.weight, mean=0.0, std=0.001)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, h, w) in [0, 1] -> (batch, 1, h*scale, w*scale)."""
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return self.pixel_shuffle(x)
