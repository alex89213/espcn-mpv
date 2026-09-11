# espcn-mpv

Train a small [ESPCN](https://arxiv.org/abs/1609.05158) super-resolution
network on images you pick, then compile the trained weights into an
[mpv](https://mpv.io/) custom shader that upscales video luma in real time.

Same idea as [haasn/fsrcnn-mpv](https://github.com/haasn/fsrcnn-mpv) but for
ESPCN, and with no pretrained weights shipped. The point is to train on
content that looks like what you actually watch (live action, anime, one
specific show) and get a shader tuned for it.

```
  images ──▶ train ──▶ checkpoint (.pt) ──▶ export ──▶ weights (.npz)
                                                            │
                                                            ▼
                                                         gen-glsl
                                                            │
                                                            ▼
                                              shader (.glsl) ──▶ mpv
```

The network runs on the luma (Y) channel only: two convolutions (5x5 then
3x3) with ReLU, a third 3x3 convolution producing `scale * scale` channels,
and a sub-pixel shuffle that turns those channels into the upscaled image.

`shaders/espcn_x2_8.glsl` is a generated example so you can see the output
format without training anything first.

## 1. Getting a training set

No training images ship with this repo. They aren't mine to redistribute, so
you supply your own. Everything under `data/` is gitignored.

The usual starting point is **DIV2K**:
<https://data.vision.ee.ethz.ch/cvl/DIV2K/>. Download `DIV2K_train_HR.zip`
(800 high-resolution photos) and unzip it to `data/DIV2K_train_HR/`. That is
enough to train a decent general-purpose upscaler. Hold a few images out into
a second folder to use as `--val-dir` if you want a PSNR number each epoch.

Because the model is tiny, it pays to train separate weights per kind of
content rather than one shader for everything. Put each kind in its own folder
and train once per folder:

```
data/
  live_action/
  animation/
```

To build those from your own video files, pull frames out with ffmpeg. Sample
every N frames rather than every frame, since consecutive frames are nearly
identical and just make an epoch slower without adding information:

```bash
# one frame every 120 (~ every 5s at 24fps), as high-quality PNGs
ffmpeg -i show.mkv -vf "select=not(mod(n\,120))" -vsync vfr data/animation/show_%05d.png
```

Use sources that are actually sharp. The training pairs are made by
downscaling your images, so if the source is soft or full of compression
artifacts the model learns to reproduce softness.

## 2. Install

```bash
uv sync
```

This pulls `torch` and `torchvision` from the CUDA 12.6 wheel index and
everything else from PyPI. Training on CPU works but is slow. Pass
`--device cpu` if you have no GPU.

## 3. Train, export, generate

```bash
# train (scale 2, 8+8 channels, 100 epochs)
uv run espcn-mpv train \
  --data-dir data/DIV2K_train_HR \
  --val-dir data/val \
  --scale 2 --d1 8 --d2 8 \
  --patch-size 96 --patches-per-image 20 \
  --batch-size 64 --epochs 100 --lr 1e-3 \
  --checkpoint-dir checkpoints --checkpoint-name espcn_x2_8

# export the final checkpoint to a flat .npz
uv run espcn-mpv export \
  --checkpoint checkpoints/espcn_x2_8_final.pt \
  --out weights/espcn_x2_8.npz

# compile the weights into an mpv shader
uv run espcn-mpv gen-glsl \
  --weights weights/espcn_x2_8.npz \
  --out shaders/espcn_x2_8.glsl \
  --name espcn_x2_8
```

Every epoch writes `checkpoints/<name>_epoch<N>.pt` (model plus optimizer, so
you can `--resume` from one). The run ends by writing a stable
`checkpoints/<name>_final.pt`, which is the file `export` expects by
convention.

### What a healthy run looks like

Loss should start around **0.2** and drop under **0.01** within the first
epoch or two. If it starts in the *single digits* instead, the output layer is
initialized too large and most of the hidden channels will die to the ReLU
within a few hundred steps. See the comment in `model.py`.

Measured on DIV2K x2, the model passes bicubic at roughly 4000 steps and
plateaus about **+1.3 dB over bicubic** around 20000 steps (~80 epochs at the
default settings). That is what a 400-parameter network has to give.

### Degradation training (`--degrade`, on by default)

Training only on bicubic-downscaled pairs teaches the model to invert bicubic
specifically. It scores well when the test input was made the same way, and
loses that edge on anything else, which is every real video. So the LR side of
each pair gets a randomized trip through what actually happens to video:
optional slight blur, a downscale kernel drawn from bicubic/bilinear/box/
lanczos, JPEG at quality 55-95 half the time, and occasional mild noise. The
HR target is always left clean.

Measured on 40 held-out DIV2K images, `--d1 8 --d2 8`, versus lanczos:

| LR produced by | plain | **with `--degrade`** |
|---|---|---|
| bicubic (matches training) | +0.78 dB | +0.66 dB |
| lanczos downscale | +0.30 dB | +0.32 dB |
| box/area downscale | -0.12 dB | **+0.27 dB** |
| bicubic + JPEG q85 | -0.04 dB | **+0.12 dB** |

Degradation costs 0.12 dB in the bicubic case, which never happens with real
video, and turns both losses into wins. Use `--no-degrade` only to reproduce a
clean-pair baseline. Validation PSNR is always computed on clean bicubic pairs
so the number stays comparable between runs.

### Picking `--d1` and `--d2`

Any width works, since the generator splits wide layers across several
textures (see section 6). Wider is not worth much, though. Measured on DIV2K
x2, 12000 steps each:

| d1 / d2 | params | passes | PSNR vs bicubic |
|---|---|---|---|
| 4 / 4 | 400 | 3 | +1.19 dB |
| **8 / 8** | 1084 | 5 | **+1.38 dB** |
| 16 / 16 | 3316 | 9 | +1.38 dB |
| 32 / 16 | 6036 | 13 | +1.46 dB |

The gains stop almost immediately. `8 / 8` captures most of it and everything
past that buys hundredths of a dB for more passes per frame. The ceiling is
ESPCN's depth (three layers over a 9x9 receptive field), not its width, so
spending on channels does not move it. Use `--d1 8 --d2 8`.

## 4. Using the shader in mpv

```bash
mpv --glsl-shaders=shaders/espcn_x2_8.glsl video.mkv
```

Or make it permanent in `mpv.conf`:

```
glsl-shaders="~~/shaders/espcn_x2_8.glsl"
```

`~~/` is mpv's own config directory, so `~/.config/mpv/` on Linux and
`%APPDATA%\mpv\` on Windows. An absolute path works too. If you run a portable
mpv build, `~~/` points at `portable_config/` next to the executable instead.
To stack several shaders, separate them with `:` on Unix or `;` on Windows.

The shader hooks `LUMA`, so chroma is left to mpv's normal scaler. A `scale 2`
shader upscales the luma plane 2x regardless of your window size, and mpv then
scales that result to fit as usual.

Every pass carries a `//!WHEN` guard copied from FSRCNNX, so the shader costs
nothing unless the output is at least 1.3x the source resolution. You can
leave it in `mpv.conf` permanently without it firing on content that is
already big enough. The flip side is that a small window silently disables it:
if you are testing and nothing seems to happen, check your output resolution
first.

**Use a float framebuffer.** The intermediate feature maps are unbounded ReLU
outputs and will exceed 1.0. An 8-bit FBO clamps them and the picture comes
out wrong. `vo=gpu-next` already picks a float format, but on older `vo=gpu`
add:

```
fbo-format=rgba16f
```

FSRCNNX documents the same requirement for the same reason.

## 5. What is verified and what is not

**Verified.** The generated GLSL loads in mpv 0.41 (libplacebo 7.371, D3D11
backend) with no compile warnings, and all five passes of an `8/8` shader
register correctly.

**Not verified.** Whether the convolution arithmetic produces the right
picture on real hardware. The final pass picks which of the `scale * scale`
conv3 outputs an output pixel needs from where that pixel falls inside its
source texel:

```glsl
vec2 fcoord = fract(conv2_1_pos * conv2_1_size);
ivec2 sub = ivec2(fcoord * 2.0);
int idx = sub.y * 2 + sub.x;      // nn.PixelShuffle channel order
```

This is done entirely in texture space, following FSRCNNX's aggregation pass.
An earlier version read `gl_FragCoord` instead, which was riskier: framebuffer
space and texture space need not share an origin convention, and mixing the
two can tile or scramble the output. Deriving the taps and the sub-pixel index
from the same coordinates rules that out.

What remains is milder. If mpv's texture y axis runs opposite to PyTorch's row
order, then every pass is consistently mirrored, and the net effect is that
the network runs with a vertically flipped kernel. Training augments with
vertical flips, so that costs a little accuracy rather than producing visible
corruption. It would show up as the shader underperforming its measured PSNR,
not as an obviously broken picture. To rule it out, flip the sign of the y
offsets in `gen_conv2_passes` and `gen_final_pass` and compare. See `eval/`
for how to set up that comparison.

Beyond that: with `--degrade` the model sees mild JPEG and noise during
training, so it tolerates ordinary compressed sources, but it is an upscaler
and not a restoration model. Heavy banding, ringing or block artifacts are not
what it was trained to remove.

## 6. How the shader is laid out

A `//!SAVE` texture has four components, so a layer carries at most 4 feature
channels per texture. Wider layers are split across several: `--d1 16` emits
four passes saving `conv1_1 .. conv1_4`, and the conv2 passes bind all four to
reassemble the 16-channel activation. This is the scheme FSRCNNX uses to carry
8 or 16 feature maps, and it is why there is no cap on `--d1` and `--d2`.

The pass count is `ceil(d1/4) + ceil(d2/4) + 1`. Everything except the last
pass runs at *source* resolution. The upscale happens only in the final pass,
where conv3 and the pixel shuffle are fused. That is the core ESPCN idea, and
the reason it is cheap enough to be a shader at all: a network that upscaled
first and then convolved would do `scale^2` times the work.

Reference implementations worth reading next to `gen_glsl.py`:

- [igv/FSRCNN-TensorFlow](https://github.com/igv/FSRCNN-TensorFlow) is the
  source of the FSRCNNX shaders. The multi-texture packing and the aggregation
  pass this generator's final pass is modeled on both come from there.
- [bjin/mpv-prescalers](https://github.com/bjin/mpv-prescalers) has RAVU and
  nnedi3 as generated GLSL. RAVU is a learned *filter bank* (RAISR-style
  gradient hashing) rather than a CNN, and stores its weights in a lookup
  texture instead of as literals.
- [bloc97/Anime4K](https://github.com/bloc97/Anime4K) exports CNN modes the
  same way, tuned for animation.

Two of their optimizations are deliberately not implemented here: bilinear tap
merging (placing a sample between two texels so the texture unit returns a
weighted sum of both, halving fetches) and compute-shader passes. Both would
need real GPU testing to validate, and neither reference uses bilinear merging
for its convolution taps.

## Repository layout

```
src/espcn_mpv/    model, dataset, training, weight export, GLSL generator
shaders/          generated .glsl (one example committed, rest gitignored)
eval/             scratch space for validating a shader, see eval/README.md
docs/images/      before/after crops used in this README
data/             training sets, gitignored
checkpoints/      training output, gitignored
weights/          exported .npz, gitignored
```
