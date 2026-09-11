# eval/

Scratch space for checking that a generated shader actually works. Everything
here except this file is gitignored, since the clips are third-party and the
frames are large and easy to regenerate.

## What lives here

- source clips (`*.y4m`, `*.mkv`), uncompressed test footage
- `hr.png`, a square-pixel reference frame pulled from a clip
- `lr.png`, that frame downscaled by `scale`, which is the shader's input
- mpv screenshots of the shader output
- `*.log`, mpv logs from validation runs

Finished before/after crops that go in the main README belong in
`docs/images/`, which is tracked.

Good sources are ones you can redistribute and that have no codec artifacts to
confuse the measurement: [Xiph test media](https://media.xiph.org/video/derf/)
or the Blender open movies.

## Building a ground-truth pair

Most broadcast test clips are anamorphic. Force square pixels first, or mpv's
aspect correction resamples on top of the shader and the comparison stops
being about the shader:

```bash
ffmpeg -i clip.y4m -vf "setsar=1,crop=720:480" -frames:v 1 -update 1 hr.png
ffmpeg -i hr.png -vf "scale=360:240:flags=lanczos" -update 1 lr.png
```

`hr.png` is the ground truth. `lr.png` is what the shader gets. Comparing the
shader's output against `hr.png` is the only way to tell whether it beat
bicubic, since side-by-side crops of any sharpening filter look better to the
eye whether or not they are closer to correct.

## Running the shader on it

The window has to be at least 1.3x the source or the `//!WHEN` guard
suppresses every pass. The passes still register in that case, so check the
log for evidence they *ran*, not just `Registering hook pass`. Sizing the
window to exactly `scale`x the input means nothing resamples after the final
pass:

```bash
mpv.com lr.png --no-config --vo=gpu-next \
  --glsl-shaders=../shaders/espcn_x2_8.glsl \
  --geometry=720x480 --keep-open=yes \
  --screenshot-format=png --screenshot-png-compression=0 \
  --screenshot-high-bit-depth=yes --log-file=run.log
```

Press `Ctrl+S`. That is `screenshot window`, the only mode that captures the
GPU-rendered output. Plain `s` and `S` bypass the shader path and give back an
unshaded frame, which looks like a working screenshot but is not one.

`--no-config` matters too. A normal mpv config usually sets `profile`,
`dscale` and similar, which means mpv is doing its own scaling around the
shader and the measurement is no longer isolated.

## Checking the y-axis orientation

Open question, see section 5 of the main README. Generate a second shader with
the y offsets negated in `gen_conv2_passes` and `gen_final_pass`, run both on
the same `lr.png`, and compare each against what the PyTorch model produces on
the same input. Whichever matches is the correct orientation. This settles it
without relying on eyeballing a picture.

Running the same shader under `--gpu-api=d3d11` and `--gpu-api=vulkan` is
worth doing as well. The two backends have opposite native texture coordinate
origins, and libplacebo normalizes it, but that is exactly the layer where a
flip would hide. If the outputs differ, there is a coordinate bug.
