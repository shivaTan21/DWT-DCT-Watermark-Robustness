# Evaluating the Robustness of DWT-DCT Watermarks Under AI-Based Image Enhancement

Reproducible research code for testing whether AI-based image enhancement
(super-resolution / "AI upscaling") behaves as a watermark-removal attack
against a classical DWT-DCT image watermark, compared against traditional
signal-processing attacks (JPEG compression, noise, filtering, crop+resize).

This project does **not** propose a new watermarking algorithm. It uses a
standard, textbook DWT-DCT watermarking scheme purely as an instrument to
measure whether AI enhancement should be treated as a distinct class of
watermark-removal attack.

## How it works

- **Embedding** (`src/watermark.py`): the 512x512 RGB cover image is
  converted to YCbCr; a single-level Haar DWT is applied to the Y (luma)
  channel only. The LL (approximation) subband is split into 8x8 blocks;
  each block is DCT-transformed, and one watermark bit is embedded per
  block by enforcing a minimum signed gap between two fixed mid-frequency
  coefficients. Embedding strength is controlled by `alpha` (default
  `0.1`), plus a small absolute margin floor so the watermark survives
  8-bit PNG quantization even in flat image regions. The watermarked Y
  channel is recombined with the untouched Cb/Cr channels and converted
  back to RGB. Attacks and metrics (PSNR, SSIM) operate on the full RGB
  image.
- **Extraction** is blind: it only needs the (possibly attacked) image and
  the watermark's bit-grid shape — no original image or stored coefficients
  required.
- **Sanity check**: before any attack runs, the pipeline embeds and
  extracts on one clean, unattacked image and requires
  `NC > 0.99`. If that fails, a `RuntimeError` is raised and the experiment
  stops — a broken codec must never silently produce "robustness" numbers.

## Project structure

```
data/
  original_images/     # put your PNG/JPG source images here
  processed_images/    # generated: 512x512 RGB versions
  watermark.png         # optional: provide your own binary watermark image
results/
  watermarked/                  # generated: watermarked images per (image, alpha)
  attacked/traditional/         # generated: traditional-attack outputs
  attacked/ai_enhancement/      # generated: AI-enhancement-attack outputs
  plots/                        # generated: comparison plots
  metrics.csv                   # generated: all per-row metrics
  summary_table.csv             # generated: metrics grouped by attack
src/
  config.py               # paths, seed, image/watermark size, alpha values
  watermark.py             # DWT-DCT embed/extract
  attacks_traditional.py   # JPEG, noise, median/Gaussian blur, crop+resize
  attacks_ai.py            # Real-ESRGAN / OpenCV DNN-SR / fallback wrappers
  metrics.py               # PSNR, SSIM, normalized correlation, BER
  run_experiment.py        # orchestrates the full pipeline -> metrics.csv
  plot_results.py          # metrics.csv -> summary_table.csv + plots
```

## 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The base install covers everything except AI super-resolution backends.
Those are optional (see `requirements.txt`):

- **Real-ESRGAN**: `pip install torch py-real-esrgan`, plus a `.pth`
  weights file you pass via `--real-esrgan-model`.
- **OpenCV DNN super-resolution**: `pip install opencv-contrib-python`
  (uninstall plain `opencv-python` first — both packages provide the same
  `cv2` module and will conflict), plus a pretrained `.pb` model passed via
  `--opencv-sr-model`.

If neither is installed, the experiment still runs: `attacks_ai.py` always
provides a non-AI `ai_fallback_enhancement` (CLAHE + denoise + unsharp mask)
so the AI-attack stage of the pipeline is never empty. You can also run an
external tool manually (Topaz, Photoshop Enhance, an online AI enhancer)
on the files in `results/watermarked/` and drop the outputs into
`results/attacked/ai_enhancement/` using the same
`<image>__alpha<alpha>__<your_attack_name>.png` naming convention — they'll
be picked up by the metrics/plotting stage like any other attack.

## 2. Add images

Drop PNG/JPG images into `data/original_images/`. They'll be resized to
512x512 RGB automatically. For a paper-style benchmark, standard test
sets (e.g. the Kodak image suite) work well. Optionally, drop a binary
watermark image at `data/watermark.png` (it will be thresholded to a 32x32
bit-grid); otherwise a reproducible random 32x32 watermark is generated
from the fixed seed (`42`).

## 3. Run the experiment

```bash
python -m src.run_experiment
python -m src.plot_results
```

`run_experiment.py` sweeps `alpha` over `[0.05, 0.1, 0.2, 0.3]` by default
(edit `ALPHA_VALUES` in `src/config.py`, or pass `--alphas`):

```bash
python -m src.run_experiment --alphas 0.05 0.1 0.2
python -m src.run_experiment --real-esrgan-model /path/to/RealESRGAN_x4.pth
python -m src.run_experiment --opencv-sr-model /path/to/EDSR_x2.pb
```

Everything is seeded (`RANDOM_SEED = 42` in `src/config.py`) — watermark
generation and Gaussian noise are reproducible across runs.

## 4. Interpreting the output

**`results/metrics.csv`** — one row per (image, alpha, attack), with:

| column | meaning |
|---|---|
| `psnr_watermark`, `ssim_watermark` | quality of the watermarked image vs. the original (before any attack) |
| `psnr_after_attack`, `ssim_after_attack` | quality after the attack (equal to the above for the `watermarked_only`/no-attack row) |
| `nc` | normalized correlation between the original and extracted watermark bit-grids, in `[-1, 1]`; `1.0` = perfect recovery |
| `ber` | bit error rate between original and extracted watermark, in `[0, 1]`; `0.0` = perfect recovery |

**`results/summary_table.csv`** — `nc`/`ber`/`psnr`/`ssim` averaged
(mean + std) per attack, across all images and alphas.

**Plots** (`results/plots/`):
- `nc_by_attack.png` / `ber_by_attack.png` — bar charts ranking attacks by
  mean watermark survival. The red dashed line on the NC plot is the
  sanity-check threshold (0.99): the `watermarked_only` bar should sit at
  or above it, and any attack below it is removing the watermark.
- `psnr_vs_nc.png` / `ssim_vs_nc.png` — scatter plots of perceptual
  distortion vs. watermark survival, colored by attack category
  (`traditional` vs. `ai_enhancement`). The central research question is
  whether `ai_enhancement` points sit at a different distortion/NC
  trade-off than `traditional` points — i.e., does AI enhancement remove
  more watermark signal than its visible distortion would suggest?

A low `nc`/high `ber` after an attack, especially when `psnr`/`ssim` after
that attack are still high (i.e. the image looks barely changed), is the
key signature of a watermark-removal attack as opposed to plain
lossy degradation.

## Notes on the embedding scheme

- The watermark is embedded in the LL (low-frequency) subband, trading some
  imperceptibility for robustness, since this experiment's purpose is to
  stress-test robustness.
- It is a block-based, spatially-registered scheme: cropping (even with
  resize back to the original dimensions) misaligns the block grid and is
  expected to destroy the watermark almost completely (`crop_resize_80pct`
  in the baseline attacks reflects this known weakness of block-based DWT-DCT
  schemes, not a bug).
- `ABSOLUTE_MARGIN_FLOOR` in `src/watermark.py` sets a minimum embedding
  margin (independent of `alpha`) so that flat/low-texture image regions
  still survive 8-bit PNG quantization. It was tuned against the bundled
  sanity check until clean-image extraction reliably exceeded NC = 0.99.
