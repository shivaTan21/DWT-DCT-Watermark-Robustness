"""Image-quality and watermark-fidelity metrics. Operate on full RGB images."""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(img1, img2):
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    return float(peak_signal_noise_ratio(img1, img2, data_range=255))


def ssim(img1, img2):
    """SSIM over an (H, W, 3) RGB image pair, averaged across channels."""
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    return float(structural_similarity(img1, img2, data_range=255, channel_axis=-1))


def normalized_correlation(wm_original, wm_extracted):
    """
    Normalized correlation (NC) between two binary watermark bit-grids.

    Bits are mapped to {-1, +1} (bipolar) before correlating. This is the
    standard convention for NC in the watermarking literature: it keeps NC
    in [-1, 1] with NC=1 for a perfect match, rather than skewing the score
    toward 0-bit agreement the way a raw {0,1} dot product would.
    """
    a = np.asarray(wm_original, dtype=np.float64).flatten() * 2 - 1
    b = np.asarray(wm_extracted, dtype=np.float64).flatten() * 2 - 1
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(a * b) / denom)


def bit_error_rate(wm_original, wm_extracted):
    a = np.asarray(wm_original).flatten()
    b = np.asarray(wm_extracted).flatten()
    return float(np.mean(a != b))
