"""
DWT-DCT digital image watermarking.

Embedding strategy
-------------------
1. A single-level 2D DWT (default: Haar) is applied to the cover image,
   producing LL/LH/HL/HH subbands. The watermark is embedded in the LL
   (approximation) subband because low-frequency content survives common
   signal-processing and AI-enhancement operations better than detail
   subbands -- this experiment is specifically about *robustness*, so we
   trade some perceptual transparency for resilience.
2. The LL subband is partitioned into non-overlapping 8x8 blocks, and each
   block is transformed with a 2D DCT-II.
3. Each watermark bit is embedded using a relative comparison between two
   fixed, similar-magnitude mid-frequency DCT coefficients in the block
   (c1 at COEFF_POS_1, c2 at COEFF_POS_2). To embed bit=1 we enforce
   c1 - c2 >= margin; to embed bit=0 we enforce c2 - c1 >= margin, where
   margin scales with both the local block energy (alpha) and a small
   absolute floor (so embedding strength doesn't vanish in flat/smooth
   regions where both coefficients start near zero).
4. Extraction only needs the (possibly attacked) image and the watermark's
   bit-grid shape -- it re-derives c1, c2 per block and reads bit=1 if
   c1 > c2 else 0. No reference to the original image is required, so this
   is a blind watermarking scheme.

This is a textbook reference implementation chosen for clarity and
reproducibility, not an attempt at a novel or maximally imperceptible
watermarking scheme.

`embed_watermark`/`extract_watermark` operate on a single-channel array.
`embed_watermark_rgb`/`extract_watermark_rgb` are colorspace wrappers for
RGB images: they convert to YCbCr, apply the same embed/extract logic to
the Y (luma) channel only, and convert back to RGB -- the rest of the
pipeline (attacks, metrics) operates on full RGB.
"""

import numpy as np
import pywt
from scipy.fftpack import dct, idct

BLOCK_SIZE = 8
COEFF_POS_1 = (4, 1)
COEFF_POS_2 = (1, 4)
# Alternate embedding positions chosen from src/analyze_dct_stability.py's
# joint stability ranking against Real-ESRGAN and JPEG Q50 -- see
# use_stable_positions below.
COEFF_POS_1_STABLE = (7, 1)
COEFF_POS_2_STABLE = (1, 7)
# Best-performing coefficient pair from frequency_sweep.csv (highest mean NC
# across all attacks).
COEFF_POS_1_OPTIMIZED = (7, 5)
COEFF_POS_2_OPTIMIZED = (7, 7)
ABSOLUTE_MARGIN_FLOOR = 30.0
DEFAULT_ALPHA = 0.1
DEFAULT_WAVELET = "haar"


def _dct2(block):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def _idct2(block):
    return idct(idct(block.T, norm="ortho").T, norm="ortho")


def generate_watermark(shape=(32, 32), seed=42):
    """Generate a reproducible random binary ({0,1}) watermark bit-grid."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=shape).astype(np.uint8)


def load_watermark_image(path, shape=(32, 32)):
    """Load a grayscale image and threshold it into a binary watermark bit-grid."""
    import cv2

    wm = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if wm is None:
        raise FileNotFoundError(f"Could not read watermark image at {path}")
    wm = cv2.resize(wm, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return (wm >= 128).astype(np.uint8)


def _max_blocks(subband_shape):
    rows, cols = subband_shape
    return (rows // BLOCK_SIZE) * (cols // BLOCK_SIZE)


def embed_watermark(image, watermark, alpha=DEFAULT_ALPHA, wavelet=DEFAULT_WAVELET,
                    use_stable_positions=False, use_optimized_positions=False):
    """
    Embed a binary watermark into a grayscale image using DWT-DCT.

    Parameters
    ----------
    image : np.ndarray, shape (H, W)
    watermark : np.ndarray of {0,1}, shape (wm_h, wm_w)
    alpha : embedding strength (scales with local block energy)
    wavelet : PyWavelets wavelet name
    use_stable_positions : if True, embed at COEFF_POS_1_STABLE/COEFF_POS_2_STABLE
    use_optimized_positions : if True, embed at COEFF_POS_1_OPTIMIZED/COEFF_POS_2_OPTIMIZED
        (takes precedence over use_stable_positions when both are True)

    Returns
    -------
    watermarked image as a float64 array, same shape as `image`.
    """
    image = image.astype(np.float64)
    bits = watermark.flatten()
    if use_optimized_positions:
        pos1, pos2 = COEFF_POS_1_OPTIMIZED, COEFF_POS_2_OPTIMIZED
    elif use_stable_positions:
        pos1, pos2 = COEFF_POS_1_STABLE, COEFF_POS_2_STABLE
    else:
        pos1, pos2 = COEFF_POS_1, COEFF_POS_2

    LL, (LH, HL, HH) = pywt.dwt2(image, wavelet)
    rows, cols = LL.shape
    blocks_per_row = cols // BLOCK_SIZE

    n_blocks_available = _max_blocks(LL.shape)
    if bits.size > n_blocks_available:
        raise ValueError(
            f"Watermark needs {bits.size} blocks but the LL subband only "
            f"offers {n_blocks_available}. Use a smaller watermark or a larger image."
        )

    LL_out = LL.copy()
    for idx, bit in enumerate(bits):
        br = (idx // blocks_per_row) * BLOCK_SIZE
        bc = (idx % blocks_per_row) * BLOCK_SIZE
        block = LL_out[br:br + BLOCK_SIZE, bc:bc + BLOCK_SIZE]
        dct_block = _dct2(block)

        c1 = dct_block[pos1]
        c2 = dct_block[pos2]
        energy = (abs(c1) + abs(c2)) / 2.0
        margin = alpha * energy + ABSOLUTE_MARGIN_FLOOR

        if bit == 1:
            current = c1 - c2
            if current < margin:
                delta = (margin - current) / 2.0
                c1 += delta
                c2 -= delta
        else:
            current = c2 - c1
            if current < margin:
                delta = (margin - current) / 2.0
                c2 += delta
                c1 -= delta

        dct_block[pos1] = c1
        dct_block[pos2] = c2
        LL_out[br:br + BLOCK_SIZE, bc:bc + BLOCK_SIZE] = _idct2(dct_block)

    watermarked = pywt.idwt2((LL_out, (LH, HL, HH)), wavelet)
    watermarked = watermarked[: image.shape[0], : image.shape[1]]
    return watermarked


def extract_watermark(image, watermark_shape, wavelet=DEFAULT_WAVELET,
                      use_stable_positions=False, use_optimized_positions=False):
    """Blindly extract a binary watermark bit-grid of shape `watermark_shape` from `image`."""
    image = image.astype(np.float64)
    n_bits = int(np.prod(watermark_shape))
    if use_optimized_positions:
        pos1, pos2 = COEFF_POS_1_OPTIMIZED, COEFF_POS_2_OPTIMIZED
    elif use_stable_positions:
        pos1, pos2 = COEFF_POS_1_STABLE, COEFF_POS_2_STABLE
    else:
        pos1, pos2 = COEFF_POS_1, COEFF_POS_2

    LL, _ = pywt.dwt2(image, wavelet)
    cols = LL.shape[1]
    blocks_per_row = cols // BLOCK_SIZE

    n_blocks_available = _max_blocks(LL.shape)
    if n_bits > n_blocks_available:
        raise ValueError(
            f"Requested {n_bits} bits but the LL subband only offers "
            f"{n_blocks_available} blocks."
        )

    bits = np.zeros(n_bits, dtype=np.uint8)
    for idx in range(n_bits):
        br = (idx // blocks_per_row) * BLOCK_SIZE
        bc = (idx % blocks_per_row) * BLOCK_SIZE
        block = LL[br:br + BLOCK_SIZE, bc:bc + BLOCK_SIZE]
        dct_block = _dct2(block)
        bits[idx] = 1 if dct_block[pos1] > dct_block[pos2] else 0

    return bits.reshape(watermark_shape)


def rgb_to_ycbcr(image):
    """Convert an (H, W, 3) RGB float array to (H, W, 3) YCbCr (ITU-R BT.601, full range)."""
    image = image.astype(np.float64)
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    return np.stack([y, cb, cr], axis=-1)


def ycbcr_to_rgb(image):
    """Convert an (H, W, 3) YCbCr float array (ITU-R BT.601, full range) back to RGB."""
    y, cb, cr = image[..., 0], image[..., 1], image[..., 2]
    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)
    return np.stack([r, g, b], axis=-1)


SVD_HH_BLOCK_SIZE = 16  # block size for HH SVD; 16×16 on a 256×256 HH → 128 pairs = 128 bits


def _svd_hh_capacity(hh_shape):
    """
    Number of embeddable bits in the HH subband.

    HH is divided into non-overlapping SVD_HH_BLOCK_SIZE × SVD_HH_BLOCK_SIZE blocks;
    bits are encoded as comparisons between the top singular values of consecutive
    block pairs.  For a 256×256 HH (from a 512×512 image) this gives
    (256//16)² / 2 = 128 bits, which equals floor(min(HH.shape)/2) = 128 and
    satisfies the ≥128-bit requirement.
    """
    nb_r = hh_shape[0] // SVD_HH_BLOCK_SIZE
    nb_c = hh_shape[1] // SVD_HH_BLOCK_SIZE
    return (nb_r * nb_c) // 2


def _hh_block_svd(HH, idx, nb_c):
    """Return (U, S, Vt, row_start, col_start) for block `idx` of HH."""
    r = (idx // nb_c) * SVD_HH_BLOCK_SIZE
    c = (idx % nb_c) * SVD_HH_BLOCK_SIZE
    block = HH[r:r + SVD_HH_BLOCK_SIZE, c:c + SVD_HH_BLOCK_SIZE]
    U, S, Vt = np.linalg.svd(block, full_matrices=False)
    return U, S, Vt, r, c


def embed_watermark_svd_hh(y_channel, watermark, delta=ABSOLUTE_MARGIN_FLOOR, wavelet=DEFAULT_WAVELET):
    """
    Embed binary watermark into Y-channel using block-wise SVD on the HH subband.

    Each bit is encoded by comparing the top singular values of two consecutive
    HH blocks: bit=1 → σ_A > σ_B + delta; bit=0 → σ_B > σ_A + delta.
    """
    y = y_channel.astype(np.float64)
    bits = watermark.flatten()

    LL, (LH, HL, HH) = pywt.dwt2(y, wavelet)

    nb_c = HH.shape[1] // SVD_HH_BLOCK_SIZE
    capacity = _svd_hh_capacity(HH.shape)
    if bits.size > capacity:
        raise ValueError(
            f"Watermark needs {bits.size} bits but HH SVD capacity is only {capacity} "
            f"(HH shape {HH.shape}, block size {SVD_HH_BLOCK_SIZE})."
        )

    HH_out = HH.copy()
    for i, bit in enumerate(bits):
        U_a, S_a, Vt_a, ra, ca = _hh_block_svd(HH_out, 2 * i, nb_c)
        U_b, S_b, Vt_b, rb, cb = _hh_block_svd(HH_out, 2 * i + 1, nb_c)

        # Only ever increase S[0] of the "winner" block so it stays the block's
        # max SV after reconstruction and re-SVD (reducing S[0] below S[1] would
        # let S[1] become the new top SV, corrupting the encoded comparison).
        s_a = max(S_a[0], 1.0)
        s_b = max(S_b[0], 1.0)

        if bit == 1:
            if s_a - s_b < delta:
                S_a[0] = s_b + delta
                HH_out[ra:ra + SVD_HH_BLOCK_SIZE, ca:ca + SVD_HH_BLOCK_SIZE] = (U_a * S_a) @ Vt_a
        else:
            if s_b - s_a < delta:
                S_b[0] = s_a + delta
                HH_out[rb:rb + SVD_HH_BLOCK_SIZE, cb:cb + SVD_HH_BLOCK_SIZE] = (U_b * S_b) @ Vt_b

    y_out = pywt.idwt2((LL, (LH, HL, HH_out)), wavelet)
    return y_out[: y_channel.shape[0], : y_channel.shape[1]]


def extract_watermark_svd_hh(y_channel, watermark_shape, wavelet=DEFAULT_WAVELET):
    """
    Blindly extract binary watermark from Y-channel by comparing top singular
    values of consecutive HH block pairs.
    """
    y = y_channel.astype(np.float64)
    n_bits = int(np.prod(watermark_shape))

    _, (_, _, HH) = pywt.dwt2(y, wavelet)

    nb_c = HH.shape[1] // SVD_HH_BLOCK_SIZE
    capacity = _svd_hh_capacity(HH.shape)
    if n_bits > capacity:
        raise ValueError(
            f"Requested {n_bits} bits but HH SVD capacity is only {capacity}."
        )

    bits = np.zeros(n_bits, dtype=np.uint8)
    for i in range(n_bits):
        _, S_a, _, _, _ = _hh_block_svd(HH, 2 * i, nb_c)
        _, S_b, _, _, _ = _hh_block_svd(HH, 2 * i + 1, nb_c)
        bits[i] = 1 if S_a[0] > S_b[0] else 0
    return bits.reshape(watermark_shape)


def embed_watermark_svd_hh_rgb(image, watermark, delta=ABSOLUTE_MARGIN_FLOOR, wavelet=DEFAULT_WAVELET):
    """
    Colorspace wrapper: embed watermark into HH subband via SVD on the Y (luma) channel.

    Parameters
    ----------
    image : np.ndarray, shape (H, W, 3), RGB float64
    watermark : np.ndarray of {0,1} — must have .size <= _svd_hh_capacity(HH.shape)
    delta : embedding margin (singular value separation enforced between pairs)
    """
    ycbcr = rgb_to_ycbcr(image)
    ycbcr[..., 0] = embed_watermark_svd_hh(ycbcr[..., 0], watermark, delta=delta, wavelet=wavelet)
    return ycbcr_to_rgb(ycbcr)


def extract_watermark_svd_hh_rgb(image, watermark_shape, wavelet=DEFAULT_WAVELET):
    """Colorspace wrapper: extract watermark from HH subband via SVD on the Y (luma) channel."""
    y = rgb_to_ycbcr(image)[..., 0]
    return extract_watermark_svd_hh(y, watermark_shape, wavelet=wavelet)


def sanity_check_svd_hh(img, wm, alpha):
    """
    Embed then extract with SVD-HH (no attacks) and return NC.
    alpha is accepted for API symmetry but is unused; embedding uses ABSOLUTE_MARGIN_FLOOR.
    """
    from .metrics import normalized_correlation
    watermarked = embed_watermark_svd_hh_rgb(img, wm)
    watermarked_u8 = np.clip(watermarked, 0, 255).astype(np.uint8)
    extracted = extract_watermark_svd_hh_rgb(watermarked_u8.astype(np.float64), wm.shape)
    return normalized_correlation(wm, extracted)


def embed_watermark_rgb(image, watermark, alpha=DEFAULT_ALPHA, wavelet=DEFAULT_WAVELET,
                        use_stable_positions=False, use_optimized_positions=False):
    """
    Colorspace wrapper around `embed_watermark`: convert RGB to YCbCr, embed
    the watermark into the Y (luma) channel only, then convert back to RGB.
    Cb/Cr are passed through unchanged.

    Parameters
    ----------
    image : np.ndarray, shape (H, W, 3), RGB

    Returns
    -------
    watermarked RGB image as a float64 array, same shape as `image`.
    """
    ycbcr = rgb_to_ycbcr(image)
    ycbcr[..., 0] = embed_watermark(ycbcr[..., 0], watermark, alpha=alpha, wavelet=wavelet,
                                     use_stable_positions=use_stable_positions,
                                     use_optimized_positions=use_optimized_positions)
    return ycbcr_to_rgb(ycbcr)


def extract_watermark_rgb(image, watermark_shape, wavelet=DEFAULT_WAVELET,
                          use_stable_positions=False, use_optimized_positions=False):
    """Colorspace wrapper around `extract_watermark`: extract from the Y (luma) channel of an RGB image."""
    y = rgb_to_ycbcr(image)[..., 0]
    return extract_watermark(y, watermark_shape, wavelet=wavelet,
                             use_stable_positions=use_stable_positions,
                             use_optimized_positions=use_optimized_positions)
