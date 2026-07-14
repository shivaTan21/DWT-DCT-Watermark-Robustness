"""Classical (non-AI) watermark-removal attacks used as a robustness baseline."""

import cv2
import numpy as np

from . import config


def jpeg_compression(image, quality=50):
    """JPEG-compress an RGB image. cv2's codecs assume BGR channel order, so
    we convert to BGR for the encode/decode round trip and back to RGB."""
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise RuntimeError("JPEG encoding failed")
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    decoded = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    return decoded.astype(np.float64)


def gaussian_noise(image, sigma=10.0, seed=config.RANDOM_SEED):
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, sigma, image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0, 255)


def median_filtering(image, ksize=3):
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.medianBlur(image_u8, ksize).astype(np.float64)


def gaussian_blur(image, ksize=5, sigma=1.0):
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(image_u8, (ksize, ksize), sigma).astype(np.float64)


def crop_resize(image, crop_fraction=0.8, target_size=config.IMAGE_SIZE):
    h, w = image.shape[:2]
    ch, cw = int(h * crop_fraction), int(w * crop_fraction)
    top, left = (h - ch) // 2, (w - cw) // 2
    cropped = image[top:top + ch, left:left + cw]
    image_u8 = np.clip(cropped, 0, 255).astype(np.uint8)
    resized = cv2.resize(image_u8, target_size, interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float64)


def get_traditional_attacks(seed=config.RANDOM_SEED):
    """Return {attack_name: callable(image) -> attacked_image}."""
    return {
        "jpeg_compression_q50": lambda img: jpeg_compression(img, quality=50),
        "gaussian_noise_s10": lambda img: gaussian_noise(img, sigma=10.0, seed=seed),
        "median_filter_3x3": lambda img: median_filtering(img, ksize=3),
        "gaussian_blur_5x5": lambda img: gaussian_blur(img, ksize=5, sigma=1.0),
        "crop_resize_80pct": lambda img: crop_resize(img, crop_fraction=0.8),
    }
