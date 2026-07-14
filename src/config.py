"""Shared configuration for the DWT-DCT watermark robustness experiment."""

import os

RANDOM_SEED = 42

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ORIGINAL_IMAGES_DIR = os.path.join(DATA_DIR, "original_images")
PROCESSED_IMAGES_DIR = os.path.join(DATA_DIR, "processed_images")
WATERMARK_IMAGE_PATH = os.path.join(DATA_DIR, "watermark.png")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
WATERMARKED_DIR = os.path.join(RESULTS_DIR, "watermarked")
ATTACKED_TRADITIONAL_DIR = os.path.join(RESULTS_DIR, "attacked", "traditional")
ATTACKED_AI_DIR = os.path.join(RESULTS_DIR, "attacked", "ai_enhancement")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
METRICS_CSV_PATH = os.path.join(RESULTS_DIR, "metrics.csv")
SUMMARY_CSV_PATH = os.path.join(RESULTS_DIR, "summary_table.csv")

IMAGE_SIZE = (512, 512)  # (width, height) passed to cv2.resize; RGB

WATERMARK_SIZE = (32, 32)  # bit-grid shape; 1024 bits == capacity of the LL subband below

DEFAULT_ALPHA = 0.1
ALPHA_VALUES = [0.05, 0.1, 0.2, 0.3]

DWT_WAVELET = "haar"

SANITY_CHECK_NC_THRESHOLD = 0.99


def ensure_directories():
    for path in [
        ORIGINAL_IMAGES_DIR,
        PROCESSED_IMAGES_DIR,
        WATERMARKED_DIR,
        ATTACKED_TRADITIONAL_DIR,
        ATTACKED_AI_DIR,
        PLOTS_DIR,
    ]:
        os.makedirs(path, exist_ok=True)
