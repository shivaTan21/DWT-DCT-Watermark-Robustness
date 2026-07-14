"""
Download additional images for DWT-DCT watermarking experiment.
Targets 90-100 total images combining existing Kodak + USC-SIPI Miscellaneous.

Usage:
    python download_images.py --output_dir data/original_images

Requirements:
    pip install requests Pillow
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from PIL import Image
import io

# --- Kodak remaining images (kodim21-24) ---
KODAK_URLS = {
    "kodim21.png": "https://r0k.us/graphics/kodak/kodak/kodim21.png",
    "kodim22.png": "https://r0k.us/graphics/kodak/kodak/kodim22.png",
    "kodim23.png": "https://r0k.us/graphics/kodak/kodak/kodim23.png",
    "kodim24.png": "https://r0k.us/graphics/kodak/kodak/kodim24.png",
}

# --- USC-SIPI Miscellaneous volume ---
# Standard benchmark images used widely in image processing literature
SIPI_URLS = {
    "sipi_4101.png": "https://sipi.usc.edu/database/misc/4.1.01.tiff",
    "sipi_4102.png": "https://sipi.usc.edu/database/misc/4.1.02.tiff",
    "sipi_4103.png": "https://sipi.usc.edu/database/misc/4.1.03.tiff",
    "sipi_4104.png": "https://sipi.usc.edu/database/misc/4.1.04.tiff",
    "sipi_4105.png": "https://sipi.usc.edu/database/misc/4.1.05.tiff",
    "sipi_4106.png": "https://sipi.usc.edu/database/misc/4.1.06.tiff",
    "sipi_4107.png": "https://sipi.usc.edu/database/misc/4.1.07.tiff",
    "sipi_4108.png": "https://sipi.usc.edu/database/misc/4.1.08.tiff",
    "sipi_4201.png": "https://sipi.usc.edu/database/misc/4.2.01.tiff",
    "sipi_4202.png": "https://sipi.usc.edu/database/misc/4.2.02.tiff",
    "sipi_4203.png": "https://sipi.usc.edu/database/misc/4.2.03.tiff",
    "sipi_4204.png": "https://sipi.usc.edu/database/misc/4.2.04.tiff",
    "sipi_4205.png": "https://sipi.usc.edu/database/misc/4.2.05.tiff",
    "sipi_4206.png": "https://sipi.usc.edu/database/misc/4.2.06.tiff",
    "sipi_4207.png": "https://sipi.usc.edu/database/misc/4.2.07.tiff",
    "sipi_4208.png": "https://sipi.usc.edu/database/misc/4.2.08.tiff",
    "sipi_5109.png": "https://sipi.usc.edu/database/misc/5.1.09.tiff",
    "sipi_5110.png": "https://sipi.usc.edu/database/misc/5.1.10.tiff",
    "sipi_5111.png": "https://sipi.usc.edu/database/misc/5.1.11.tiff",
    "sipi_5112.png": "https://sipi.usc.edu/database/misc/5.1.12.tiff",
    "sipi_5113.png": "https://sipi.usc.edu/database/misc/5.1.13.tiff",
    "sipi_5114.png": "https://sipi.usc.edu/database/misc/5.1.14.tiff",
    "sipi_5208.png": "https://sipi.usc.edu/database/misc/5.2.08.tiff",
    "sipi_5209.png": "https://sipi.usc.edu/database/misc/5.2.09.tiff",
    "sipi_5210.png": "https://sipi.usc.edu/database/misc/5.2.10.tiff",
    "sipi_5301.png": "https://sipi.usc.edu/database/misc/5.3.01.tiff",
    "sipi_5302.png": "https://sipi.usc.edu/database/misc/5.3.02.tiff",
    "sipi_7101.png": "https://sipi.usc.edu/database/misc/7.1.01.tiff",
    "sipi_7102.png": "https://sipi.usc.edu/database/misc/7.1.02.tiff",
    "sipi_7103.png": "https://sipi.usc.edu/database/misc/7.1.03.tiff",
    "sipi_7104.png": "https://sipi.usc.edu/database/misc/7.1.04.tiff",
    "sipi_7105.png": "https://sipi.usc.edu/database/misc/7.1.05.tiff",
    "sipi_7106.png": "https://sipi.usc.edu/database/misc/7.1.06.tiff",
    "sipi_7107.png": "https://sipi.usc.edu/database/misc/7.1.07.tiff",
    "sipi_7108.png": "https://sipi.usc.edu/database/misc/7.1.08.tiff",
    "sipi_7109.png": "https://sipi.usc.edu/database/misc/7.1.09.tiff",
    "sipi_7110.png": "https://sipi.usc.edu/database/misc/7.1.10.tiff",
    "sipi_7201.png": "https://sipi.usc.edu/database/misc/7.2.01.tiff",
    "sipi_boat.png": "https://sipi.usc.edu/database/misc/boat.512.tiff",
    "sipi_elaine.png": "https://sipi.usc.edu/database/misc/elaine.512.tiff",
    "sipi_lena.png": "https://sipi.usc.edu/database/misc/lena.tiff",
    "sipi_mandril.png": "https://sipi.usc.edu/database/misc/mandril_color.tiff",
    "sipi_peppers.png": "https://sipi.usc.edu/database/misc/peppers_color.tiff",
    "sipi_tulips.png": "https://sipi.usc.edu/database/misc/tulips.tiff",
    "sipi_baboon.png": "https://sipi.usc.edu/database/misc/baboon.tiff",
    "sipi_house.png": "https://sipi.usc.edu/database/misc/house.tiff",
    "sipi_couple.png": "https://sipi.usc.edu/database/misc/couple.tiff",
}


def download_and_convert(url, dest_path, timeout=30):
    """Download image from URL, convert to RGB PNG, save to dest_path."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        # Convert to RGB (handles grayscale, RGBA, palette modes)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(dest_path, "PNG")
        return True, f"OK ({img.size[0]}x{img.size[1]})"
    except requests.exceptions.HTTPError as e:
        return False, f"HTTP {e.response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Connection error"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Download images for watermarking experiment")
    parser.add_argument(
        "--output_dir",
        default="data/original_images",
        help="Directory to save images (default: data/original_images)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip files that already exist (default: True)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_urls = {**KODAK_URLS, **SIPI_URLS}
    total = len(all_urls)
    success = 0
    skipped = 0
    failed = []

    print(f"Target directory: {output_dir.resolve()}")
    print(f"Images to download: {total}")
    print(f"Skip existing: {args.skip_existing}")
    print("-" * 60)

    for i, (filename, url) in enumerate(all_urls.items(), 1):
        dest = output_dir / filename
        prefix = f"[{i:3d}/{total}] {filename:<30}"

        if args.skip_existing and dest.exists():
            print(f"{prefix} SKIP (exists)")
            skipped += 1
            continue

        ok, msg = download_and_convert(url, dest)
        if ok:
            print(f"{prefix} {msg}")
            success += 1
        else:
            print(f"{prefix} FAILED: {msg}")
            failed.append((filename, url, msg))

    # Final summary
    print("\n" + "=" * 60)
    print(f"Downloaded:  {success}")
    print(f"Skipped:     {skipped}")
    print(f"Failed:      {len(failed)}")

    if failed:
        print("\nFailed downloads:")
        for fname, url, reason in failed:
            print(f"  {fname}: {reason}")

    # Count total images now in output_dir
    existing = sorted(output_dir.glob("*.png")) + sorted(output_dir.glob("*.jpg"))
    print(f"\nTotal images in {output_dir}: {len(existing)}")
    for p in existing:
        print(f"  {p.name}")

    if len(existing) < 50:
        print("\nWARNING: Fewer than 50 images. Some downloads may have failed.")
        print("Try running again — USC-SIPI can be slow to respond.")
        sys.exit(1)
    else:
        print(f"\nReady to run experiment with {len(existing)} images.")


if __name__ == "__main__":
    main()