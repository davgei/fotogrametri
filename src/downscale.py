"""
Downscale images to a maximum edge length, preserving aspect ratio and EXIF.

High-resolution images make SfM/MVS slow and memory-hungry without improving
results much. Resizing the longest edge to ~1600-2000 px is a good tradeoff.

Usage:
    py -3.14 src/downscale.py --input images --output images_small --max_size 1600
"""

import argparse
import shutil
from pathlib import Path

from PIL import Image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def downscale_image(src: Path, dst: Path, max_size: int) -> bool:
    with Image.open(src) as img:
        longest = max(img.size)
        if longest <= max_size:
            shutil.copy2(src, dst)
            return False

        scale = max_size / longest
        new_size = (round(img.size[0] * scale), round(img.size[1] * scale))
        resized = img.resize(new_size, Image.LANCZOS)

        exif = img.info.get("exif")
        if exif:
            resized.save(dst, quality=95, exif=exif)
        else:
            resized.save(dst, quality=95)
        return True


def downscale_directory(input_dir: Path, output_dir: Path, max_size: int) -> None:
    images = find_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No images found in '{input_dir}'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downscaling {len(images)} images to max edge {max_size} px...")

    resized_count = 0
    for src in images:
        dst = output_dir / src.name
        if downscale_image(src, dst, max_size):
            resized_count += 1

    print(f"  Resized {resized_count}, copied {len(images) - resized_count} (already small).")
    print(f"  Output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Downscale images for photogrammetry")
    parser.add_argument("--input", type=Path, default=Path("images"))
    parser.add_argument("--output", type=Path, default=Path("images_small"))
    parser.add_argument("--max_size", type=int, default=1600,
                        help="Maximum length of the longest edge in pixels (default: 1600)")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input directory not found: '{args.input}'")

    downscale_directory(args.input, args.output, args.max_size)


if __name__ == "__main__":
    main()
