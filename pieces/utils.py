from __future__ import annotations
import os
from typing import Tuple
from PIL import Image, ImageDraw


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory for a file path if it doesn't exist."""
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)


def open_image(path: str) -> Image.Image:
    """Open an image from disk (lazy-loaded PIL Image)."""
    return Image.open(path)


def open_image_rgb(path: str) -> Image.Image:
    """Open an image and normalize to RGB (drops alpha, converts other modes)."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def save_image(path: str, img: Image.Image) -> None:
    """Save an image to disk, creating parent folders if needed."""
    ensure_parent_dir(path)
    img.save(path)


def save_image_rgb(path: str, img: Image.Image) -> None:
    """Save an image as true RGB (3 channels), creating parent folders if needed."""
    ensure_parent_dir(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(path)


def save_image_gray(path: str, img: Image.Image) -> None:
    """Save an image as true grayscale (mode 'L'), creating parent folders if needed."""
    ensure_parent_dir(path)
    if img.mode != "L":
        img = img.convert("L")
    img.save(path)


def clamp_crop_box(left: int, top: int, right: int, bottom: int, width: int, height: int) -> Tuple[int, int, int, int]:
    """Clamp a crop box to image bounds and enforce non-empty rectangle."""
    l = max(0, min(width, int(left)))
    t = max(0, min(height, int(top)))
    r = max(0, min(width, int(right)))
    b = max(0, min(height, int(bottom)))
    r = max(r, l + 1)
    b = max(b, t + 1)
    return l, t, r, b


def validate_crop_box(left: int, top: int, right: int, bottom: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Validate that a crop box fits fully inside image bounds."""
    left = int(left)
    top = int(top)
    right = int(right)
    bottom = int(bottom)

    if left < 0 or top < 0:
        raise ValueError("Crop box coordinates must be non-negative.")
    if right <= left or bottom <= top:
        raise ValueError("Crop box must define a non-empty rectangle.")
    if width < right or height < bottom:
        raise ValueError(
            f"Image too small for fixed crop box {(left, top, right, bottom)}: got size {(width, height)}"
        )

    return left, top, right, bottom


def apply_inscribed_circle_mask(img: Image.Image, background: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Keep only the centered inscribed circle; pixels outside are set to background."""
    img = img.convert("RGB")
    width, height = img.size
    diameter = min(width, height)
    left = (width - diameter) / 2
    top = (height - diameter) / 2
    right = left + diameter
    bottom = top + diameter

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((left, top, right, bottom), fill=255)

    bg = Image.new("RGB", (width, height), background)
    return Image.composite(img, bg, mask)


def translate_image(img: Image.Image, dx: int, dy: int, fill: int | Tuple[int, int, int] = 0) -> Image.Image:
    """Translate image by integer pixels (no wrap-around). Areas moved-in are filled.
    Works for RGB and L modes.
    """
    bg = Image.new(img.mode, img.size, color=fill)
    bg.paste(img, (dx, dy))
    return bg
