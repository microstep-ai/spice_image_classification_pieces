from __future__ import annotations
import os
from typing import Tuple
from PIL import Image, ImageDraw


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory for a file path if it doesn't exist."""
    try:
        print(f"[ensure_parent_dir] path={path}")
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)
        print(f"[ensure_parent_dir] parent created/exists: {parent}")
    except Exception as e:
        print(f"[ERROR] ensure_parent_dir failed: {e}")
        raise


def open_image(path: str) -> Image.Image:
    """Open an image from disk (lazy-loaded PIL Image)."""
    try:
        print(f"[open_image] path={path}")
        img = Image.open(path)
        print(f"[open_image] opened successfully, mode={img.mode}, size={img.size}")
        return img
    except Exception as e:
        print(f"[ERROR] open_image failed for path={path}: {e}")
        raise


def open_image_rgb(path: str) -> Image.Image:
    """Open an image and normalize to RGB (drops alpha, converts other modes)."""
    try:
        print(f"[open_image_rgb] path={path}")
        img = Image.open(path)
        print(f"[open_image_rgb] original mode={img.mode}, size={img.size}")

        if img.mode != "RGB":
            print(f"[open_image_rgb] converting image from {img.mode} to RGB")
            img = img.convert("RGB")

        print(f"[open_image_rgb] returning mode={img.mode}, size={img.size}")
        return img
    except Exception as e:
        print(f"[ERROR] open_image_rgb failed for path={path}: {e}")
        raise


def save_image(path: str, img: Image.Image) -> None:
    """Save an image to disk, creating parent folders if needed."""
    try:
        print(f"[save_image] path={path}, mode={img.mode}, size={img.size}")
        ensure_parent_dir(path)
        img.save(path)
        print(f"[save_image] saved successfully to {path}")
    except Exception as e:
        print(f"[ERROR] save_image failed for path={path}: {e}")
        raise


def save_image_rgb(path: str, img: Image.Image) -> None:
    """Save an image as true RGB (3 channels), creating parent folders if needed."""
    try:
        print(f"[save_image_rgb] path={path}, mode={img.mode}, size={img.size}")
        ensure_parent_dir(path)

        if img.mode != "RGB":
            print(f"[save_image_rgb] converting image from {img.mode} to RGB before saving")
            img = img.convert("RGB")

        img.save(path)
        print(f"[save_image_rgb] saved successfully to {path}")
    except Exception as e:
        print(f"[ERROR] save_image_rgb failed for path={path}: {e}")
        raise


def save_image_gray(path: str, img: Image.Image) -> None:
    """Save an image as true grayscale (mode 'L'), creating parent folders if needed."""
    try:
        print(f"[save_image_gray] path={path}, mode={img.mode}, size={img.size}")
        ensure_parent_dir(path)

        if img.mode != "L":
            print(f"[save_image_gray] converting image from {img.mode} to L before saving")
            img = img.convert("L")

        img.save(path)
        print(f"[save_image_gray] saved successfully to {path}")
    except Exception as e:
        print(f"[ERROR] save_image_gray failed for path={path}: {e}")
        raise


def clamp_crop_box(left: int, top: int, right: int, bottom: int, width: int, height: int) -> Tuple[int, int, int, int]:
    """Clamp a crop box to image bounds and enforce non-empty rectangle."""
    try:
        print(
            f"[clamp_crop_box] input: left={left}, top={top}, right={right}, "
            f"bottom={bottom}, width={width}, height={height}"
        )

        l = max(0, min(width, int(left)))
        t = max(0, min(height, int(top)))
        r = max(0, min(width, int(right)))
        b = max(0, min(height, int(bottom)))
        r = max(r, l + 1)
        b = max(b, t + 1)

        print(f"[clamp_crop_box] output: {(l, t, r, b)}")
        return l, t, r, b
    except Exception as e:
        print(f"[ERROR] clamp_crop_box failed: {e}")
        raise


def validate_crop_box(left: int, top: int, right: int, bottom: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Validate that a crop box fits fully inside image bounds."""
    try:
        print(
            f"[validate_crop_box] input: left={left}, top={top}, right={right}, "
            f"bottom={bottom}, width={width}, height={height}"
        )

        left = int(left)
        top = int(top)
        right = int(right)
        bottom = int(bottom)

        print(
            f"[validate_crop_box] converted to int: left={left}, top={top}, "
            f"right={right}, bottom={bottom}"
        )

        if left < 0 or top < 0:
            raise ValueError("Crop box coordinates must be non-negative.")
        if right <= left or bottom <= top:
            raise ValueError("Crop box must define a non-empty rectangle.")
        if width < right or height < bottom:
            raise ValueError(
                f"Image too small for fixed crop box {(left, top, right, bottom)}: got size {(width, height)}"
            )

        print(f"[validate_crop_box] valid crop box: {(left, top, right, bottom)}")
        return left, top, right, bottom
    except Exception as e:
        print(f"[ERROR] validate_crop_box failed: {e}")
        raise


def apply_inscribed_circle_mask(img: Image.Image, background: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Keep only the centered inscribed circle; pixels outside are set to background."""
    try:
        print(f"[apply_inscribed_circle_mask] input mode={img.mode}, size={img.size}, background={background}")

        img = img.convert("RGB")
        width, height = img.size
        diameter = min(width, height)
        left = (width - diameter) / 2
        top = (height - diameter) / 2
        right = left + diameter
        bottom = top + diameter

        print(
            f"[apply_inscribed_circle_mask] circle bounds: "
            f"left={left}, top={top}, right={right}, bottom={bottom}, diameter={diameter}"
        )

        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((left, top, right, bottom), fill=255)

        bg = Image.new("RGB", (width, height), background)
        result = Image.composite(img, bg, mask)

        print("[apply_inscribed_circle_mask] mask applied successfully")
        return result
    except Exception as e:
        print(f"[ERROR] apply_inscribed_circle_mask failed: {e}")
        raise


def translate_image(img: Image.Image, dx: int, dy: int, fill: int | Tuple[int, int, int] = 0) -> Image.Image:
    """Translate image by integer pixels (no wrap-around). Areas moved-in are filled.
    Works for RGB and L modes.
    """
    try:
        print(f"[translate_image] mode={img.mode}, size={img.size}, dx={dx}, dy={dy}, fill={fill}")

        bg = Image.new(img.mode, img.size, color=fill)
        bg.paste(img, (dx, dy))

        print("[translate_image] translated successfully")
        return bg
    except Exception as e:
        print(f"[ERROR] translate_image failed: {e}")
        raise