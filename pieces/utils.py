from __future__ import annotations
import os
import logging
from typing import Tuple
from PIL import Image, ImageDraw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[]
)
logger = logging.getLogger(__name__)


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory for a file path if it doesn't exist."""
    try:
        logger.info(f"ensure_parent_dir called with path={path}")
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)
        logger.info(f"Parent directory ensured: {parent}")
    except Exception:
        logger.exception("Error in ensure_parent_dir")
        raise


def open_image(path: str) -> Image.Image:
    """Open an image from disk (lazy-loaded PIL Image)."""
    try:
        logger.info(f"open_image called with path={path}")
        img = Image.open(path)
        logger.info(f"Image opened successfully: mode={img.mode}, size={img.size}")
        return img
    except Exception:
        logger.exception(f"Error in open_image for path={path}")
        raise


def open_image_rgb(path: str) -> Image.Image:
    """Open an image and normalize to RGB (drops alpha, converts other modes)."""
    try:
        logger.info(f"open_image_rgb called with path={path}")
        img = Image.open(path)
        logger.info(f"Original image opened: mode={img.mode}, size={img.size}")

        if img.mode != "RGB":
            img = img.convert("RGB")
            logger.info("Image converted to RGB")

        return img
    except Exception:
        logger.exception(f"Error in open_image_rgb for path={path}")
        raise


def save_image(path: str, img: Image.Image) -> None:
    """Save an image to disk, creating parent folders if needed."""
    try:
        logger.info(f"save_image called with path={path}, mode={img.mode}, size={img.size}")
        ensure_parent_dir(path)
        img.save(path)
        logger.info(f"Image saved successfully to {path}")
    except Exception:
        logger.exception(f"Error in save_image for path={path}")
        raise


def save_image_rgb(path: str, img: Image.Image) -> None:
    """Save an image as true RGB (3 channels), creating parent folders if needed."""
    try:
        logger.info(f"save_image_rgb called with path={path}, mode={img.mode}, size={img.size}")
        ensure_parent_dir(path)

        if img.mode != "RGB":
            img = img.convert("RGB")
            logger.info("Image converted to RGB before saving")

        img.save(path)
        logger.info(f"RGB image saved successfully to {path}")
    except Exception:
        logger.exception(f"Error in save_image_rgb for path={path}")
        raise


def save_image_gray(path: str, img: Image.Image) -> None:
    """Save an image as true grayscale (mode 'L'), creating parent folders if needed."""
    try:
        logger.info(f"save_image_gray called with path={path}, mode={img.mode}, size={img.size}")
        ensure_parent_dir(path)

        if img.mode != "L":
            img = img.convert("L")
            logger.info("Image converted to grayscale (L) before saving")

        img.save(path)
        logger.info(f"Grayscale image saved successfully to {path}")
    except Exception:
        logger.exception(f"Error in save_image_gray for path={path}")
        raise


def clamp_crop_box(left: int, top: int, right: int, bottom: int, width: int, height: int) -> Tuple[int, int, int, int]:
    """Clamp a crop box to image bounds and enforce non-empty rectangle."""
    try:
        logger.info(
            f"clamp_crop_box called with left={left}, top={top}, right={right}, "
            f"bottom={bottom}, width={width}, height={height}"
        )

        l = max(0, min(width, int(left)))
        t = max(0, min(height, int(top)))
        r = max(0, min(width, int(right)))
        b = max(0, min(height, int(bottom)))
        r = max(r, l + 1)
        b = max(b, t + 1)

        logger.info(f"Clamped crop box: {(l, t, r, b)}")
        return l, t, r, b
    except Exception:
        logger.exception("Error in clamp_crop_box")
        raise


def validate_crop_box(left: int, top: int, right: int, bottom: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Validate that a crop box fits fully inside image bounds."""
    try:
        logger.info(
            f"validate_crop_box called with left={left}, top={top}, right={right}, "
            f"bottom={bottom}, width={width}, height={height}"
        )

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

        logger.info(f"Crop box validated successfully: {(left, top, right, bottom)}")
        return left, top, right, bottom
    except Exception:
        logger.exception("Error in validate_crop_box")
        raise


def apply_inscribed_circle_mask(img: Image.Image, background: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Keep only the centered inscribed circle; pixels outside are set to background."""
    try:
        logger.info(
            f"apply_inscribed_circle_mask called with mode={img.mode}, size={img.size}, background={background}"
        )

        img = img.convert("RGB")
        width, height = img.size
        diameter = min(width, height)
        left = (width - diameter) / 2
        top = (height - diameter) / 2
        right = left + diameter
        bottom = top + diameter

        logger.info(
            f"Calculated circle bounds: left={left}, top={top}, right={right}, bottom={bottom}, diameter={diameter}"
        )

        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((left, top, right, bottom), fill=255)

        bg = Image.new("RGB", (width, height), background)
        result = Image.composite(img, bg, mask)

        logger.info("Circle mask applied successfully")
        return result
    except Exception:
        logger.exception("Error in apply_inscribed_circle_mask")
        raise


def translate_image(img: Image.Image, dx: int, dy: int, fill: int | Tuple[int, int, int] = 0) -> Image.Image:
    """Translate image by integer pixels (no wrap-around). Areas moved-in are filled.
    Works for RGB and L modes.
    """
    try:
        logger.info(
            f"translate_image called with mode={img.mode}, size={img.size}, dx={dx}, dy={dy}, fill={fill}"
        )

        bg = Image.new(img.mode, img.size, color=fill)
        bg.paste(img, (dx, dy))

        logger.info("Image translated successfully")
        return bg
    except Exception:
        logger.exception("Error in translate_image")
        raise