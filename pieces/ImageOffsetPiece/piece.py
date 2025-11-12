import logging
import os

from domino.base_piece import BasePiece
import numpy as np
import matplotlib.image as mpimg

from .models import InputModel, OutputModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def _read_image(path: str):
    arr = mpimg.imread(path)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    return arr


def _save_image(path: str, arr):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    mpimg.imsave(path, np.clip(arr, 0.0, 1.0))


def _translate(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    # source slices
    if dx >= 0:
        xs_src = slice(0, w - dx)
        xs_dst = slice(dx, w)
    else:
        xs_src = slice(-dx, w)
        xs_dst = slice(0, w + dx)
    if dy >= 0:
        ys_src = slice(0, h - dy)
        ys_dst = slice(dy, h)
    else:
        ys_src = slice(-dy, h)
        ys_dst = slice(0, h + dy)
    out[ys_dst, xs_dst] = img[ys_src, xs_src]
    return out


class ImageOffsetPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        img = _read_image(input_data.input_image_path)
        out = _translate(img, input_data.dx, input_data.dy)
        _save_image(input_data.output_image_path, out)
        return OutputModel(output_image_path=input_data.output_image_path)
