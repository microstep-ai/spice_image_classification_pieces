import logging
import os

from domino.base_piece import BasePiece
from PIL import Image

from .models import InputModel, OutputModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


_ROTATE_MAP = {
    0: None,
    90: Image.ROTATE_90,
    180: Image.ROTATE_180,
    270: Image.ROTATE_270,
}


def _open_image_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    # Normalize to RGB to avoid alpha surprises across readers
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")
    return img


def _save_image_rgb(path: str, img: Image.Image):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(path)


class ImageRotatePiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        if input_data.rotation not in _ROTATE_MAP:
            raise ValueError("rotation must be one of 0, 90, 180, 270")
        img = _open_image_rgb(input_data.input_image_path)
        method = _ROTATE_MAP[input_data.rotation]
        out = img if method is None else img.transpose(method)
        _save_image_rgb(input_data.output_image_path, out)
        return OutputModel(output_image_path=input_data.output_image_path)
