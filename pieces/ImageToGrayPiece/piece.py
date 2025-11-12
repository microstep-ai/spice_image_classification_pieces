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


def _open_image(path: str) -> Image.Image:
    return Image.open(path)


def _save_image_gray(path: str, img: Image.Image):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if img.mode != 'L':
        img = img.convert('L')
    img.save(path)


class ImageToGrayPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        img = _open_image(input_data.input_image_path)
        gray = img.convert('L')
        _save_image_gray(input_data.output_image_path, gray)
        return OutputModel(output_image_path=input_data.output_image_path)
