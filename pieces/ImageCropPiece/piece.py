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


def _save_image(path: str, img: Image.Image):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    img.save(path)


class ImageCropPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        img = _open_image(input_data.input_image_path)
        w, h = img.size
        l = max(0, min(w, int(input_data.left)))
        t = max(0, min(h, int(input_data.top)))
        r = max(0, min(w, int(input_data.right)))
        b = max(0, min(h, int(input_data.bottom)))
        r = max(r, l + 1)
        b = max(b, t + 1)
        out = img.crop((l, t, r, b))
        _save_image(input_data.output_image_path, out)
        return OutputModel(output_image_path=input_data.output_image_path)
