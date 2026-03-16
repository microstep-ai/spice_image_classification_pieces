import logging

from pieces.ImageProcessingBasePiece import ImageBasePiece
from .models import InputModel, OutputModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[]
)
logger = logging.getLogger(__name__)

try:
    try:
        from ..utils import open_image_rgb, save_image_rgb, validate_crop_box, apply_inscribed_circle_mask
    except ImportError:  # pragma: no cover
        from pieces.utils import open_image_rgb, save_image_rgb, validate_crop_box, apply_inscribed_circle_mask
except Exception as e:
    logger.exception(f"Could not import utils.py: {e}")
    raise e


class ImageCircularCropPiece(ImageBasePiece):
    def process_image(self, file_path, output_path, input_data: InputModel):
        img = open_image_rgb(file_path)
        width, height = img.size
        left, top, right, bottom = validate_crop_box(
            input_data.left,
            input_data.top,
            input_data.right,
            input_data.bottom,
            width,
            height,
        )
        cropped = img.crop((left, top, right, bottom))
        background_color = tuple(int(value) for value in input_data.background_color)
        output_image = apply_inscribed_circle_mask(cropped, background=background_color)
        save_image_rgb(output_path, output_image)

    def return_output_model(self, input_data):
        return OutputModel(output_image_path=input_data.output_image_path)
