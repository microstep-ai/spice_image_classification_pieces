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
    raise


class ImageCircularCropPiece(ImageBasePiece):
    def process_image(self, file_path, output_path, input_data: InputModel):
        try:
            logger.info("Starting process_image")
            logger.info(f"file_path={file_path}, output_path={output_path}")
            logger.info(
                f"input_data: left={input_data.left}, top={input_data.top}, "
                f"right={input_data.right}, bottom={input_data.bottom}, "
                f"background_color={input_data.background_color}"
            )

            img = open_image_rgb(file_path)
            logger.info("Image opened successfully")

            width, height = img.size
            logger.info(f"Image size: width={width}, height={height}")

            left, top, right, bottom = validate_crop_box(
                input_data.left,
                input_data.top,
                input_data.right,
                input_data.bottom,
                width,
                height,
            )
            logger.info(
                f"Validated crop box: left={left}, top={top}, right={right}, bottom={bottom}"
            )

            cropped = img.crop((left, top, right, bottom))
            logger.info("Image cropped successfully")

            background_color = tuple(int(value) for value in input_data.background_color)
            logger.info(f"Background color parsed: {background_color}")

            output_image = apply_inscribed_circle_mask(
                cropped,
                background=background_color
            )
            logger.info("Circle mask applied successfully")

            save_image_rgb(output_path, output_image)
            logger.info(f"Output image saved successfully to {output_path}")

        except Exception as e:
            logger.exception("Error in process_image")
            raise

    def return_output_model(self, input_data):
        try:
            return OutputModel(output_image_path=input_data.output_image_path)
        except Exception as e:
            logger.exception("Error in return_output_model")
            raise