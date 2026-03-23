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
    print(f"[ERROR] Could not import utils.py: {e}")
    raise


class ImageCircularCropPiece(ImageBasePiece):
    def process_image(self, file_path, output_path, input_data: InputModel):
        try:
            print("Starting process_image")
            print(f"file_path={file_path}, output_path={output_path}")
            print(
                f"input_data: left={input_data.left}, top={input_data.top}, "
                f"right={input_data.right}, bottom={input_data.bottom}, "
                f"background_color={input_data.background_color}"
            )

            print("Opening image...")
            img = open_image_rgb(file_path)
            print("Image opened successfully")

            width, height = img.size
            print(f"Image size: width={width}, height={height}")

            print("Validating crop box...")
            left, top, right, bottom = validate_crop_box(
                input_data.left,
                input_data.top,
                input_data.right,
                input_data.bottom,
                width,
                height,
            )
            print(f"Validated crop box: left={left}, top={top}, right={right}, bottom={bottom}")

            print("Cropping image...")
            cropped = img.crop((left, top, right, bottom))
            print("Image cropped successfully")

            print("Parsing background color...")
            background_color = tuple(int(value) for value in input_data.background_color)
            print(f"Background color parsed: {background_color}")

            print("Applying circle mask...")
            output_image = apply_inscribed_circle_mask(
                cropped,
                background=background_color
            )
            print("Circle mask applied successfully")

            print("Saving output image...")
            save_image_rgb(output_path, output_image)
            print(f"Output image saved successfully to {output_path}")

        except Exception as e:
            print(f"[ERROR] Error in process_image: {e}")
            raise

    def return_output_model(self, input_data):
        try:
            print("Creating output model...")
            return OutputModel(output_image_path=input_data.output_image_path)
        except Exception as e:
            print(f"[ERROR] Error in return_output_model: {e}")
            raise