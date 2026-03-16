from pydantic import BaseModel, Field


class InputModel(BaseModel):
    input_image_path: str = Field(
        title="Input image path",
        description="Path to the source image file or folder of images on disk."
    )
    output_image_path: str = Field(
        title="Output image path",
        description="Where to save the processed image(s)."
    )
    left: int = Field(
        title="Left",
        description="Left coordinate (pixels) of the crop box."
    )
    top: int = Field(
        title="Top",
        description="Top coordinate (pixels) of the crop box."
    )
    right: int = Field(
        title="Right",
        description="Right coordinate (pixels, exclusive) of the crop box."
    )
    bottom: int = Field(
        title="Bottom",
        description="Bottom coordinate (pixels, exclusive) of the crop box."
    )
    background_color: list[int] = Field(
        title="Background color",
        description="RGB color used outside the inscribed circle.",
        default=[0, 0, 0],
        min_length=3,
        max_length=3,
    )


class OutputModel(BaseModel):
    output_image_path: str = Field(
        title="Output image path",
        description="Path to the saved output image(s)."
    )
