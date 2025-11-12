from typing import Literal
from pydantic import BaseModel, Field


class InputModel(BaseModel):
    input_image_path: str = Field(title="input image path")
    output_image_path: str = Field(title="output image path")
    rotation: Literal[0, 90, 180, 270] = Field(
        title="rotation",
        description="Rotation angle in degrees (must be one of 0, 90, 180, 270)",
        default=0,
    )


class OutputModel(BaseModel):
    output_image_path: str
