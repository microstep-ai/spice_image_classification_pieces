from pydantic import BaseModel, Field


class InputModel(BaseModel):
    input_image_path: str = Field(title="input image path")
    output_image_path: str = Field(title="output image path")
    left: int = Field(title="left")
    top: int = Field(title="top")
    right: int = Field(title="right")
    bottom: int = Field(title="bottom")


class OutputModel(BaseModel):
    output_image_path: str
