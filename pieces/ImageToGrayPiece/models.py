from pydantic import BaseModel, Field


class InputModel(BaseModel):
    input_image_path: str = Field(title="input image path")
    output_image_path: str = Field(title="output image path")


class OutputModel(BaseModel):
    output_image_path: str
