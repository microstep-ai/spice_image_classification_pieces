from pydantic import BaseModel, Field


class InputModel(BaseModel):
    input_image_path: str = Field(title="input image path")
    output_image_path: str = Field(title="output image path")
    dx: int = Field(default=0, description="X translation in pixels (right is positive)")
    dy: int = Field(default=0, description="Y translation in pixels (down is positive)")


class OutputModel(BaseModel):
    output_image_path: str
