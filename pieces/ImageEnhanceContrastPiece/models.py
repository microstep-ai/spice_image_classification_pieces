from pydantic import BaseModel, Field


class InputModel(BaseModel):
    input_image_path: str = Field(title="input image path")
    output_image_path: str = Field(title="output image path")
    factor: float = Field(title="contrast factor", description=">1 increases contrast, <1 decreases", default=1.5)


class OutputModel(BaseModel):
    output_image_path: str = Field(description="Path to the saved output image.")
