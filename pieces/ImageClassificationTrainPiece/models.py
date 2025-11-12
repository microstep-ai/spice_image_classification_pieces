from typing import List, Tuple

from pydantic import BaseModel, Field


class InputModel(BaseModel):
    """
    ImageClassificationTrainPiece Input Model
    """

    train_data_path: str = Field(
        title="train data path",
        description="Path to the train data.",
    )

    validation_data_path: str | None = Field(
        title="validation data path",
        description="Path to the validation data if validation data is pre-split",
        default=None,
    )

    validation_split: float | None = Field(
        title="validation split",
        description="Percentage of data to be used for validation if validation data is to be split.",
        default=None,
    )

    image_size: Tuple[int, int] = Field(
        title="image size",
        description="Size of the input image",
        default=(256, 256)
    )

    num_layers: int = Field(
        title="number of layers",
        default=3,
        description="Number of convolutional layers."
    )

    filters_per_layer: List[int] = Field(
        title="number of filters",
        default=[64, 64, 64],
        description="Number of filters for each convolutional layer."
    )

    kernel_sizes: List[Tuple[int, int]] = Field(
        title="kernel sizes",
        default=[(3,3), (3,3), (3,3)],
        description="Kernel size for each convolutional layer."
    )

    batch_size: int = Field(
        title="batch size",
        default=32,
        description="Batch size."
    )

    epochs: int = Field(
        title="epochs",
        default=500,
        description="Number of epochs."
    )

    early_stop_patience: int = Field(
        title="patience",
        default=500,
        description="Number of epochs with no improvement before stopping the training."
    )

    dropout_rate: float = Field(
        title="dropout rate",
        default=0,
        description="Dropout rate."
    )


class OutputModel(BaseModel):
    """
    ImageClassificationTrainPiece Output Model
    """
    best_model_file_path: str = Field(
        description="Path to the saved best model."
    )
    last_model_file_path: str = Field(
        description="Path to the saved last model."
    )
    config_path: str = Field(
        description="Path to the saved config."
    )
