from typing import Literal

from pydantic import BaseModel, Field


class InputModel(BaseModel):
    model_config = {
        'protected_namespaces': ()
    }

    train_data_path: str = Field(
        title="Train data path",
        description="Path to a folder containing training images. Each filename must be unique."
    )
    model_output_path: str = Field(
        title="Model output path",
        description="Path where the trained model artifacts will be saved."
    )
    train_ratio: float = Field(
        title="Train ratio",
        description="Fraction of images used for training; the remainder is used for validation.",
        default=0.8,
    )
    crop_left: int = Field(title="Crop left", default=122)
    crop_top: int = Field(title="Crop top", default=32)
    crop_right: int = Field(title="Crop right", default=538)
    crop_bottom: int = Field(title="Crop bottom", default=448)
    image_size: int = Field(
        title="Image size",
        description="Square image size after resize.",
        default=224,
    )
    circle_background_color: list[int] = Field(
        title="Circle background color",
        description="RGB color used outside the inscribed circle.",
        default=[0, 0, 0],
        min_length=3,
        max_length=3,
    )
    cardinal_gap_deg: float = Field(
        title="Cardinal gap",
        description="Excluded angular gap around 0, 90, 180, 270 degrees.",
        default=2.0,
    )
    train_pairs_per_epoch: int = Field(title="Train pairs per epoch", default=4000)
    val_pairs_per_epoch: int = Field(title="Validation pairs per epoch", default=1000)
    positive_prob: float = Field(title="Positive pair probability", default=0.5)
    backbone: Literal['resnet18', 'resnet50'] = Field(title="Backbone", default='resnet18')
    embedding_dim: int = Field(title="Embedding dimension", default=128)
    batch_size: int = Field(title="Batch size", default=32)
    epochs: int = Field(title="Epochs", default=10)
    learning_rate: float = Field(title="Learning rate", default=1e-4)
    weight_decay: float = Field(title="Weight decay", default=1e-4)
    cosine_margin: float = Field(title="Cosine margin", default=0.2)
    cosine_threshold: float = Field(title="Cosine threshold", default=0.75)
    random_seed: int = Field(title="Random seed", default=42)


class OutputModel(BaseModel):
    best_model_file_path: str = Field(description="Path to the best model checkpoint.")
    last_model_file_path: str = Field(description="Path to the final model checkpoint.")
    config_path: str = Field(description="Path to the saved configuration JSON.")
    training_plot_file_path: str = Field(description="Path to the saved training plot.")
