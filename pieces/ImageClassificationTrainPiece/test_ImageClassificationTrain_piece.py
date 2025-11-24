from typing import List, Tuple

import numpy as np
from domino.testing import piece_dry_run
from domino.testing.utils import skip_envs
from tensorflow import keras
from tensorflow.keras.layers import Conv2D


def run_piece(
    train_data_path: str,
    image_size: Tuple[int, int],
    validation_data_path: str | None,
    validation_split: float,
    num_layers: int,
    filters_per_layer: List[int],
    kernel_sizes: List[int],
    batch_size: int,
    epochs: int,
    early_stopping_patience: int,
    dropout_rate: float,
):
    return piece_dry_run(
        piece_name="ImageClassificationTrainPiece",
        input_data={
            'train_data_path': train_data_path,
            'image_size': image_size,
            'validation_data_path': validation_data_path,
            'validation_split': validation_split,
            'num_layers': num_layers,
            'filters_per_layer': filters_per_layer,
            'kernel_sizes': kernel_sizes,
            'batch_size': batch_size,
            'epochs': epochs,
            'early_stopping_patience': early_stopping_patience,
            'dropout_rate': dropout_rate,
        }
    )


@skip_envs('github')
def test_ImageClassificationTrainPiece():
    piece_kwargs = {
        'train_data_path': '/home/michal-skalican/Projects/SPICE/image_classification_pieces/sample_data',
        'image_size': [256, 256],
        'validation_data_path': None,
        'validation_split': 0.2,
        'num_layers': 1,
        'filters_per_layer': [64] * 1,
        'kernel_sizes': [3] * 1,
        'batch_size': 32,
        'epochs': 1,
        'early_stopping_patience': 300,
        'dropout_rate': 0.2,
    }
    output = run_piece(
        **piece_kwargs
    )
    m = keras.models.load_model(output['best_model_file_path'])
    conv2D_layers = [layer for layer in m.layers if isinstance(layer, Conv2D)]
    assert len(conv2D_layers) == piece_kwargs['num_layers']
    for layer, filters, kernel_size in zip(
        conv2D_layers,
        piece_kwargs['filters_per_layer'],
        piece_kwargs['kernel_sizes']
    ):
        assert layer.filters == filters
        assert layer.kernel_size == (kernel_size, kernel_size)
