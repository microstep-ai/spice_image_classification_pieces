import json
import os

import matplotlib.image as mpimg
import numpy as np
from domino.testing import piece_dry_run
from domino.testing.utils import skip_envs


def _write_img(path, arr):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    mpimg.imsave(path, np.clip(arr, 0, 1))


def _create_dataset(root_dir: str):
    os.makedirs(root_dir, exist_ok=True)
    for index in range(4):
        image = np.zeros((600, 600, 3), dtype=float)
        image[150:450, 150:450, :] = (index + 1) / 4
        image[200 + index:260 + index, 220:280, 0] = 1.0
        _write_img(os.path.join(root_dir, f'nozzle_{index}.png'), image)


def run_piece(train_data_path: str, model_output_path: str):
    return piece_dry_run(
        piece_name='SiameseImageSimilarityTrainPiece',
        input_data={
            'train_data_path': train_data_path,
            'model_output_path': model_output_path,
            'train_ratio': 0.5,
            'crop_left': 122,
            'crop_top': 32,
            'crop_right': 538,
            'crop_bottom': 448,
            'image_size': 64,
            'circle_background_color': [0, 0, 0],
            'cardinal_gap_deg': 2.0,
            'train_pairs_per_epoch': 4,
            'val_pairs_per_epoch': 4,
            'positive_prob': 0.5,
            'backbone': 'resnet18',
            'embedding_dim': 16,
            'batch_size': 2,
            'epochs': 1,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'cosine_margin': 0.2,
            'cosine_threshold': 0.75,
            'random_seed': 42,
        }
    )


@skip_envs('github')
def test_SiameseImageSimilarityTrainPiece(tmp_path):
    dataset_dir = tmp_path / 'dataset'
    output_dir = tmp_path / 'output'
    _create_dataset(str(dataset_dir))

    output = run_piece(str(dataset_dir), str(output_dir))

    assert os.path.exists(output['best_model_file_path'])
    assert os.path.exists(output['last_model_file_path'])
    assert os.path.exists(output['config_path'])
    assert os.path.exists(output['training_plot_file_path'])

    with open(output['config_path'], 'r', encoding='utf-8') as file:
        config = json.load(file)

    assert len(config['train_files']) == 2
    assert len(config['validation_files']) == 2
