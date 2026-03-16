import os
import numpy as np
import matplotlib.image as mpimg
import pytest
from domino.testing import piece_dry_run
from domino.testing.utils import skip_envs


def _write_img(path, arr):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    mpimg.imsave(path, np.clip(arr, 0, 1))


def run_piece(input_image_path: str, output_image_path: str, left: int, top: int, right: int, bottom: int, background_color: list[int]):
    return piece_dry_run(
        piece_name="ImageCircularCropPiece",
        input_data={
            'input_image_path': input_image_path,
            'output_image_path': output_image_path,
            'left': left,
            'top': top,
            'right': right,
            'bottom': bottom,
            'background_color': background_color,
        }
    )


@skip_envs('github')
def test_ImageCircularCropPiece_single_image(tmp_path):
    inp = tmp_path / 'in.png'
    outp = tmp_path / 'out.png'

    img = np.ones((20, 20, 3), dtype=float)
    _write_img(str(inp), img)

    output = run_piece(str(inp), str(outp), 2, 2, 18, 18, [0, 0, 0])
    out = mpimg.imread(str(outp))

    assert out.shape[0] == 16 and out.shape[1] == 16
    assert np.allclose(out[0, 0, :3], [0, 0, 0], atol=1 / 255)
    assert output['output_image_path'] == str(outp)


@skip_envs('github')
def test_ImageCircularCropPiece_folder(tmp_path):
    inp_dir = tmp_path / 'input_images'
    out_dir = tmp_path / 'output_images'
    os.makedirs(inp_dir, exist_ok=True)

    img = np.ones((20, 20, 3), dtype=float)
    for index in range(3):
        _write_img(str(inp_dir / f'in_{index}.png'), img)

    output = run_piece(str(inp_dir), str(out_dir), 2, 2, 18, 18, [255, 0, 0])

    for index in range(3):
        out_path = out_dir / f'in_{index}.png'
        assert os.path.exists(out_path)
        out = mpimg.imread(str(out_path))
        assert out.shape[0] == 16 and out.shape[1] == 16

    assert output['output_image_path'] == str(out_dir)


@skip_envs('github')
def test_ImageCircularCropPiece_too_small_image(tmp_path):
    inp = tmp_path / 'in.png'
    outp = tmp_path / 'out.png'

    img = np.ones((10, 10, 3), dtype=float)
    _write_img(str(inp), img)

    with pytest.raises(Exception):
        run_piece(str(inp), str(outp), 0, 0, 20, 20, [0, 0, 0])
