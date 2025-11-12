import os
import numpy as np
import matplotlib.image as mpimg
from domino.testing import piece_dry_run
from domino.testing.utils import skip_envs


def _write_img(path, arr):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    mpimg.imsave(path, np.clip(arr, 0, 1))


def run_piece(input_image_path: str, output_image_path: str, factor: float):
    return piece_dry_run(
        piece_name="ImageEnhanceBrightnessPiece",
        input_data={
            'input_image_path': input_image_path,
            'output_image_path': output_image_path,
            'factor': factor,
        }
    )


@skip_envs('github')
def test_ImageEnhanceBrightnessPiece(tmp_path):
    inp = tmp_path / 'in.png'
    outp = tmp_path / 'out.png'
    img = np.full((10, 10, 3), 0.2, dtype=float)
    _write_img(str(inp), img)
    output = run_piece(str(inp), str(outp), 2.0)
    out = mpimg.imread(str(outp))
    assert out.mean() > img.mean()
    assert output['output_image_path'] == str(outp)
