import os
import numpy as np
import matplotlib.image as mpimg
from domino.testing import piece_dry_run
from domino.testing.utils import skip_envs


def _write_img(path, arr):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    mpimg.imsave(path, np.clip(arr, 0, 1))


def run_piece(input_image_path: str, output_image_path: str):
    return piece_dry_run(
        piece_name="ImageToGrayPiece",
        input_data={'input_image_path': input_image_path, 'output_image_path': output_image_path}
    )


@skip_envs('github')
def test_ImageToGrayPiece(tmp_path):
    inp = tmp_path / 'in.png'
    outp = tmp_path / 'out.png'
    img = np.zeros((5, 6, 3), dtype=float)
    img[..., 0] = 1.0  # red
    _write_img(str(inp), img)
    run_piece(str(inp), str(outp))
    out = mpimg.imread(str(outp))
    assert out.ndim == 2 or out.shape[-1] == 3
