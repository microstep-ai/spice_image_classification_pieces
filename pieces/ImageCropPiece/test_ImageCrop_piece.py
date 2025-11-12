import os
import numpy as np
import matplotlib.image as mpimg
from domino.testing import piece_dry_run
from domino.testing.utils import skip_envs


def _write_img(path, arr):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    mpimg.imsave(path, np.clip(arr, 0, 1))


def run_piece(input_image_path: str, output_image_path: str, left: int, top: int, right: int, bottom: int):
    return piece_dry_run(
        piece_name="ImageCropPiece",
        input_data={'input_image_path': input_image_path, 'output_image_path': output_image_path,
                    'left': left, 'top': top, 'right': right, 'bottom': bottom}
    )


@skip_envs('github')
def test_ImageCropPiece(tmp_path):
    inp = tmp_path / 'in.png'
    outp = tmp_path / 'out.png'
    img = np.zeros((20, 30, 3), dtype=float)
    _write_img(str(inp), img)
    run_piece(str(inp), str(outp), 5, 6, 25, 16)
    out = mpimg.imread(str(outp))
    assert out.shape[0] == 10 and out.shape[1] == 20
