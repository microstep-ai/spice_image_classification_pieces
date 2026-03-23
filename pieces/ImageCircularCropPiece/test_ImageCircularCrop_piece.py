import os
import numpy as np
import matplotlib.image as mpimg
import pytest
from PIL import Image
from domino.testing import piece_dry_run
from domino.testing.utils import skip_envs


def _write_img(path, arr):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mpimg.imsave(path, np.clip(arr, 0, 1))


def _write_pil_image(path, size=(20, 20), mode="RGB", color=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if color is None:
        if mode == "RGB":
            color = (255, 255, 255)
        elif mode == "RGBA":
            color = (255, 255, 255, 255)
        elif mode == "L":
            color = 255
        else:
            color = 0

    img = Image.new(mode, size, color)
    img.save(path)


def run_piece(
    input_image_path: str,
    output_image_path: str,
    left,
    top,
    right,
    bottom,
    background_color,
):
    return piece_dry_run(
        piece_name="ImageCircularCropPiece",
        input_data={
            "input_image_path": input_image_path,
            "output_image_path": output_image_path,
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "background_color": background_color,
        },
    )


@skip_envs("github")
def test_ImageCircularCropPiece_basic(tmp_path):
    inp = tmp_path / "in.png"
    outp = tmp_path / "out.png"

    img = np.ones((20, 20, 3), dtype=float)
    _write_img(str(inp), img)

    run_piece(str(inp), str(outp), 2, 2, 18, 18, [0, 0, 0])

    assert outp.exists()

    out = mpimg.imread(str(outp))
    assert out.shape[0] == 16
    assert out.shape[1] == 16
    assert np.allclose(out[0, 0, :3], [0, 0, 0], atol=1 / 255)


@skip_envs("github")
def test_ImageCircularCropPiece_folder(tmp_path):
    inp_dir = tmp_path / "input_images"
    out_dir = tmp_path / "output_images"
    os.makedirs(inp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    img = np.ones((20, 20, 3), dtype=float)

    for i in range(10):
        _write_img(str(inp_dir / f"in_{i}.png"), img)

    run_piece(str(inp_dir), str(out_dir), 2, 2, 18, 18, [255, 0, 0])

    for i in range(10):
        out_path = out_dir / f"in_{i}.png"
        assert out_path.exists()
        out = mpimg.imread(str(out_path))
        assert out.shape[0] == 16
        assert out.shape[1] == 16


@skip_envs("github")
def test_ImageCircularCropPiece_too_small_image(tmp_path):
    inp = tmp_path / "in.png"
    outp = tmp_path / "out.png"

    img = np.ones((10, 10, 3), dtype=float)
    _write_img(str(inp), img)

    with pytest.raises(Exception, match="Image too small for fixed crop box"):
        run_piece(str(inp), str(outp), 0, 0, 20, 20, [0, 0, 0])


@skip_envs("github")
@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_ImageCircularCropPiece_supported_input_modes(tmp_path, mode):
    inp = tmp_path / f"in_{mode}.png"
    outp = tmp_path / f"out_{mode}.png"

    _write_pil_image(str(inp), size=(20, 20), mode=mode)

    run_piece(str(inp), str(outp), 2, 2, 18, 18, [0, 0, 0])

    assert outp.exists()
    out = Image.open(outp)
    assert out.size == (16, 16)


@skip_envs("github")
def test_ImageCircularCropPiece_exact_image_bounds_crop(tmp_path):
    inp = tmp_path / "in.png"
    outp = tmp_path / "out.png"

    _write_pil_image(str(inp), size=(20, 20), mode="RGBA")

    run_piece(str(inp), str(outp), 0, 0, 20, 20, [0, 0, 0])

    assert outp.exists()
    out = Image.open(outp)
    assert out.size == (20, 20)


@skip_envs("github")
@pytest.mark.parametrize(
    "left,top,right,bottom,expected_message",
    [
        (-1, 0, 10, 10, "Crop box coordinates must be non-negative"),
        (0, -1, 10, 10, "Crop box coordinates must be non-negative"),
        (10, 0, 10, 10, "Crop box must define a non-empty rectangle"),
        (0, 10, 10, 10, "Crop box must define a non-empty rectangle"),
        (15, 0, 10, 10, "Crop box must define a non-empty rectangle"),
        (0, 15, 10, 10, "Crop box must define a non-empty rectangle"),
    ],
)
def test_ImageCircularCropPiece_invalid_crop_box(
    tmp_path, left, top, right, bottom, expected_message
):
    inp = tmp_path / "in.png"
    outp = tmp_path / "out.png"

    _write_pil_image(str(inp), size=(20, 20), mode="RGB")

    with pytest.raises(Exception, match=expected_message):
        run_piece(str(inp), str(outp), left, top, right, bottom, [0, 0, 0])


@skip_envs("github")
def test_ImageCircularCropPiece_crop_box_as_numeric_strings(tmp_path):
    inp = tmp_path / "in.png"
    outp = tmp_path / "out.png"

    _write_pil_image(str(inp), size=(20, 20), mode="RGB")

    run_piece(str(inp), str(outp), "2", "2", "18", "18", [0, 0, 0])

    assert outp.exists()
    out = Image.open(outp)
    assert out.size == (16, 16)


@skip_envs("github")
@pytest.mark.parametrize(
    "background_color",
    [
        None,
        [],
        [0],
        [0, 0],
        [0, 0, 0, 0],
        ["a", 0, 0],
        [None, 0, 0],
    ],
)
def test_ImageCircularCropPiece_invalid_background_color(tmp_path, background_color):
    inp = tmp_path / "in.png"
    outp = tmp_path / "out.png"

    _write_pil_image(str(inp), size=(20, 20), mode="RGB")

    with pytest.raises(Exception):
        run_piece(str(inp), str(outp), 2, 2, 18, 18, background_color)


@skip_envs("github")
def test_ImageCircularCropPiece_background_color_string_numbers(tmp_path):
    inp = tmp_path / "in.png"
    outp = tmp_path / "out.png"

    _write_pil_image(str(inp), size=(20, 20), mode="RGB")

    run_piece(str(inp), str(outp), 2, 2, 18, 18, ["255", "0", "0"])

    assert outp.exists()
    out = Image.open(outp)
    assert out.size == (16, 16)


@skip_envs("github")
def test_ImageCircularCropPiece_creates_output_directory(tmp_path):
    inp = tmp_path / "in.png"
    outp = tmp_path / "nested" / "deep" / "folder" / "out.png"

    _write_pil_image(str(inp), size=(20, 20), mode="RGB")

    run_piece(str(inp), str(outp), 2, 2, 18, 18, [0, 0, 0])

    assert outp.exists()


@skip_envs("github")
def test_ImageCircularCropPiece_nonexistent_input_path(tmp_path):
    inp = tmp_path / "does_not_exist.png"
    outp = tmp_path / "out.png"

    with pytest.raises(Exception):
        run_piece(str(inp), str(outp), 2, 2, 18, 18, [0, 0, 0])


@skip_envs("github")
def test_ImageCircularCropPiece_folder_with_non_image_file(tmp_path):
    inp_dir = tmp_path / "input_images"
    out_dir = tmp_path / "output_images"
    os.makedirs(inp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    _write_pil_image(str(inp_dir / "good.png"), size=(20, 20), mode="RGB")

    with open(inp_dir / "note.txt", "w", encoding="utf-8") as f:
        f.write("this is not an image")

    run_piece(str(inp_dir), str(out_dir), 2, 2, 18, 18, [0, 0, 0])

    out_path = out_dir / "good.png"
    assert out_path.exists()

    out = Image.open(out_path)
    assert out.size == (16, 16)

    txt_out = out_dir / "note.txt"
    assert not txt_out.exists()


@skip_envs("github")
def test_ImageCircularCropPiece_folder_with_non_image_file_logs_warning(tmp_path, caplog):
    inp_dir = tmp_path / "input_images"
    out_dir = tmp_path / "output_images"
    os.makedirs(inp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    _write_pil_image(str(inp_dir / "good.png"), size=(20, 20), mode="RGB")

    with open(inp_dir / "note.txt", "w", encoding="utf-8") as f:
        f.write("this is not an image")

    run_piece(str(inp_dir), str(out_dir), 2, 2, 18, 18, [0, 0, 0])

    assert (out_dir / "good.png").exists()
    assert "Could not process file note.txt" in caplog.text


@skip_envs("github")
def test_ImageCircularCropPiece_empty_input_folder(tmp_path):
    inp_dir = tmp_path / "input_images"
    out_dir = tmp_path / "output_images"
    os.makedirs(inp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    result = run_piece(str(inp_dir), str(out_dir), 2, 2, 18, 18, [0, 0, 0])

    assert result is not None


@skip_envs("github")
@pytest.mark.parametrize(
    "size,left,top,right,bottom,expected_size",
    [
        ((20, 20), 2, 2, 18, 18, (16, 16)),
        ((30, 20), 5, 2, 25, 18, (20, 16)),
        ((20, 30), 2, 5, 18, 25, (16, 20)),
    ],
)
def test_ImageCircularCropPiece_various_image_sizes(
    tmp_path, size, left, top, right, bottom, expected_size
):
    inp = tmp_path / f"in_{size[0]}x{size[1]}.png"
    outp = tmp_path / f"out_{size[0]}x{size[1]}.png"

    _write_pil_image(str(inp), size=size, mode="RGB")

    run_piece(str(inp), str(outp), left, top, right, bottom, [0, 0, 0])

    out = Image.open(outp)
    assert out.size == expected_size


@skip_envs("github")
def test_ImageCircularCropPiece_output_corner_matches_background(tmp_path):
    inp = tmp_path / "in.png"
    outp = tmp_path / "out.png"

    _write_pil_image(str(inp), size=(20, 20), mode="RGB", color=(255, 255, 255))

    run_piece(str(inp), str(outp), 2, 2, 18, 18, [255, 0, 0])

    out = mpimg.imread(str(outp))
    assert np.allclose(out[0, 0, :3], [1.0, 0.0, 0.0], atol=1 / 255)


@skip_envs("github")
def test_ImageCircularCropPiece_folder_many_files(tmp_path):
    inp_dir = tmp_path / "input_images"
    out_dir = tmp_path / "output_images"
    os.makedirs(inp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    for i in range(50):
        _write_pil_image(str(inp_dir / f"in_{i}.png"), size=(20, 20), mode="RGB")

    run_piece(str(inp_dir), str(out_dir), 2, 2, 18, 18, [0, 0, 0])

    outputs = list(out_dir.glob("*.png"))
    assert len(outputs) == 50


@skip_envs("github")
def test_ImageCircularCropPiece_background_color_none(tmp_path):
    inp = tmp_path / "in.png"
    outp = tmp_path / "out.png"

    _write_pil_image(str(inp), size=(20, 20), mode="RGB")

    with pytest.raises(Exception):
        run_piece(str(inp), str(outp), 2, 2, 18, 18, None)


@skip_envs("github")
def test_ImageCircularCropPiece_background_color_wrong_length(tmp_path):
    inp = tmp_path / "in.png"
    outp = tmp_path / "out.png"

    _write_pil_image(str(inp), size=(20, 20), mode="RGB")

    with pytest.raises(Exception):
        run_piece(str(inp), str(outp), 2, 2, 18, 18, [255, 0])


@skip_envs("github")
def test_ImageCircularCropPiece_non_numeric_crop_values(tmp_path):
    inp = tmp_path / "in.png"
    outp = tmp_path / "out.png"

    _write_pil_image(str(inp), size=(20, 20), mode="RGB")

    with pytest.raises(Exception):
        run_piece(str(inp), str(outp), "abc", 2, 18, 18, [0, 0, 0])