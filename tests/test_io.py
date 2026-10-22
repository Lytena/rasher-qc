"""Input validation: every unsupported input must be rejected with a reason."""

import numpy as np
import pytest
from PIL import Image

from rasherqc.io import InvalidInputError, load_valid_image, read_image, resize_square, validate_image


def good_image(rng, h=540, w=720):
    return rng.integers(40, 200, (h, w, 3), dtype=np.uint8)


def test_valid_rgb_passes(rng):
    rgb, res = validate_image(good_image(rng))
    assert res.ok and rgb.shape == (540, 720, 3) and not res.reasons


def test_rgba_is_accepted_with_warning(rng):
    img = np.dstack([good_image(rng), np.full((540, 720), 255, np.uint8)])
    rgb, res = validate_image(img)
    assert res.ok and rgb.shape[2] == 3 and "alpha" in res.warnings[0]


@pytest.mark.parametrize(
    "make, reason",
    [
        (lambda r: r.integers(0, 255, (540, 720), dtype=np.uint8), "greyscale"),
        (lambda r: np.repeat(r.integers(40, 200, (540, 720, 1), dtype=np.uint8), 3, 2), "channels identical"),
        (lambda r: r.integers(40, 200, (100, 120, 3), dtype=np.uint8), "too small"),
        (lambda r: np.full((540, 720, 3), 5, np.uint8) + r.integers(0, 3, (540, 720, 3), dtype=np.uint8), "exposure"),
        (lambda r: np.full((540, 720, 3), 255, np.uint8), "exposure"),
        (lambda r: r.random((540, 720, 3)).astype(np.float32), "dtype"),
        (lambda r: r.integers(0, 255, (540, 720, 2), dtype=np.uint8), "shape"),
    ],
)
def test_unsupported_inputs_rejected(rng, make, reason):
    rgb, res = validate_image(make(rng))
    assert rgb is None and not res.ok
    assert any(reason in m for m in res.reasons)


def test_clipped_fraction_rejected(rng):
    img = good_image(rng)
    img[:300] = 255  # >25 % of pixels saturated, mean still in range
    _, res = validate_image(img)
    assert not res.ok and any("clipped" in m for m in res.reasons)


def test_empty_input():
    _, res = validate_image(np.zeros((0, 0, 3), np.uint8))
    assert not res.ok


def test_read_missing_and_corrupt(tmp_path):
    with pytest.raises(InvalidInputError):
        read_image(tmp_path / "nope.jpg")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(InvalidInputError):
        read_image(bad)


def test_roundtrip_and_load_valid(tmp_path, rng):
    p = tmp_path / "a.png"
    img = good_image(rng)
    Image.fromarray(img).save(p)
    np.testing.assert_array_equal(read_image(p), img)
    np.testing.assert_array_equal(load_valid_image(p), img)
    Image.fromarray(img[..., 0]).save(tmp_path / "g.png")
    with pytest.raises(InvalidInputError, match="greyscale"):
        load_valid_image(tmp_path / "g.png")


def test_resize_square_shape_and_mean(rng):
    img = good_image(rng)
    out = resize_square(img, 64)
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8
    assert abs(out.mean() - img.mean()) < 1.0  # area averaging preserves the mean
