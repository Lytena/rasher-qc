"""Detector wrapper tests (need the trained synthetic model; skipped otherwise)."""

import numpy as np
import pytest
from PIL import Image

from rasherqc import config
from rasherqc.metrics import Detections

WEIGHTS = config.MODEL_DIR / "synthetic" / "yolo_pretrained.pt"
pytestmark = pytest.mark.skipif(not WEIGHTS.is_file(), reason="trained model not available")


@pytest.fixture(scope="module")
def det():
    pytest.importorskip("ultralytics")
    from rasherqc.detector import Detector

    return Detector(WEIGHTS, imgsz=config.DEFAULT_IMGSZ)


def test_names_match_dataset(det):
    assert det.names == list(config.SYNTH_CLASSES)


def test_array_and_path_inputs_agree(det, tmp_path):
    """Guards the RGB/BGR conversion: an RGB array must give the same result as its file."""
    from rasherqc.synthetic import render

    img = render(np.random.default_rng(3), [0, 3]).image
    p = tmp_path / "t.png"
    Image.fromarray(img).save(p)
    a, b = det.predict([img])[0], det.predict([p])[0]
    assert isinstance(a, Detections)
    np.testing.assert_allclose(a.boxes, b.boxes, atol=0.5)
    np.testing.assert_allclose(a.scores, b.scores, atol=1e-3)


def test_boxes_in_pixel_units(det):
    from rasherqc.synthetic import render

    sc = render(np.random.default_rng(4), [0])
    d = det.predict([sc.image])[0]
    assert (d.boxes[:, 2] <= 720 + 1).all() and (d.boxes[:, 3] <= 540 + 1).all()
    assert (d.scores >= 0).all() and (d.scores <= 1).all()


def test_obvious_defect_is_found(det):
    """A loose rasher across the shingle should be detected with IoU >= 0.5."""
    from rasherqc.metrics import box_iou
    from rasherqc.synthetic import render

    sc = render(np.random.default_rng(21), [0])
    d = det.predict([sc.image])[0]
    sel = (d.classes == 0) & (d.scores >= 0.25)
    assert sel.any() and box_iou(sc.boxes[sc.classes == 0], d.boxes[sel]).max() >= 0.5
