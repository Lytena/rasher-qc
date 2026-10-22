"""Dataset loading (YOLO and COCO), tray labels, leakage check and generator."""

import json
import shutil

import numpy as np
import pytest
from PIL import Image

from rasherqc import config
from rasherqc.dataset import (
    class_statistics,
    cross_split_duplicates,
    dhash,
    is_defect_name,
    load_dataset,
    tray_fail_label,
)
from rasherqc.io import InvalidInputError
from rasherqc.synthetic import H, W, render


def test_yolo_load(tiny_dataset):
    ds = load_dataset(tiny_dataset)
    assert ds.names == list(config.SYNTH_CLASSES)
    assert [len(ds.split(s)) for s in ("train", "valid", "test")] == [6, 3, 3]
    for r in ds.records:
        assert (r.width, r.height) == (W, H)
        assert r.boxes.shape == (len(r.classes), 4)
        assert (r.boxes[:, 2] > r.boxes[:, 0]).all() and (r.boxes[:, 3] > r.boxes[:, 1]).all()
    assert ds.yolo_yaml.is_file()
    assert ds.defect_ids == list(range(5))


def test_yolo_labels_roundtrip_pixels(tiny_dataset):
    """Boxes read back from normalised YOLO text equal the rendered ones (<0.01 px)."""
    rng = np.random.default_rng(7)
    sc = render(rng)  # first image of the train split uses the same first draw
    ds = load_dataset(tiny_dataset)
    r = ds.split("train")[0]
    np.testing.assert_allclose(r.boxes, sc.boxes, atol=0.01)
    np.testing.assert_array_equal(r.classes, sc.classes)


def test_coco_export_gives_same_annotations(tiny_dataset, tmp_path):
    """Convert the tiny YOLO set to COCO and check the COCO loader agrees."""
    ds = load_dataset(tiny_dataset)
    root = tmp_path / "coco"
    for split in ("train", "valid", "test"):
        d = root / split
        d.mkdir(parents=True)
        imgs, anns = [], []
        for i, r in enumerate(ds.split(split)):
            shutil.copy(r.path, d / r.path.name)
            imgs.append({"id": i, "file_name": r.path.name, "width": r.width, "height": r.height})
            for b, c in zip(r.boxes, r.classes):
                anns.append({"id": len(anns), "image_id": i, "category_id": int(c) + 1,
                             "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]]})
        cats = [{"id": 0, "name": "rasher-dataset"}] + [{"id": k + 1, "name": n} for k, n in enumerate(ds.names)]
        (d / "_annotations.coco.json").write_text(json.dumps({"images": imgs, "annotations": anns, "categories": cats}))
    cd = load_dataset(root)
    used = sorted({int(c) for r in ds.records for c in r.classes})
    assert cd.names == [ds.names[u] for u in used]  # dummy super-category dropped
    a = sorted((r.path.name, tuple(np.round(r.boxes.ravel(), 2))) for r in ds.records)
    b = sorted((r.path.name, tuple(np.round(r.boxes.ravel(), 2))) for r in cd.records)
    assert a == b


def test_bad_layouts(tmp_path):
    with pytest.raises(InvalidInputError):
        load_dataset(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(InvalidInputError):
        load_dataset(tmp_path / "empty")


def test_label_with_out_of_range_class(tiny_dataset, tmp_path):
    root = tmp_path / "bad"
    shutil.copytree(tiny_dataset, root)
    lab = sorted((root / "train" / "labels").glob("*.txt"))[0]
    lab.write_text("9 0.5 0.5 0.1 0.1")
    with pytest.raises(InvalidInputError, match="class id 9"):
        load_dataset(root)


def test_defect_name_rules():
    assert is_defect_name("loose-meat") and is_defect_name("wrinkle")
    assert not is_defect_name("Rasher") and not is_defect_name("good")


def test_tray_label_and_stats(tiny_dataset):
    ds = load_dataset(tiny_dataset)
    for r in ds.records:
        assert tray_fail_label(r, ds.defect_ids) == int(len(r.classes) > 0)
    st = class_statistics(ds)
    assert st["train"]["images"] == 6
    assert sum(st["train"]["boxes"].values()) == sum(len(r.classes) for r in ds.split("train"))


def test_dhash_detects_reencoded_duplicate(tiny_dataset, tmp_path):
    ds = load_dataset(tiny_dataset)
    src = ds.split("train")[0].path
    img = np.asarray(Image.open(src))
    p = tmp_path / "copy.jpg"
    Image.fromarray(np.clip(img * 1.1, 0, 255).astype(np.uint8)).save(p, quality=60)
    assert bin(dhash(src) ^ dhash(p)).count("1") <= 4
    other = ds.split("train")[1].path
    assert bin(dhash(src) ^ dhash(other)).count("1") > 4


def test_cross_split_duplicates_finds_planted_copy(tiny_dataset, tmp_path):
    root = tmp_path / "leak"
    shutil.copytree(tiny_dataset, root)
    ds = load_dataset(root)
    leak = ds.split("train")[0].path
    shutil.copy(leak, root / "test" / "images" / "leaked.jpg")
    dups = cross_split_duplicates(load_dataset(root))
    assert any(d[0].name == "leaked.jpg" and d[2] == 0 for d in dups)


class TestGenerator:
    def test_deterministic(self):
        a = render(np.random.default_rng(3), [0, 4])
        b = render(np.random.default_rng(3), [0, 4])
        np.testing.assert_array_equal(a.image, b.image)
        np.testing.assert_array_equal(a.boxes, b.boxes)

    @pytest.mark.parametrize("cls", range(5))
    def test_each_class_produces_a_box_inside_image(self, cls):
        sc = render(np.random.default_rng(10 + cls), [cls])
        assert cls in sc.classes
        assert sc.image.shape == (H, W, 3) and sc.image.dtype == np.uint8
        assert (sc.boxes[:, [0, 1]] >= 0).all() and (sc.boxes[:, 2] <= W).all() and (sc.boxes[:, 3] <= H).all()

    def test_defect_free_tray_has_no_boxes(self):
        sc = render(np.random.default_rng(1), [])
        assert len(sc.boxes) == 0

    def test_wrinkle_box_contains_the_crease(self):
        """Manufactured-solution check: thin bright structure is concentrated in the labelled box.

        The white top-hat response (bright detail narrower than the disc)
        inside the wrinkle box must exceed that of every same-size window
        sampled elsewhere on the tray in most cases.
        """
        from skimage.morphology import disk, white_tophat

        sc = render(np.random.default_rng(5), [4])
        assert list(sc.classes) == [4]
        g = sc.image.mean(2) / 255
        wth = white_tophat(g, disk(3))
        x1, y1, x2, y2 = sc.boxes[0].astype(int)
        inside = np.percentile(wth[y1:y2, x1:x2], 99)
        rng = np.random.default_rng(0)
        bw, bh = x2 - x1, y2 - y1
        others = []
        for _ in range(40):
            ox, oy = rng.integers(60, W - 60 - bw), rng.integers(60, H - 60 - bh)
            if ox < x2 and ox + bw > x1 and oy < y2 and oy + bh > y1:
                continue
            others.append(np.percentile(wth[oy:oy + bh, ox:ox + bw], 99))
        assert inside > np.percentile(others, 75)

    def test_invalid_class(self):
        with pytest.raises(ValueError):
            render(np.random.default_rng(0), [7])
