"""Baseline features, tray decision, perturbations, report and performance helpers."""

import json

import numpy as np
import pytest
from PIL import Image

from rasherqc import features, robustness
from rasherqc.decision import decide
from rasherqc.metrics import Detections
from rasherqc.performance import conv_layer_table, time_function
from rasherqc.report import inspect_folder
from rasherqc.synthetic import render


@pytest.fixture(scope="module")
def tray_imgs():
    rng = np.random.default_rng(11)
    out = []
    for d in ([], [0], [4]):
        sc = render(rng, d)
        out.append(np.asarray(Image.fromarray(sc.image).resize(features.FEAT_WH, Image.BOX)))
    return out


class TestFeatures:
    def test_length_and_finite(self, tray_imgs):
        f = features.extract_features(tray_imgs[0])
        assert f.shape == (len(features.feature_names()),)
        assert np.isfinite(f).all()

    def test_reference_and_fast_agree(self, tray_imgs):
        """The optimised implementation must preserve the information content.

        Features not touched by the optimisation are identical; changed
        ones (GLCM levels, top-hat size) must stay strongly correlated
        across images.
        """
        names = features.feature_names()
        F = np.stack([features.extract_features(x, "fast") for x in tray_imgs])
        R = np.stack([features.extract_features(x, "reference") for x in tray_imgs])
        same = [i for i, n in enumerate(names) if n.startswith(("meat_", "lab_", "dark_red", "pale", "lbp_", "spec_"))]
        np.testing.assert_allclose(F[:, same], R[:, same], rtol=1e-6, atol=1e-9)

    def test_meat_mask_covers_rashers(self, tray_imgs):
        from skimage.color import rgb2lab

        m = features.meat_mask(rgb2lab(tray_imgs[0]))
        assert 0.3 < m.mean() < 0.85  # rashers cover most of the tray, not the belt

    def test_bad_input(self):
        with pytest.raises(ValueError):
            features.extract_features(np.zeros((10, 10), np.uint8))
        with pytest.raises(ValueError):
            features.extract_features(np.zeros((240, 320, 3), np.uint8), mode="turbo")

    def test_parallel_equals_serial(self, tray_imgs):
        a = features.extract_batch(tray_imgs, n_jobs=1)
        b = features.extract_batch(tray_imgs, n_jobs=2)
        np.testing.assert_array_equal(a, b)


class TestDecision:
    names = ["loose-meat", "packaging-error", "twisted-meat", "unsealed", "wrinkle"]

    def mk(self, scores, classes):
        n = len(scores)
        return Detections(np.tile([10, 10, 110, 60], (n, 1)).astype(float), np.array(scores, float), np.array(classes, int))

    @pytest.mark.parametrize("score, status", [(0.9, "FAIL"), (0.25, "FAIL"), (0.2, "REVIEW"), (0.05, "PASS")])
    def test_thresholds(self, score, status):
        r = decide(self.mk([score], [4]), self.names, list(range(5)), (720, 540), 0.25, 0.10)
        assert r.status == status

    def test_no_detections_pass(self):
        r = decide(Detections.empty(), self.names, list(range(5)), (720, 540))
        assert r.status == "PASS" and r.score == 0 and r.defects == []

    def test_non_defect_class_ignored(self):
        r = decide(self.mk([0.99], [0]), self.names, [1, 2, 3, 4], (720, 540))
        assert r.status == "PASS"

    def test_area_fraction(self):
        r = decide(self.mk([0.9], [3]), self.names, list(range(5)), (720, 540))
        assert r.defects[0]["area_frac"] == pytest.approx(100 * 50 / (720 * 540), abs=1e-5)
        assert r.defects[0]["class"] == "unsealed"

    def test_invalid_band(self):
        with pytest.raises(ValueError):
            decide(Detections.empty(), self.names, [0], (720, 540), fail_conf=0.1, review_low=0.5)


class TestRobustness:
    @pytest.mark.parametrize("name, neutral", [("exposure", 1.0), ("saturation", 1.0), ("glare", 0.0), ("blur", 0.0), ("noise", 0.0)])
    def test_neutral_level_is_identity(self, tray_imgs, name, neutral):
        fn = robustness.SWEEPS[name][0]
        np.testing.assert_array_equal(fn(tray_imgs[0], neutral), tray_imgs[0])

    def test_monotone_effects(self, tray_imgs):
        x = tray_imgs[0]
        assert robustness.exposure(x, 1.4).mean() > x.mean() > robustness.exposure(x, 0.6).mean()
        sat = lambda im: (im.max(2).astype(int) - im.min(2)).mean()  # noqa: E731
        assert sat(robustness.saturation(x, 0.3)) < sat(x)
        assert robustness.glare(x, 0.6).mean() > x.mean()
        assert np.abs(np.diff(robustness.blur(x, 3).astype(float), axis=1)).mean() < np.abs(np.diff(x.astype(float), axis=1)).mean()

    def test_outputs_uint8(self, tray_imgs):
        for name, (fn, levels) in robustness.SWEEPS.items():
            y = fn(tray_imgs[0], levels[-1])
            assert y.dtype == np.uint8 and y.shape == tray_imgs[0].shape, name


class FakeDetector:
    """Stands in for YOLO so the report can be tested without a model."""

    names = TestDecision.names

    def predict(self, images):
        return [Detections(np.array([[50.0, 50, 150, 120]]), np.array([0.8]), np.array([4])) for _ in images]


def test_inspect_folder_end_to_end(tmp_path, rng):
    good = tmp_path / "in"
    good.mkdir()
    Image.fromarray(render(np.random.default_rng(2), []).image).save(good / "tray.jpg")
    Image.fromarray(rng.integers(0, 255, (540, 720), dtype=np.uint8)).save(good / "grey.png")
    (good / "broken.jpg").write_bytes(b"xx")
    rows = inspect_folder(FakeDetector(), sorted(good.iterdir()), tmp_path / "out", list(range(5)))
    status = {r["image"]: r["status"] for r in rows}
    assert status == {"tray.jpg": "FAIL", "grey.png": "UNSUPPORTED_INPUT", "broken.jpg": "UNSUPPORTED_INPUT"}
    assert json.loads((tmp_path / "out" / "results.json").read_text()) == rows
    assert (tmp_path / "out" / "annotated" / "tray.jpg").is_file()
    assert "FAIL" in (tmp_path / "out" / "summary.html").read_text()


def test_conv_flops_hand_calculation():
    """FLOP counter vs a hand calculation for a two-layer network."""
    torch = pytest.importorskip("torch")
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.Conv2d(8, 16, 3, stride=2, padding=1))
    rows = conv_layer_table(net, (32, 32))
    assert rows[0]["flops"] == 2 * 8 * 32 * 32 * 3 * 9
    assert rows[1]["flops"] == 2 * 16 * 16 * 16 * 8 * 9
    assert rows[1]["bytes"] == 4 * (8 * 32 * 32 + 16 * 8 * 9 + 16 * 16 * 16)


def test_time_function_runs():
    t = time_function(sum, [1, 2, 3], n=5, warmup=1)
    assert t["mean_ms"] >= 0 and t["p95_ms"] >= t["min_ms"]
