"""Verification of the metric implementations against hand-computed values.

These are the "analytical solutions" of the project: small cases whose
correct answer can be worked out on paper.
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from rasherqc.metrics import (
    Detections,
    average_precision,
    binary_confusion,
    bootstrap_ci,
    box_iou,
    evaluate_detections,
    pr_curve,
    review_band_metrics,
    roc_auc,
    tray_scores,
)


def det(boxes, scores, classes):
    return Detections(np.array(boxes, float).reshape(-1, 4), np.array(scores, float), np.array(classes, int))


class TestIoU:
    def test_identical_boxes(self):
        assert box_iou([[0, 0, 10, 10]], [[0, 0, 10, 10]])[0, 0] == pytest.approx(1.0)

    def test_half_overlap(self):
        # 10x10 boxes shifted by 5 px: inter 50, union 150 -> 1/3
        assert box_iou([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0] == pytest.approx(1 / 3)

    def test_disjoint_and_touching(self):
        iou = box_iou([[0, 0, 10, 10]], [[20, 20, 30, 30], [10, 0, 20, 10]])
        assert np.all(iou == 0)

    def test_contained(self):
        assert box_iou([[0, 0, 10, 10]], [[2, 2, 7, 7]])[0, 0] == pytest.approx(25 / 100)

    def test_degenerate_box_gives_zero_not_nan(self):
        iou = box_iou([[5, 5, 5, 5]], [[5, 5, 5, 5]])
        assert np.isfinite(iou).all() and iou[0, 0] == 0

    def test_shape_and_symmetry(self, rng):
        a = rng.uniform(0, 50, (6, 2))
        a = np.hstack([a, a + rng.uniform(1, 20, (6, 2))])
        b = rng.uniform(0, 50, (4, 2))
        b = np.hstack([b, b + rng.uniform(1, 20, (4, 2))])
        m = box_iou(a, b)
        assert m.shape == (6, 4)
        np.testing.assert_allclose(m, box_iou(b, a).T)
        assert (m >= 0).all() and (m <= 1).all()


class TestAP:
    def test_perfect_detector(self):
        rec, pre = pr_curve(np.array([0.9, 0.8]), np.array([True, True]), 2)
        for m in ("continuous", "coco101"):
            assert average_precision(rec, pre, m) == pytest.approx(1.0, abs=1e-9)
        # Ultralytics' trapezoidal variant tops out at 0.995: the interpolated
        # envelope drops to the sentinel 0 at recall 1, halving the last slice.
        assert average_precision(rec, pre, "trapz101") == pytest.approx(0.995)

    def test_hand_computed_continuous(self):
        # ranking TP, FP, TP with 2 GT: recall .5,.5,1 precision 1,.5,.667
        # envelope: 1 up to r=.5, .667 up to r=1 -> AP = .5*1 + .5*2/3 = 0.8333
        rec, pre = pr_curve(np.array([0.9, 0.8, 0.7]), np.array([True, False, True]), 2)
        assert average_precision(rec, pre, "continuous") == pytest.approx(0.5 + 0.5 * 2 / 3)

    def test_hand_computed_coco101(self):
        rec, pre = pr_curve(np.array([0.9, 0.8, 0.7]), np.array([True, False, True]), 2)
        # recall points 0..0.50 (51 pts) -> 1.0 ; 0.51..1.0 (50 pts) -> 2/3
        assert average_precision(rec, pre, "coco101") == pytest.approx((51 * 1 + 50 * 2 / 3) / 101)

    def test_missed_gt_caps_ap(self):
        # one TP out of 2 GT, no FP -> AP = 0.5 (recall never exceeds 0.5)
        rec, pre = pr_curve(np.array([0.9]), np.array([True]), 2)
        assert average_precision(rec, pre, "continuous") == pytest.approx(0.5)

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            average_precision(np.array([1.0]), np.array([1.0]), "nope")


class TestEvaluateDetections:
    def test_perfect_predictions(self):
        gt_b = [np.array([[0, 0, 10, 10]]), np.array([[20, 20, 40, 40], [50, 50, 60, 60]])]
        gt_c = [np.array([0]), np.array([1, 0])]
        preds = [det(gt_b[0], [0.9], [0]), det(gt_b[1], [0.8, 0.7], [1, 0])]
        r = evaluate_detections(gt_b, gt_c, preds, n_classes=3)
        assert r["map50"] == pytest.approx(1.0)
        assert r["map50_95"] == pytest.approx(1.0)
        assert np.isnan(r["ap50"][2])  # class with no GT is excluded
        assert list(r["n_gt"]) == [2, 1, 0]

    def test_wrong_class_is_false_positive(self):
        r = evaluate_detections([np.array([[0, 0, 10, 10]])], [np.array([0])], [det([[0, 0, 10, 10]], [0.9], [1])], 2)
        assert r["ap50"][0] == 0.0

    def test_duplicate_detection_counts_once(self):
        gt = [np.array([[0, 0, 10, 10]])]
        preds = [det([[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8], [0, 0])]
        r = evaluate_detections(gt, [np.array([0])], preds, 1, conf=0.5)
        assert r["precision"][0] == pytest.approx(0.5)
        assert r["recall"][0] == pytest.approx(1.0)
        assert r["ap50"][0] == pytest.approx(1.0)  # FP ranked after the TP

    def test_iou_threshold_dependence(self):
        # IoU = 1/3: TP at threshold 0.3, FP at 0.5
        gt = [np.array([[0, 0, 10, 10]])]
        p = [det([[5, 0, 15, 10]], [0.9], [0])]
        assert evaluate_detections(gt, [np.array([0])], p, 1, iou_thresholds=[0.3])["map50"] == pytest.approx(1)
        assert evaluate_detections(gt, [np.array([0])], p, 1, iou_thresholds=[0.5])["map50"] == 0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            evaluate_detections([np.zeros((0, 4))], [], [], 1)


class TestTrayMetrics:
    def test_tray_scores_use_defect_classes_only(self):
        preds = [det([[0, 0, 1, 1], [0, 0, 2, 2]], [0.9, 0.4], [0, 1]), Detections.empty()]
        np.testing.assert_allclose(tray_scores(preds, defect_ids=[1]), [0.4, 0.0])

    def test_binary_confusion_hand_case(self):
        c = binary_confusion([1, 1, 1, 0, 0], [1, 0, 1, 1, 0])
        assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (2, 1, 1, 1)
        assert c["fail_recall"] == pytest.approx(2 / 3)
        assert c["false_reject_rate"] == pytest.approx(0.5)

    def test_auc_matches_sklearn(self, rng):
        y = rng.integers(0, 2, 200)
        s = np.round(rng.random(200) + 0.3 * y, 2)  # ties included on purpose
        assert roc_auc(y, s) == pytest.approx(roc_auc_score(y, s))

    def test_review_band(self):
        m = review_band_metrics(np.array([1, 1, 0, 0]), np.array([0.05, 0.5, 0.15, 0.02]), 0.1, 0.4)
        assert m["escape_rate"] == pytest.approx(0.5)
        assert m["review_rate"] == pytest.approx(0.25)
        assert m["false_reject_rate"] == 0

    def test_bootstrap_contains_point_estimate(self, rng):
        x = rng.normal(5, 1, 300)
        lo, hi = bootstrap_ci(lambda idx: x[idx].mean(), len(x), n_boot=300)
        assert lo < x.mean() < hi
