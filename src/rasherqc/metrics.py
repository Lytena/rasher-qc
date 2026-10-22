"""Detection and tray-decision metrics, implemented from first principles.

These functions are written independently of the detector library so that
the reported numbers can be *verified*: unit tests check them against hand-
computed cases, and ``scripts/03_evaluate.py`` cross-checks our mAP@0.5
against the value Ultralytics reports for the same predictions.

Definitions
-----------
IoU (intersection over union) of boxes A, B::

    IoU = area(A ∩ B) / (area(A) + area(B) - area(A ∩ B))

Matching (PASCAL VOC / COCO rule): predictions of one class are processed in
descending confidence; each is a true positive (TP) if its best-overlapping
*still unmatched* ground-truth (GT) box of that class has IoU >= threshold,
otherwise a false positive (FP). Unmatched GT boxes are false negatives.

Average precision (AP) is the area under the precision-recall curve after
making precision monotonically non-increasing (the "precision envelope"):

* ``method="continuous"``: exact area (all-point interpolation, VOC2010+);
* ``method="coco101"``: mean envelope precision at recall 0, 0.01, ..., 1
  (pycocotools);
* ``method="trapz101"``: trapezoidal integral of the interpolated envelope
  at the same 101 points (the Ultralytics implementation).

mAP@0.5 averages AP over classes that have GT boxes; mAP@0.5:0.95 also
averages over IoU thresholds 0.50, 0.55, ..., 0.95.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class Detections:
    """Predictions for one image.

    Attributes:
        boxes: (N, 4) pixel ``x1, y1, x2, y2``.
        scores: (N,) confidences in [0, 1].
        classes: (N,) class ids.
    """

    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray

    @staticmethod
    def empty() -> "Detections":
        """An image with no predictions."""
        return Detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of ``xyxy`` boxes.

    Args:
        a: (N, 4) boxes.
        b: (M, 4) boxes.

    Returns:
        (N, M) IoU matrix; 0 where boxes do not overlap. Degenerate boxes
        (zero area) give IoU 0 rather than NaN.
    """
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def _match_class(gts: list[np.ndarray], preds: list[Detections], cls: int, iou_thr: float):
    """TP flags and scores of all predictions of one class, plus the GT count."""
    scores, tps = [], []
    n_gt = 0
    for g_boxes, p in zip(gts, preds):
        g = g_boxes[1][g_boxes[0] == cls] if isinstance(g_boxes, tuple) else g_boxes
        n_gt += len(g)
        sel = p.classes == cls
        pb, ps = p.boxes[sel], p.scores[sel]
        order = np.argsort(-ps, kind="stable")
        pb, ps = pb[order], ps[order]
        used = np.zeros(len(g), bool)
        ious = box_iou(pb, g) if len(g) and len(pb) else np.zeros((len(pb), len(g)))
        for i in range(len(pb)):
            tp = False
            if len(g):
                cand = np.where(used, -1.0, ious[i])
                j = int(np.argmax(cand))
                if cand[j] >= iou_thr:
                    used[j] = True
                    tp = True
            scores.append(ps[i])
            tps.append(tp)
    return np.array(scores), np.array(tps, bool), n_gt


def pr_curve(scores: np.ndarray, tps: np.ndarray, n_gt: int) -> tuple[np.ndarray, np.ndarray]:
    """Precision and recall after each prediction (sorted by descending score).

    Args:
        scores: Prediction confidences.
        tps: True-positive flags aligned with ``scores``.
        n_gt: Number of ground-truth boxes.

    Returns:
        ``(recall, precision)`` arrays of length ``len(scores)``.
    """
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(tps[order])
    fp = np.cumsum(~tps[order])
    recall = tp / max(n_gt, 1)
    precision = tp / np.maximum(tp + fp, 1)
    return recall, precision


def average_precision(recall: np.ndarray, precision: np.ndarray, method: str = "coco101") -> float:
    """Area under the precision envelope.

    Args:
        recall: Recall values (non-decreasing).
        precision: Matching precision values.
        method: ``"continuous"``, ``"coco101"`` or ``"trapz101"`` (see module docstring).

    Returns:
        AP in [0, 1].
    """
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))  # envelope
    if method == "continuous":
        i = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    if method == "coco101":
        # pycocotools: envelope of the *observed* points only (no sentinels);
        # precision at recall level x = envelope at the first point with
        # recall >= x, or 0 if that recall is never reached.
        x = np.linspace(0, 1, 101)
        env = np.flip(np.maximum.accumulate(np.flip(np.asarray(precision, float))))
        idx = np.searchsorted(np.asarray(recall, float), x, side="left")
        q = np.where(idx < len(env), env[np.clip(idx, 0, max(len(env) - 1, 0))] if len(env) else 0.0, 0.0)
        return float(np.mean(q))
    if method == "trapz101":
        # Ultralytics' variant: trapezoidal integral of the linearly
        # interpolated envelope at 101 recall points.
        x = np.linspace(0, 1, 101)
        y = np.interp(x, mrec, mpre)
        return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))
    raise ValueError(f"unknown AP method {method!r}")


def evaluate_detections(
    gt_boxes: list[np.ndarray],
    gt_classes: list[np.ndarray],
    preds: list[Detections],
    n_classes: int,
    iou_thresholds: np.ndarray | None = None,
    method: str = "coco101",
    conf: float = 0.25,
) -> dict:
    """Per-class AP, mAP and precision/recall for a set of images.

    Args:
        gt_boxes: Per image, (N, 4) GT boxes.
        gt_classes: Per image, (N,) GT class ids.
        preds: Per image, :class:`Detections`.
        n_classes: Number of classes.
        iou_thresholds: IoU thresholds (default 0.50:0.05:0.95).
        method: AP interpolation method.
        conf: Confidence threshold at which precision/recall/F1 are reported.

    Returns:
        Dict with ``ap50`` and ``ap50_95`` (per-class arrays, NaN for classes
        without GT), ``map50``, ``map50_95``, ``precision``, ``recall``,
        ``f1`` (per class at ``conf``) and ``n_gt`` per class.
    """
    if not len(gt_boxes) == len(gt_classes) == len(preds):
        raise ValueError("gt_boxes, gt_classes and preds must have the same length")
    thr = np.arange(0.5, 0.96, 0.05) if iou_thresholds is None else np.asarray(iou_thresholds)
    gts = list(zip(gt_classes, gt_boxes))
    ap = np.full((n_classes, len(thr)), np.nan)
    P, R, F = (np.full(n_classes, np.nan) for _ in range(3))
    n_gt_all = np.zeros(n_classes, int)
    for c in range(n_classes):
        for k, t in enumerate(thr):
            s, tp, n_gt = _match_class(gts, preds, c, float(t))
            n_gt_all[c] = n_gt
            if n_gt == 0:
                continue
            if len(s) == 0:
                ap[c, k] = 0.0
                continue
            rec, pre = pr_curve(s, tp, n_gt)
            ap[c, k] = average_precision(rec, pre, method)
            if k == 0:
                keep = s >= conf
                tpc = int(tp[keep].sum())
                P[c] = tpc / max(int(keep.sum()), 1)
                R[c] = tpc / n_gt
                F[c] = 2 * P[c] * R[c] / (P[c] + R[c]) if P[c] + R[c] > 0 else 0.0
        if n_gt_all[c] == 0:
            P[c] = R[c] = F[c] = np.nan
    i50 = int(np.argmin(np.abs(thr - 0.5)))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows = classes without GT
        ap_cls = np.nanmean(ap, axis=1)
        map50, map5095 = float(np.nanmean(ap[:, i50])), float(np.nanmean(ap))
    return {
        "ap50": ap[:, i50],
        "ap50_95": ap_cls,
        "map50": map50,
        "map50_95": map5095,
        "precision": P,
        "recall": R,
        "f1": F,
        "n_gt": n_gt_all,
    }


# ----------------------------------------------------------------------------
# Tray-level PASS / FAIL decision metrics
# ----------------------------------------------------------------------------
def tray_scores(preds: list[Detections], defect_ids: list[int]) -> np.ndarray:
    """Tray defect score: highest confidence among defect-class detections.

    Args:
        preds: Per-image detections.
        defect_ids: Class ids that are defects.

    Returns:
        (N,) scores in [0, 1]; 0 when nothing was detected.
    """
    out = np.zeros(len(preds))
    for i, p in enumerate(preds):
        sel = np.isin(p.classes, defect_ids)
        out[i] = float(p.scores[sel].max()) if sel.any() else 0.0
    return out


def binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Confusion counts and rates with FAIL (1) as the positive class.

    Returns:
        Dict with ``tp, fp, tn, fn``, ``fail_recall`` (sensitivity: share of
        defective trays caught), ``false_reject_rate`` (good trays failed),
        ``precision``, ``accuracy`` and ``f1``.
    """
    y_true = np.asarray(y_true, bool)
    y_pred = np.asarray(y_pred, bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    rec = tp / max(tp + fn, 1)
    pre = tp / max(tp + fp, 1)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "fail_recall": rec,
        "false_reject_rate": fp / max(fp + tn, 1),
        "precision": pre,
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "f1": 2 * pre * rec / (pre + rec) if pre + rec > 0 else 0.0,
    }


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve (probability a FAIL tray outscores a PASS tray).

    Computed with the Mann-Whitney U statistic (ties count one half), which
    equals the trapezoidal area under the empirical ROC curve.
    """
    y = np.asarray(y_true, bool)
    s = np.asarray(scores, float)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def threshold_sweep(y_true: np.ndarray, scores: np.ndarray, thresholds: np.ndarray | None = None) -> list[dict]:
    """Tray metrics for a range of FAIL thresholds (FAIL if score >= t)."""
    ts = np.round(np.arange(0.05, 0.96, 0.05), 2) if thresholds is None else thresholds
    return [{"threshold": float(t), **binary_confusion(y_true, scores >= t)} for t in ts]


def review_band_metrics(y_true: np.ndarray, scores: np.ndarray, low: float, high: float) -> dict:
    """Three-way decision: PASS (< low), REVIEW ([low, high)), FAIL (>= high).

    Returns:
        Dict with ``escape_rate`` (defective trays auto-PASSED, the critical
        error), ``false_reject_rate`` (good trays auto-FAILED), ``review_rate``
        and the counts.
    """
    y = np.asarray(y_true, bool)
    s = np.asarray(scores, float)
    passed, failed = s < low, s >= high
    review = ~passed & ~failed
    return {
        "low": low, "high": high,
        "escape_rate": float(np.sum(y & passed) / max(y.sum(), 1)),
        "false_reject_rate": float(np.sum(~y & failed) / max((~y).sum(), 1)),
        "review_rate": float(review.mean()),
        "n_escaped": int(np.sum(y & passed)),
        "n_review": int(review.sum()),
    }


def bootstrap_ci(values_fn, n: int, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap confidence interval over images.

    Args:
        values_fn: Function ``f(idx) -> float`` computing the metric on the
            resampled image indices ``idx``.
        n: Number of images.
        n_boot: Bootstrap replicates.
        seed: Random seed.
        alpha: 1 - confidence level.

    Returns:
        ``(lower, upper)`` bounds.
    """
    rng = np.random.default_rng(seed)
    vals = [values_fn(rng.integers(0, n, n)) for _ in range(n_boot)]
    vals = np.array([v for v in vals if np.isfinite(v)])
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))
