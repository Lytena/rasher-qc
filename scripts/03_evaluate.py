"""Step 3 - validation: detection accuracy, tray decision and method comparison.

Protocol
--------
1. Every method produces a tray defect score on the **valid** and **test** splits
   (detectors: highest defect-box confidence; baseline: random-forest P(FAIL)).
2. Decision thresholds are chosen on **valid** only:
   * ``fail_conf``  = highest threshold with valid FAIL-recall >= 0.95
     (target: at most 1 in 20 defective trays not failed automatically);
   * ``review_low`` = highest threshold with valid FAIL-recall >= 0.99
     (trays between the two go to a person).
3. The fixed thresholds are applied once to **test**; 95 % bootstrap
   confidence intervals (1000 resamples of test images) are reported.
4. Detection quality (AP per class, mAP@0.5, mAP@0.5:0.95) is computed with our
   own metric code and cross-checked against Ultralytics' validator.

Outputs: results/<tag>/tables/evaluation.json and figures/*.png

Usage: python scripts/03_evaluate.py --data data/<dataset> [--imgsz 416]
"""

from __future__ import annotations

import pickle
import sys
import time

import numpy as np
from _common import METHOD_COLOURS, METHOD_LABELS, out_dirs, parse, plt, save_json
from PIL import Image, ImageDraw

from rasherqc import baseline, config
from rasherqc.dataset import load_dataset, tray_fail_label
from rasherqc.metrics import (
    Detections,
    binary_confusion,
    bootstrap_ci,
    box_iou,
    evaluate_detections,
    pr_curve,
    review_band_metrics,
    roc_auc,
    threshold_sweep,
    tray_scores,
    _match_class,
)

TARGET_RECALL, REVIEW_RECALL = 0.95, 0.99
#: A class needs at least this many ground-truth boxes in a split for its AP
#: to be reported as evidence (below it, one box changes AP by >= 10 points).
MIN_GT = 10


def extra(ap):
    """Add this script's evaluation options (image size, device, bootstrap size, ...)."""
    ap.add_argument("--imgsz", type=int, default=config.DEFAULT_IMGSZ)
    ap.add_argument("--device", default="cpu")


a = parse(__doc__.split("\n")[0], extra)
fig_dir, tab_dir, model_dir = out_dirs(a.tag)
cache = config.RESULTS_DIR / a.tag / "cache"
cache.mkdir(parents=True, exist_ok=True)
ds = load_dataset(a.data)
nc, D = len(ds.names), ds.defect_ids
splits = {s: ds.split(s) for s in ("valid", "test")}
y = {s: np.array([tray_fail_label(r, D) for r in recs]) for s, recs in splits.items()}
out: dict = {"classes": ds.names, "n_images": {s: len(r) for s, r in splits.items()},
             "fail_trays": {s: int(v.sum()) for s, v in y.items()}, "methods": {}}


def choose_thresholds(yv, sv):
    """Pick the tray thresholds on the *validation* split.

    Args:
        yv: Validation tray labels (1 = defective).
        sv: Validation tray scores (highest defect confidence).

    Returns:
        ``(t_fail, t_low)``: the highest thresholds that still reach the target
        FAIL recall (95 %) and review recall (99 %) respectively.
    """
    grid = np.unique(np.concatenate([[0.0], np.round(sv, 4)]))
    rec = np.array([binary_confusion(yv, sv >= t)["fail_recall"] for t in grid])
    fail = float(grid[rec >= TARGET_RECALL].max())
    low = float(grid[rec >= REVIEW_RECALL].max())
    return fail, min(low, fail)


def tray_block(name, scores):
    """Choose thresholds on valid, report test metrics with bootstrap CIs."""
    sv, st = scores["valid"], scores["test"]
    fail_t, low_t = choose_thresholds(y["valid"], sv)
    yt = y["test"]
    conf = binary_confusion(yt, st >= fail_t)
    band = review_band_metrics(yt, st, low_t, fail_t)
    ci = {
        "fail_recall": bootstrap_ci(lambda i: binary_confusion(yt[i], st[i] >= fail_t)["fail_recall"], len(yt)),
        "false_reject_rate": bootstrap_ci(lambda i: binary_confusion(yt[i], st[i] >= fail_t)["false_reject_rate"], len(yt)),
        "auc": bootstrap_ci(lambda i: roc_auc(yt[i], st[i]), len(yt)),
    }
    return {"fail_conf": fail_t, "review_low": low_t, "valid_auc": roc_auc(y["valid"], sv),
            "test_auc": roc_auc(yt, st), "test_at_fail_conf": conf, "test_review_band": band, "test_ci95": ci,
            "valid_sweep": threshold_sweep(y["valid"], sv)}


# ---------------------------------------------------------------- detectors
preds_all: dict[str, dict[str, list[Detections]]] = {}
for name in ("yolo_pretrained", "yolo_scratch"):
    w = model_dir / f"{name}.pt"
    if not w.is_file():
        print(f"skip {name}: {w} not found (run 02_train_detectors.py)")
        continue
    from rasherqc.detector import Detector

    det = Detector(w, imgsz=a.imgsz, device=a.device)
    preds_all[name] = {}
    m: dict = {"type": "detector"}
    for s, recs in splits.items():
        pfile = cache / f"preds_{name}_{s}_{a.imgsz}.pkl"
        if pfile.is_file():
            preds_all[name][s] = pickle.loads(pfile.read_bytes())
        else:
            t0 = time.perf_counter()
            preds_all[name][s] = det.predict([r.path for r in recs])
            m[f"{s}_predict_s"] = time.perf_counter() - t0
            pfile.write_bytes(pickle.dumps(preds_all[name][s]))
        ev = evaluate_detections([r.boxes for r in recs], [r.classes for r in recs], preds_all[name][s], nc, method="coco101")
        ev_u = evaluate_detections([r.boxes for r in recs], [r.classes for r in recs], preds_all[name][s], nc,
                                   iou_thresholds=[0.5], method="trapz101")
        ok = np.array(ev["n_gt"]) >= MIN_GT  # classes with enough boxes to estimate AP
        m[s] = {"map50": ev["map50"], "map50_95": ev["map50_95"], "ap50": ev["ap50"], "ap50_95": ev["ap50_95"],
                "evaluable_classes": [ds.names[k] for k in np.where(ok)[0]],
                "map50_evaluable": float(np.nanmean(ev["ap50"][ok])) if ok.any() else None,
                "map50_95_evaluable": float(np.nanmean(ev["ap50_95"][ok])) if ok.any() else None,
                "precision@0.25": ev["precision"], "recall@0.25": ev["recall"], "n_gt": ev["n_gt"],
                "map50_trapz101": ev_u["map50"]}
    # cross-check with the Ultralytics validator on the test split
    try:
        r = det.model.val(data=str(ds.yolo_yaml), split="test", imgsz=a.imgsz, conf=0.001, iou=0.6,
                          device=a.device, verbose=False, plots=False, batch=16,
                          workers=0 if sys.platform == "win32" else 2)
        m["ultralytics_test"] = {"map50": float(r.box.map50), "map50_95": float(r.box.map),
                                 "ap50": [float(v) for v in r.box.ap50]}
    except Exception as exc:  # noqa: BLE001 - cross-check is informative only
        m["ultralytics_test"] = {"error": str(exc)}
    scores = {s: tray_scores(preds_all[name][s], D) for s in splits}
    m["tray"] = tray_block(name, scores)
    # per-class presence AUC (max confidence of that class on the tray)
    pres_auc = {}
    for k in range(nc):
        yk = np.array([int(k in r.classes) for r in splits["test"]])
        sk = np.array([p.scores[p.classes == k].max() if (p.classes == k).any() else 0 for p in preds_all[name]["test"]])
        pres_auc[ds.names[k]] = roc_auc(yk, sk)
    m["presence_auc_test"] = pres_auc
    m["_scores"] = {s: v.tolist() for s, v in scores.items()}
    out["methods"][name] = m
    print(f"{name}: evaluable classes {m['test']['evaluable_classes']} mAP50={m['test']['map50_evaluable']}")
    print(f"{name}: test mAP50={m['test']['map50']:.3f} (ultralytics {m['ultralytics_test'].get('map50', float('nan')):.3f}) "
          f"mAP50-95={m['test']['map50_95']:.3f} tray AUC={m['tray']['test_auc']:.3f}")

# ---------------------------------------------------------------- baseline
ffile = cache / "features.npz"
if ffile.is_file():
    z = np.load(ffile)
    F = {s: z[s] for s in ("train", "valid", "test")}
    feat_time = float(z["time"])
else:
    t0 = time.perf_counter()
    print("baseline: extracting hand-crafted features (CPU, a few minutes) ...", flush=True)
    F = {s: baseline.features_for(ds.split(s), n_jobs=2) for s in ("train", "valid", "test")}
    print("baseline: fitting random forest with 5-fold cross-validation ...", flush=True)
    feat_time = time.perf_counter() - t0
    np.savez(ffile, time=feat_time, **F)
yf_tr, yp_tr = baseline.labels(ds.split("train"), nc, D)
t0 = time.perf_counter()
bl = baseline.fit(F["train"], yf_tr, yp_tr)
fit_time = time.perf_counter() - t0
(model_dir / "rf_baseline.pkl").write_bytes(pickle.dumps(bl))
scores = {s: bl.predict_fail_proba(F[s]) for s in splits}
mb: dict = {"type": "classical", "feature_time_s": feat_time, "fit_time_s": fit_time, "cv": bl.cv_results}
mb["tray"] = tray_block("rf_baseline", scores)
pp = bl.predict_presence_proba(F["test"])
_, yp_te = baseline.labels(splits["test"], nc, D)
mb["presence_auc_test"] = {ds.names[k]: roc_auc(yp_te[:, k], pp[:, k]) for k in range(nc)}
from rasherqc.features import feature_names  # noqa: E402

imp = bl.fail_model.feature_importances_
order = np.argsort(-imp)[:15]
mb["top_features"] = [[feature_names()[i], float(imp[i])] for i in order]
mb["_scores"] = {s: v.tolist() for s, v in scores.items()}
out["methods"]["rf_baseline"] = mb
print(f"rf_baseline: CV AUC={bl.cv_results['best_score']:.3f} tray test AUC={mb['tray']['test_auc']:.3f}")

# ---------------------------------------------------------------- recall vs defect size
size_bins = np.array([0, 0.002, 0.005, 0.01, 0.02, 0.05, 1.0])
size_tab = {}
for name, P in preds_all.items():
    fail_t = out["methods"][name]["tray"]["fail_conf"]
    hit, tot = np.zeros(len(size_bins) - 1), np.zeros(len(size_bins) - 1)
    for r, p in zip(splits["test"], P["test"]):
        keep = p.scores >= fail_t
        for b, c in zip(r.boxes, r.classes):
            af = (b[2] - b[0]) * (b[3] - b[1]) / (r.width * r.height)
            k = int(np.clip(np.searchsorted(size_bins, af, side="right") - 1, 0, len(size_bins) - 2))
            tot[k] += 1
            same = keep & (p.classes == c)
            if same.any() and box_iou(b[None], p.boxes[same]).max() >= config.IOU_MATCH:
                hit[k] += 1
    size_tab[name] = {"bins": size_bins.tolist(), "recall": (hit / np.maximum(tot, 1)).tolist(), "n": tot.tolist()}
out["recall_vs_size"] = size_tab
save_json(out, tab_dir / "evaluation.json")

# ---------------------------------------------------------------- figures
methods = list(out["methods"])
# 1 tray ROC (test)
fig, ax = plt.subplots(figsize=(4, 3.6))
for n in methods:
    s = np.array(out["methods"][n]["_scores"]["test"])
    ts = np.unique(np.concatenate([[-1, 2], s]))[::-1]
    tpr = [np.mean(s[y["test"] == 1] >= t) for t in ts]
    fpr = [np.mean(s[y["test"] == 0] >= t) for t in ts]
    ax.plot(fpr, tpr, color=METHOD_COLOURS[n], label=f"{METHOD_LABELS[n]} (AUC {out['methods'][n]['tray']['test_auc']:.3f})")
ax.plot([0, 1], [0, 1], ":", color="grey")
ax.set_xlabel("False-reject rate (good trays failed)")
ax.set_ylabel("FAIL recall (defective trays caught)")
ax.set_title("Tray decision ROC - test split")
ax.legend(fontsize=7, loc="lower right")
fig.savefig(fig_dir / "tray_roc.png")

# 2 threshold sweep (valid) for main method
main = "yolo_pretrained" if "yolo_pretrained" in methods else methods[0]
sw = out["methods"][main]["tray"]["valid_sweep"]
fig, ax = plt.subplots(figsize=(5, 3))
t = [r["threshold"] for r in sw]
ax.plot(t, [r["fail_recall"] for r in sw], label="FAIL recall")
ax.plot(t, [r["false_reject_rate"] for r in sw], label="false-reject rate")
ax.plot(t, [r["precision"] for r in sw], label="precision")
ax.axvline(out["methods"][main]["tray"]["fail_conf"], color="k", ls="--", lw=1, label="chosen fail_conf")
ax.axhline(TARGET_RECALL, color="grey", ls=":", lw=1)
ax.set_xlabel("FAIL threshold on tray score")
ax.set_title(f"Threshold selection on validation - {METHOD_LABELS[main]}")
ax.legend(fontsize=7)
fig.savefig(fig_dir / "threshold_sweep.png")

# 3 per-class AP50 bars
det_methods = [n for n in methods if out["methods"][n]["type"] == "detector"]
if det_methods:
    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(nc)
    for k, n in enumerate(det_methods):
        ax.bar(x + (k - 0.5) * 0.38, np.nan_to_num(np.array(out["methods"][n]["test"]["ap50"], float)), 0.38,
               color=METHOD_COLOURS[n], label=METHOD_LABELS[n])
    ax.set_xticks(x, ds.names, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AP@0.5 (test)")
    ax.set_title("Detection accuracy per defect class")
    ax.legend(fontsize=7)
    fig.savefig(fig_dir / "ap_per_class.png")

    # 4 PR curves, main method
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    recs = splits["test"]
    gts = list(zip([r.classes for r in recs], [r.boxes for r in recs]))
    for k in range(nc):
        s, tp, ng = _match_class(gts, preds_all[main]["test"], k, 0.5)
        if ng == 0 or len(s) == 0:
            continue
        rc, pc = pr_curve(s, tp, ng)
        ax.plot(rc, pc, label=f"{ds.names[k]} (AP {out['methods'][main]['test']['ap50'][k]:.2f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall @IoU 0.5 (test)")
    ax.legend(fontsize=7)
    fig.savefig(fig_dir / "pr_curves.png")

    # 5 recall vs size
    fig, ax = plt.subplots(figsize=(5, 3))
    labels = [f"{size_bins[i]*100:.1f}-{size_bins[i+1]*100:.1f}" for i in range(len(size_bins) - 1)]
    for n in det_methods:
        ax.plot(labels, size_tab[n]["recall"], "o-", color=METHOD_COLOURS[n], label=METHOD_LABELS[n])
    ax.set_xlabel("defect box area (% of image)")
    ax.set_ylabel("recall at fail_conf")
    ax.set_ylim(0, 1.05)
    ax.set_title("Detection rate vs defect size (test)")
    ax.legend(fontsize=7)
    fig.savefig(fig_dir / "recall_vs_size.png")

    # 6 error gallery: missed defective trays and wrongly failed good trays
    ft = out["methods"][main]["tray"]["fail_conf"]
    st = np.array(out["methods"][main]["_scores"]["test"])
    fn_idx = [i for i in np.argsort(st) if y["test"][i] == 1 and st[i] < ft][:3]
    fp_idx = [i for i in np.argsort(-st) if y["test"][i] == 0 and st[i] >= ft][:3]
    tp_idx = [i for i in np.argsort(-st) if y["test"][i] == 1 and st[i] >= ft][:3]
    panels = [("caught (TP)", i) for i in tp_idx] + [("missed (FN)", i) for i in fn_idx] + [("false reject (FP)", i) for i in fp_idx]
    if panels:
        fig, axs = plt.subplots(int(np.ceil(len(panels) / 3)), 3, figsize=(10, 2.6 * np.ceil(len(panels) / 3)))
        for ax, (lab, i) in zip(np.atleast_1d(axs).ravel(), panels):
            r, p = splits["test"][i], preds_all[main]["test"][i]
            im = Image.open(r.path).convert("RGB")
            d = ImageDraw.Draw(im)
            for b in r.boxes:
                d.rectangle(list(b), outline=(0, 220, 0), width=3)
            for b, sc, c in zip(p.boxes, p.scores, p.classes):
                if sc >= min(ft, 0.1):
                    d.rectangle(list(b), outline=(255, 40, 40), width=2)
                    d.text((b[0] + 3, b[3] - 12), f"{ds.names[c]} {sc:.2f}", fill=(255, 60, 60))
            ax.imshow(im)
            ax.set_title(f"{lab}  score {st[i]:.2f}", fontsize=8)
            ax.axis("off")
        for ax in np.atleast_1d(axs).ravel()[len(panels):]:
            ax.axis("off")
        fig.suptitle("green = ground truth, red = prediction", fontsize=8)
        fig.savefig(fig_dir / "error_gallery.png")

# 7 feature importances
fig, ax = plt.subplots(figsize=(5, 3.4))
tf = out["methods"]["rf_baseline"]["top_features"][:12][::-1]
ax.barh([n for n, _ in tf], [v for _, v in tf], color=METHOD_COLOURS["rf_baseline"])
ax.set_xlabel("impurity importance")
ax.set_title("Baseline: most important features")
fig.savefig(fig_dir / "feature_importance.png")
print("done ->", tab_dir / "evaluation.json")
