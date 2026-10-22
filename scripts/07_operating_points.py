"""Step 7 - operating points and resolution sweep on the evaluable classes.

The headline threshold rule of step 3 (catch >= 95 % of defective validation
trays) is one point on a trade-off curve. This script tabulates the whole
curve on the test split so that the plant can choose, and repeats the
resolution sweep of step 5 with mAP restricted to the *evaluable* classes
(>= 10 test boxes) -- the all-class mAP is dominated by classes with 1-4 boxes,
where one detection moves AP by 25-100 points.

Outputs: results/<tag>/tables/operating_points.json, figures/operating_points.png

Usage: python scripts/07_operating_points.py --data data/<dataset> [--device cpu]
"""

from __future__ import annotations

import numpy as np
from _common import METHOD_COLOURS, METHOD_LABELS, load_json, out_dirs, parse, plt, save_json
from PIL import Image

from rasherqc import config
from rasherqc.dataset import load_dataset, tray_fail_label
from rasherqc.metrics import binary_confusion, bootstrap_ci, evaluate_detections

MIN_GT = 10


def extra(ap):
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sizes", nargs="+", type=int, default=[256, 320, 416, 512, 640])


a = parse(__doc__.split("\n")[0], extra)
fig_dir, tab_dir, model_dir = out_dirs(a.tag)
ev = load_json(tab_dir / "evaluation.json")
ds = load_dataset(a.data)
recs = ds.split("test")
y = np.array([tray_fail_label(r, ds.defect_ids) for r in recs])
out: dict = {"n_test": len(y), "n_fail": int(y.sum()), "n_good": int((1 - y).sum()), "methods": {}}

fig, ax = plt.subplots(figsize=(5, 3.4))
for n, m in ev["methods"].items():
    s = np.array(m["_scores"]["test"])
    rows = []
    for target in (0.80, 0.85, 0.90, 0.95, 0.99):
        # highest threshold reaching the target recall ON TEST (descriptive, not a selection)
        ts = np.unique(s)[::-1]
        t = next((t for t in ts if binary_confusion(y, s >= t)["fail_recall"] >= target), ts[-1])
        c = binary_confusion(y, s >= t)
        lo, hi = bootstrap_ci(lambda i: binary_confusion(y[i], s[i] >= t)["false_reject_rate"], len(y), n_boot=500)
        rows.append({"target_recall": target, "threshold": float(t), "recall": c["fail_recall"],
                     "false_reject_rate": c["false_reject_rate"], "frr_ci95": [lo, hi]})
    out["methods"][n] = rows
    ax.plot([r["recall"] for r in rows], [r["false_reject_rate"] for r in rows], "o-", color=METHOD_COLOURS[n],
            label=METHOD_LABELS[n])
ax.set_xlabel("FAIL recall (defective trays caught)")
ax.set_ylabel("false-reject rate (good trays failed)")
ax.set_title(f"Operating points, test split ({out['n_good']} good / {out['n_fail']} defective trays)")
ax.legend(fontsize=7)
fig.savefig(fig_dir / "operating_points.png")

# resolution sweep with evaluable-class mAP
w = model_dir / "yolo_pretrained.pt"
if w.is_file():
    from rasherqc.detector import Detector

    det = Detector(w, device=a.device)
    imgs = [np.asarray(Image.open(r.path).convert("RGB")) for r in recs]
    sweep = {}
    for sz in a.sizes:
        det.imgsz = sz
        p = det.predict(imgs)
        e = evaluate_detections([r.boxes for r in recs], [r.classes for r in recs], p, len(ds.names))
        ok = np.array(e["n_gt"]) >= MIN_GT
        sweep[sz] = {"map50_evaluable": float(np.nanmean(e["ap50"][ok])), "map50_95_evaluable": float(np.nanmean(e["ap50_95"][ok])),
                     "ap50": e["ap50"], "evaluable": [ds.names[k] for k in np.where(ok)[0]]}
        print(f"imgsz {sz}: evaluable mAP50={sweep[sz]['map50_evaluable']:.3f} mAP50-95={sweep[sz]['map50_95_evaluable']:.3f}")
    out["resolution_sweep_evaluable"] = sweep
save_json(out, tab_dir / "operating_points.json")
for n, rows in out["methods"].items():
    print(METHOD_LABELS[n])
    for r in rows:
        print(f"  recall>={r['target_recall']:.2f}: recall {r['recall']:.3f} false-reject {r['false_reject_rate']:.3f} "
              f"[{r['frr_ci95'][0]:.2f}, {r['frr_ci95'][1]:.2f}]")
