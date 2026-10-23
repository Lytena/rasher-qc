"""Step 4 - sensitivity / robustness to acquisition conditions.

Each perturbation in :data:`rasherqc.robustness.SWEEPS` (exposure, saturation,
glare, blur, noise, JPEG) is applied at increasing severity to every **test**
image. The thresholds chosen on validation in step 3 are kept fixed, as they
would be on a production line, and we record:

* FAIL recall and false-reject rate of the tray decision;
* mAP@0.5 of the detector (our metric code).

Outputs: results/<tag>/tables/robustness.json, figures/robustness.png

Usage: python scripts/04_robustness.py --data data/<dataset> [--methods yolo_pretrained rf_baseline]
"""

from __future__ import annotations

import pickle

import numpy as np
from _common import METHOD_COLOURS, METHOD_LABELS, load_json, out_dirs, parse, plt, save_json
from PIL import Image

from rasherqc import config, robustness
from rasherqc.baseline import load_feature_images
from rasherqc.dataset import load_dataset, tray_fail_label
from rasherqc.features import FEAT_WH, extract_batch
from rasherqc.metrics import binary_confusion, evaluate_detections, tray_scores


def extra(ap):
    """Add this script's robustness options (image size, device, ...)."""
    ap.add_argument("--imgsz", type=int, default=config.DEFAULT_IMGSZ)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--methods", nargs="+", default=["yolo_pretrained", "rf_baseline"])
    ap.add_argument("--limit", type=int, default=0, help="use only the first N test images (0 = all)")


a = parse(__doc__.split("\n")[0], extra)
fig_dir, tab_dir, model_dir = out_dirs(a.tag)
ev = load_json(tab_dir / "evaluation.json")
ds = load_dataset(a.data)
recs = ds.split("test")[: a.limit or None]
y = np.array([tray_fail_label(r, ds.defect_ids) for r in recs])
imgs = [np.asarray(Image.open(r.path).convert("RGB")) for r in recs]
res: dict = {}
for m in a.methods:
    if m not in ev["methods"]:
        print("skip", m)
        continue
    ft = ev["methods"][m]["tray"]["fail_conf"]
    if m.startswith("yolo"):
        from rasherqc.detector import Detector

        det = Detector(model_dir / f"{m}.pt", imgsz=a.imgsz, device=a.device)
    else:
        bl = pickle.loads((model_dir / "rf_baseline.pkl").read_bytes())
    res[m] = {}
    for name, (fn, levels) in robustness.SWEEPS.items():
        rows = []
        for lv in levels:
            pert = [fn(im, lv) for im in imgs]
            if m.startswith("yolo"):
                p = det.predict(pert)
                s = tray_scores(p, ds.defect_ids)
                mp = evaluate_detections([r.boxes for r in recs], [r.classes for r in recs], p, len(ds.names),
                                         iou_thresholds=[0.5])["map50"]
            else:
                small = [np.asarray(Image.fromarray(x).resize(FEAT_WH, Image.BOX)) for x in pert]
                s = bl.predict_fail_proba(extract_batch(small, n_jobs=2))
                mp = None
            c = binary_confusion(y, s >= ft)
            rows.append({"level": lv, "fail_recall": c["fail_recall"], "false_reject_rate": c["false_reject_rate"], "map50": mp})
            print(f"{m:16s} {name:10s} {lv:>6} recall={c['fail_recall']:.3f} FRR={c['false_reject_rate']:.3f} mAP50={mp}")
        res[m][name] = rows
    save_json(res, tab_dir / "robustness.json")

fig, axs = plt.subplots(2, 3, figsize=(10, 5.4))
for ax, name in zip(axs.ravel(), robustness.SWEEPS):
    for m in res:
        lv = [r["level"] for r in res[m][name]]
        xs = np.arange(len(lv))
        ax.plot(xs, [r["fail_recall"] for r in res[m][name]], "o-", color=METHOD_COLOURS[m], label=f"{METHOD_LABELS[m]} recall")
        ax.plot(xs, [r["false_reject_rate"] for r in res[m][name]], "x--", color=METHOD_COLOURS[m], label=f"{METHOD_LABELS[m]} false reject", alpha=0.7)
        ax.set_xticks(xs, [str(v) for v in lv])
    ax.set_title(name)
    ax.set_ylim(-0.02, 1.02)
axs[0, 0].legend(fontsize=6)
fig.suptitle("Tray decision under perturbation (test split, thresholds fixed from validation)")
fig.tight_layout()
fig.savefig(fig_dir / "robustness.png")
