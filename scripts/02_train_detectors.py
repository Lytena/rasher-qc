"""Step 2 - train the detectors that are compared.

* ``yolo_pretrained``: YOLOv8n initialised from COCO weights (transfer learning) -- main method
* ``yolo_scratch``:    the same architecture from random initialisation -- tests whether
  pretraining matters for a dataset of this size

Both use identical data, resolution, epochs, augmentation and seed, so the only
difference is the initialisation. Model selection (best epoch) uses the
validation split only. Outputs: models/<tag>/<name>.pt, the Ultralytics run
folder under results/<tag>/runs/<name>/ (results.csv, curves) and
results/<tag>/tables/training.json (wall time, epochs run).

Usage: python scripts/02_train_detectors.py --data data/<dataset> [--epochs 40] [--imgsz 416] [--device 0]
"""

from __future__ import annotations

import shutil
import time

from _common import load_json, out_dirs, parse, save_json

from rasherqc import config
from rasherqc.dataset import load_dataset
from rasherqc.detector import train


def extra(ap):
    """Add this script's training options (epochs, batch size, device, ...)."""
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=config.DEFAULT_IMGSZ)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--which", choices=["both", "pretrained", "scratch"], default="both")


a = parse(__doc__.split("\n")[0], extra)
_, tab_dir, model_dir = out_dirs(a.tag)
ds = load_dataset(a.data)
log_path = tab_dir / "training.json"
log = load_json(log_path) if log_path.is_file() else {}
runs = {"pretrained": config.PRETRAINED_WEIGHTS, "scratch": None}
for which, weights in runs.items():
    if a.which not in ("both", which):
        continue
    name = f"yolo_{which}"
    t0 = time.perf_counter()
    best = train(ds.yolo_yaml, config.RESULTS_DIR / a.tag / "runs", name, weights=weights, imgsz=a.imgsz,
                 epochs=a.epochs, batch=a.batch, device=a.device)
    dt = time.perf_counter() - t0
    shutil.copy(best, model_dir / f"{name}.pt")
    rows = (config.RESULTS_DIR / a.tag / "runs" / name / "results.csv").read_text().strip().splitlines()
    log[name] = {"wall_time_s": dt, "epochs_run": len(rows) - 1, "imgsz": a.imgsz, "batch": a.batch,
                 "device": a.device, "epochs_max": a.epochs}
    save_json(log, log_path)
    print(f"{name}: {dt / 60:.1f} min, {len(rows) - 1} epochs -> {model_dir / (name + '.pt')}")
