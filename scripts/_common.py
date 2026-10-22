"""Shared helpers for the experiment scripts (argument parsing, output folders, plotting style)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rasherqc import config  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

#: Colours used consistently for the three compared methods.
METHOD_COLOURS = {"yolo_pretrained": "#2a6fdb", "yolo_scratch": "#e08a1e", "rf_baseline": "#5a5a5a"}
METHOD_LABELS = {"yolo_pretrained": "YOLOv8n (COCO-pretrained)", "yolo_scratch": "YOLOv8n (from scratch)",
                 "rf_baseline": "Features + random forest"}


def parse(desc: str, extra=None) -> argparse.Namespace:
    """Common ``--data`` / ``--tag`` arguments; ``tag`` names the results folder."""
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--data", default=str(config.DATA_DIR / "synthetic"), help="dataset root")
    ap.add_argument("--tag", default=None, help="results sub-folder (default: dataset folder name)")
    if extra:
        extra(ap)
    a = ap.parse_args()
    a.tag = a.tag or Path(a.data).name
    return a


def out_dirs(tag: str) -> tuple[Path, Path, Path]:
    """(figures, tables, models) folders for a dataset tag, created if needed."""
    f = config.RESULTS_DIR / tag / "figures"
    t = config.RESULTS_DIR / tag / "tables"
    m = config.MODEL_DIR / tag
    for d in (f, t, m):
        d.mkdir(parents=True, exist_ok=True)
    return f, t, m


def save_json(obj, path: Path) -> None:
    """Write JSON, converting NumPy types."""
    def conv(o):
        if isinstance(o, float):
            return None if np.isnan(o) else o
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return [conv(v) for v in o.tolist()] if o.dtype.kind == "f" else o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=2, default=conv))


def load_json(path: Path):
    return json.loads(Path(path).read_text())
