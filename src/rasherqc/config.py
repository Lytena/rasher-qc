"""Project-wide constants: defect classes, input limits, decision settings, paths.

The class list of the real dataset is read from its ``data.yaml`` / COCO
file at run time (see :mod:`rasherqc.dataset`). The constants here only
describe the synthetic verification dataset and the defaults of the tray
PASS/FAIL decision.
"""

from __future__ import annotations

from pathlib import Path

#: Defect classes, identical (names and order) to the provided Roboflow dataset
#: "Pork Rasher Error (Packaging)" so that synthetic and real labels align.
#: Every one of them is a defect.
SYNTH_CLASSES: tuple[str, ...] = ("loose-meat", "packaging-error", "twisted-meat", "unsealed", "wrinkle")

#: Class names that describe acceptable objects rather than defects. If the
#: real dataset also annotates e.g. every rasher or the tray, those classes
#: are listed here so they never cause a FAIL. Matching is case-insensitive
#: and ignores spaces/hyphens/underscores.
NON_DEFECT_NAMES: tuple[str, ...] = ("rasher", "rashers", "meat", "tray", "good", "ok", "pass", "normal", "sealed")

# --------------------------------------------------------------------------
# Input-domain limits used by :func:`rasherqc.io.validate_image`.
# --------------------------------------------------------------------------
#: Smallest accepted image side (px). Below this a wrinkle (~2-4 mm wide)
#: falls under one pixel on a ~300 mm tray.
MIN_SIDE_PX: int = 240
#: Largest accepted image side (px); guards against memory blow-up.
MAX_SIDE_PX: int = 8192
#: Accepted mean grey level (0-255). Outside it the image is grossly
#: under/over-exposed.
MEAN_INTENSITY_RANGE: tuple[float, float] = (20.0, 235.0)
#: Maximum fraction of clipped (0 or 255) pixels before rejection.
MAX_CLIPPED_FRACTION: float = 0.25
#: Native image size of the provided dataset (width, height).
NATIVE_WH: tuple[int, int] = (720, 540)

# --------------------------------------------------------------------------
# Detector / decision defaults
# --------------------------------------------------------------------------
#: Training/inference resolution (longest side, px) used for the CPU runs.
DEFAULT_IMGSZ: int = 416
#: A tray FAILS when any defect is detected with confidence >= this value.
#: Chosen on the validation set (see ``scripts/04_tray_decision.py``).
DEFAULT_FAIL_CONF: float = 0.25
#: Trays whose highest defect confidence lies in [REVIEW_LOW, FAIL_CONF)
#: are routed to a human ("REVIEW") instead of passing automatically.
DEFAULT_REVIEW_LOW: float = 0.10
#: IoU above which a prediction matches a ground-truth box (PASCAL VOC rule).
IOU_MATCH: float = 0.5

#: Assumed line rate for the real-time check: trays per minute.
LINE_RATE_TRAYS_PER_MIN: float = 60.0

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "data"
RESULTS_DIR: Path = REPO_ROOT / "results"
FIG_DIR: Path = RESULTS_DIR / "figures"
TABLE_DIR: Path = RESULTS_DIR / "tables"
MODEL_DIR: Path = REPO_ROOT / "models"
#: COCO-pretrained YOLOv8n weights (downloaded once, see README).
PRETRAINED_WEIGHTS: Path = MODEL_DIR / "yolov8n_coco_pretrained.pt"

#: Global seed used by all experiments for repeatability.
SEED: int = 42
