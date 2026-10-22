"""Command-line interface.

Examples::

    rasherqc synth   --out data/synthetic                # make verification data
    rasherqc train   --data data/pork-rasher --name yolo_pre --epochs 40
    rasherqc inspect --weights models/best.pt --data data/pork-rasher --images path/to/folder --out inspection/

``inspect`` writes ``results.json``, annotated images and ``summary.html``
(see :mod:`rasherqc.report`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config


def main(argv: list[str] | None = None) -> None:
    """Entry point of the ``rasherqc`` command."""
    ap = argparse.ArgumentParser(prog="rasherqc", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate the synthetic verification dataset")
    s.add_argument("--out", default=str(config.DATA_DIR / "synthetic"))
    s.add_argument("--n-train", type=int, default=600)
    s.add_argument("--n-valid", type=int, default=150)
    s.add_argument("--n-test", type=int, default=150)
    s.add_argument("--seed", type=int, default=config.SEED)

    t = sub.add_parser("train", help="fine-tune YOLOv8n on a dataset")
    t.add_argument("--data", required=True, help="dataset root (YOLO or COCO export)")
    t.add_argument("--name", default="yolo_pretrained")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--imgsz", type=int, default=config.DEFAULT_IMGSZ)
    t.add_argument("--scratch", action="store_true", help="random initialisation instead of COCO weights")
    t.add_argument("--device", default="cpu")

    i = sub.add_parser("inspect", help="inspect a folder of tray images")
    i.add_argument("--weights", required=True)
    i.add_argument("--data", required=True, help="dataset root (for class names / defect classes)")
    i.add_argument("--images", required=True, help="folder of images")
    i.add_argument("--out", default="inspection")
    i.add_argument("--imgsz", type=int, default=config.DEFAULT_IMGSZ)
    i.add_argument("--fail-conf", type=float, default=config.DEFAULT_FAIL_CONF)
    i.add_argument("--review-low", type=float, default=config.DEFAULT_REVIEW_LOW)

    a = ap.parse_args(argv)
    if a.cmd == "synth":
        from .synthetic import generate_dataset

        df = generate_dataset(a.out, a.n_train, a.n_valid, a.n_test, a.seed)
        print(f"wrote {len(df)} images to {a.out}")
    elif a.cmd == "train":
        from .dataset import load_dataset
        from .detector import train

        ds = load_dataset(a.data)
        best = train(ds.yolo_yaml, config.RESULTS_DIR / "runs", a.name,
                     weights=None if a.scratch else config.PRETRAINED_WEIGHTS,
                     imgsz=a.imgsz, epochs=a.epochs, device=a.device)
        print(f"best weights: {best}")
    elif a.cmd == "inspect":
        from .dataset import IMG_EXTS, load_dataset
        from .detector import Detector
        from .report import inspect_folder

        ds = load_dataset(a.data)
        det = Detector(a.weights, imgsz=a.imgsz)
        imgs = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in IMG_EXTS)
        rows = inspect_folder(det, imgs, a.out, ds.defect_ids, a.fail_conf, a.review_low)
        print(f"inspected {len(rows)} images -> {a.out}/summary.html")


if __name__ == "__main__":
    main()
