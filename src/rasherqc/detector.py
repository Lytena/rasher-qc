"""YOLOv8 object detector: training, inference and conversion of outputs.

Method summary
--------------
YOLOv8 is a one-stage, anchor-free detector. A convolutional backbone
(CSP-Darknet with C2f blocks) extracts features at strides 8, 16 and 32; a
feature-pyramid neck fuses them; a decoupled head predicts, for every grid
cell, class probabilities and a box (as a distribution over distances to the
four sides). Training minimises a weighted sum of classification BCE, CIoU
box loss and distribution-focal loss, with "task-aligned" assignment of
cells to GT boxes. Non-maximum suppression (NMS) removes duplicate boxes.

Why it was chosen (alternatives in the report): single-pass inference fits a
real-time line; the nano variant (3.0 M parameters, ~8 GFLOPs at 640 px)
runs on a CPU; pretrained COCO weights give transfer learning, important with
~1.4 k training images; the library is widely used and reproducible.

The ``ultralytics`` import is deferred so the rest of the package (metrics,
baseline, tests) works without it.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import config
from .metrics import Detections


def train(
    data_yaml: str | Path,
    out_dir: str | Path,
    name: str,
    weights: str | Path | None = config.PRETRAINED_WEIGHTS,
    imgsz: int = config.DEFAULT_IMGSZ,
    epochs: int = 40,
    batch: int = 16,
    seed: int = config.SEED,
    device: str = "cpu",
    workers: int = 2,
    patience: int = 15,
    **overrides,
) -> Path:
    """Train (fine-tune) a YOLOv8n detector.

    Args:
        data_yaml: Normalised dataset yaml from :func:`rasherqc.dataset.load_dataset`.
        out_dir: Directory where the run folder is created.
        name: Run name (sub-folder of ``out_dir``).
        weights: Pretrained ``.pt`` to fine-tune, or ``None`` to train the
            same architecture from random initialisation (``yolov8n.yaml``).
        imgsz: Training resolution (longest side, px; multiple of 32).
        epochs: Maximum epochs.
        batch: Batch size.
        seed: Random seed (``deterministic=True`` is also set).
        device: ``"cpu"``, ``"0"`` (first GPU) or ``"mps"``.
        workers: Data-loader workers.
        patience: Early-stopping patience on validation fitness
            (0.1 mAP50 + 0.9 mAP50-95).
        **overrides: Any other Ultralytics training argument.

    Returns:
        Path to the best checkpoint (``weights/best.pt``).
    """
    import sys

    from ultralytics import YOLO

    if sys.platform == "win32":
        # Windows starts data-loader workers by re-importing the calling script,
        # which re-runs it; in-process loading avoids this (GPU still dominates).
        workers = 0
    model = YOLO(str(weights) if weights is not None else "yolov8n.yaml")
    args = dict(
        data=str(data_yaml), project=str(Path(out_dir).resolve()), name=name, exist_ok=True,
        imgsz=imgsz, epochs=epochs, batch=batch, seed=seed, deterministic=True, device=device,
        workers=workers, patience=patience, plots=True, verbose=False,
        # geometric augmentations suited to a top-down camera; the dataset's own
        # exposure/saturation augmentation is complemented by HSV jitter.
        fliplr=0.5, flipud=0.5, degrees=0.0, mosaic=1.0, close_mosaic=5,
        pretrained=weights is not None,
    )
    args.update(overrides)
    model.train(**args)
    return Path(out_dir).resolve() / name / "weights" / "best.pt"


class Detector:
    """Thin inference wrapper returning :class:`~rasherqc.metrics.Detections`.

    Args:
        weights: Trained ``.pt`` checkpoint.
        imgsz: Inference resolution (px).
        conf: Minimum confidence kept (low by default so that thresholds can
            be applied afterwards; mAP needs the full ranking).
        iou: NMS IoU threshold.
        device: Inference device.
    """

    def __init__(self, weights: str | Path, imgsz: int = config.DEFAULT_IMGSZ, conf: float = 0.001,
                 iou: float = 0.6, device: str = "cpu"):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.imgsz, self.conf, self.iou, self.device = imgsz, conf, iou, device
        self.names: list[str] = [self.model.names[i] for i in sorted(self.model.names)]

    def predict(self, images: list, batch: int = 16) -> list[Detections]:
        """Run detection.

        Args:
            images: File paths or (H, W, 3) **RGB** uint8 arrays.
            batch: Images per forward pass.

        Returns:
            One :class:`Detections` per image, boxes in original pixel units.
        """
        out: list[Detections] = []
        for i in range(0, len(images), batch):
            chunk = [im[..., ::-1].copy() if isinstance(im, np.ndarray) else str(im) for im in images[i : i + batch]]
            # Ultralytics expects BGR arrays (OpenCV convention), hence the flip.
            res = self.model.predict(chunk, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                     device=self.device, verbose=False, max_det=100)
            for r in res:
                b = r.boxes
                out.append(Detections(b.xyxy.cpu().numpy().astype(float), b.conf.cpu().numpy().astype(float),
                                      b.cls.cpu().numpy().astype(int)))
        return out

    def latency(self, image: np.ndarray, n: int = 30, warmup: int = 5) -> dict:
        """Single-image end-to-end latency (pre-process + inference + NMS).

        Args:
            image: (H, W, 3) RGB uint8 array.
            n: Timed repetitions.
            warmup: Untimed warm-up runs (exclude one-off initialisation).

        Returns:
            Dict with ``mean_ms``, ``std_ms``, ``p95_ms`` and Ultralytics'
            per-stage split (``pre_ms``, ``inf_ms``, ``post_ms``).
        """
        bgr = image[..., ::-1].copy()
        for _ in range(warmup):
            self.model.predict(bgr, imgsz=self.imgsz, conf=0.25, device=self.device, verbose=False)
        t, stages = [], []
        for _ in range(n):
            t0 = time.perf_counter()
            r = self.model.predict(bgr, imgsz=self.imgsz, conf=0.25, device=self.device, verbose=False)
            t.append((time.perf_counter() - t0) * 1e3)
            stages.append([r[0].speed["preprocess"], r[0].speed["inference"], r[0].speed["postprocess"]])
        t = np.array(t)
        s = np.array(stages).mean(0)
        return {"mean_ms": float(t.mean()), "std_ms": float(t.std()), "p95_ms": float(np.percentile(t, 95)),
                "pre_ms": float(s[0]), "inf_ms": float(s[1]), "post_ms": float(s[2])}
