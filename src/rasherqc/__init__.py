"""rasherqc -- automated visual inspection of packaged pork rashers.

Engineering problem
-------------------
Sliced-bacon packing lines place shingled rashers in a tray and heat-seal a
lidding film over it. Placement errors (*loose* or *twisted* rashers) and
packaging errors (*unsealed* flanges, *wrinkled* or damaged film) cause
customer rejects, leaking packs and shortened shelf life. Manual inspection
at line speed is tiring and inconsistent. This package detects and localises
those defects in a top-view RGB image and turns the detections into a tray
decision -- **PASS**, **REVIEW** (send to a person) or **FAIL**.

Top-level inputs
----------------
* RGB tray image, uint8, 720 x 540 px in the provided dataset (other sizes
  accepted within the limits of :mod:`rasherqc.io`); one tray, top view.
* For training/evaluation: a YOLO- or COCO-format export with bounding boxes
  (pixels / normalised 0-1) for the classes *loose-meat, packaging-error,
  twisted-meat, unsealed, wrinkle* and predefined train/valid/test splits
  (:mod:`rasherqc.dataset`).

Top-level outputs
-----------------
* Per tray: decision, tray defect score (0-1), and a list of defects with
  class, confidence (0-1), box (pixels, x1 y1 x2 y2) and box area as a
  fraction of the image (:mod:`rasherqc.decision`).
* Per run: ``results.json``, annotated images, ``summary.html``
  (:mod:`rasherqc.report`).
* Invalid / unsupported images: decision ``UNSUPPORTED_INPUT`` with reasons;
  never an automatic PASS.

Modules
-------
* :mod:`rasherqc.io` -- image reading and input-domain validation
* :mod:`rasherqc.dataset` -- dataset loading, tray labels, leakage check
* :mod:`rasherqc.synthetic` -- synthetic verification data with exact labels
* :mod:`rasherqc.detector` -- YOLOv8 training and inference (main method)
* :mod:`rasherqc.features`, :mod:`rasherqc.baseline` -- classical baseline
* :mod:`rasherqc.metrics` -- IoU, AP/mAP, tray metrics (written from scratch)
* :mod:`rasherqc.decision` -- PASS/REVIEW/FAIL rule and defect extent
* :mod:`rasherqc.robustness` -- perturbations for sensitivity analysis
* :mod:`rasherqc.performance` -- FLOPs, arithmetic intensity, timing
* :mod:`rasherqc.report`, :mod:`rasherqc.cli` -- reports and command line
"""

__version__ = "1.0.0"
