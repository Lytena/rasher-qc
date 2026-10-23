"""Tray PASS / REVIEW / FAIL decision and defect-extent measures.

Decision rule (per tray)
------------------------
Let ``s`` be the highest confidence among detected *defect* boxes (0 if none):

* ``FAIL``   if ``s >= fail_conf``
* ``REVIEW`` if ``review_low <= s < fail_conf``  (sent to a person)
* ``PASS``   if ``s < review_low``

An image outside the supported input domain (see :mod:`rasherqc.io`) is
``UNSUPPORTED_INPUT`` and is never passed automatically.

Asymmetric costs justify the review band: passing a defective tray (an
*escape*) can mean a leaking pack reaching a customer; failing a good tray
only costs a re-check. Thresholds are selected on the validation split
(``scripts/03_evaluate.py``), never on the test split.

Defect extent
-------------
For each detected defect the box area as a fraction of the image area is
reported. It is an upper bound on the true defect area (boxes are axis-
aligned) and is meant for ranking severity, not for measurement in mm².
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import config
from .metrics import Detections


@dataclass
class TrayResult:
    """Decision for one tray image.

    Attributes:
        status: ``"PASS"``, ``"REVIEW"``, ``"FAIL"`` or ``"UNSUPPORTED_INPUT"``.
        score: Highest defect confidence (0-1), ``None`` for unsupported input.
        defects: One dict per detected defect at or above ``review_low``:
            ``class``, ``confidence``, ``box_xyxy`` (px) and ``area_frac``.
        reasons: Why the input was rejected (unsupported input only).
    """

    status: str
    score: float | None
    defects: list[dict] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serialisable representation."""
        return asdict(self)


def decide(
    det: Detections,
    names: list[str],
    defect_ids: list[int],
    image_wh: tuple[int, int],
    fail_conf: float = config.DEFAULT_FAIL_CONF,
    review_low: float = config.DEFAULT_REVIEW_LOW,
) -> TrayResult:
    """Apply the decision rule to one image's detections.

    Args:
        det: Detections for the image.
        names: Class names (index = class id).
        defect_ids: Class ids that are defects.
        image_wh: Image (width, height) in pixels, for the extent measure.
        fail_conf: FAIL threshold.
        review_low: Lower bound of the REVIEW band (``<= fail_conf``).

    Returns:
        A :class:`TrayResult`.

    Raises:
        ValueError: If ``review_low > fail_conf``.
    """
    if review_low > fail_conf:
        raise ValueError("review_low must not exceed fail_conf")
    w, h = image_wh
    # Report every defect an operator might want to look at (REVIEW band and up),
    # most confident first; weaker detections are noise for the operator.
    keep = np.isin(det.classes, defect_ids) & (det.scores >= review_low)
    defects = []
    for b, s, c in sorted(zip(det.boxes[keep], det.scores[keep], det.classes[keep]), key=lambda t: -t[1]):
        defects.append({
            "class": names[int(c)],
            "confidence": round(float(s), 4),
            "box_xyxy": [round(float(v), 1) for v in b],
            "area_frac": round(float((b[2] - b[0]) * (b[3] - b[1]) / (w * h)), 5),
        })
    # The tray is judged by its single most confident defect: one real defect
    # is enough to reject a pack, however many clean regions it has.
    sel = np.isin(det.classes, defect_ids)
    score = float(det.scores[sel].max()) if sel.any() else 0.0
    status = "FAIL" if score >= fail_conf else "REVIEW" if score >= review_low else "PASS"
    return TrayResult(status, round(score, 4), defects)
