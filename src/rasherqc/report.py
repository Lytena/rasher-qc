"""Inspection report: per-tray JSON, annotated images and an HTML summary.

Output contract (for a folder of tray images)
---------------------------------------------
``<out>/results.json``
    list of ``{"image": str, "status": "PASS"|"REVIEW"|"FAIL"|"UNSUPPORTED_INPUT",
    "score": float|null, "defects": [{"class", "confidence", "box_xyxy",
    "area_frac"}], "reasons": [str]}``. ``box_xyxy`` is in pixels of the
    original image, origin top-left.
``<out>/annotated/<image>.jpg``
    the image with defect boxes drawn (colour per class) and the decision.
``<out>/summary.html``
    counts per status and per defect class, plus a table of every tray, for
    the line supervisor.
"""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import config
from .decision import TrayResult, decide
from .io import InvalidInputError, read_image, validate_image

_COLOURS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240)]
_STATUS_COL = {"PASS": (40, 160, 60), "REVIEW": (230, 160, 0), "FAIL": (210, 30, 30), "UNSUPPORTED_INPUT": (120, 120, 120)}


def annotate(img: np.ndarray, res: TrayResult, names: list[str]) -> Image.Image:
    """Draw defect boxes and the tray decision onto a copy of ``img``."""
    im = Image.fromarray(img).copy()
    d = ImageDraw.Draw(im)
    for df in res.defects:
        c = _COLOURS[names.index(df["class"]) % len(_COLOURS)] if df["class"] in names else (255, 255, 255)
        d.rectangle(df["box_xyxy"], outline=c, width=3)
        d.text((df["box_xyxy"][0] + 3, df["box_xyxy"][1] + 2), f"{df['class']} {df['confidence']:.2f}", fill=c)
    d.rectangle([0, 0, 230, 22], fill=_STATUS_COL[res.status])
    d.text((6, 5), f"{res.status}  score={res.score if res.score is not None else '-'}", fill=(255, 255, 255))
    return im


def inspect_folder(detector, images: list[Path], out_dir: str | Path, defect_ids: list[int],
                   fail_conf: float = config.DEFAULT_FAIL_CONF, review_low: float = config.DEFAULT_REVIEW_LOW) -> list[dict]:
    """Validate, detect, decide and report for a list of image files.

    Args:
        detector: A :class:`rasherqc.detector.Detector`.
        images: Image file paths.
        out_dir: Output folder (created).
        defect_ids: Class ids that are defects.
        fail_conf: FAIL threshold.
        review_low: Lower bound of the REVIEW band.

    Returns:
        The list written to ``results.json``.
    """
    out = Path(out_dir)
    (out / "annotated").mkdir(parents=True, exist_ok=True)
    rows = []
    for p in images:
        p = Path(p)
        try:
            rgb, val = validate_image(read_image(p))
        except InvalidInputError as exc:
            rgb, reasons = None, [str(exc)]
        else:
            reasons = val.reasons
        if rgb is None:
            res = TrayResult("UNSUPPORTED_INPUT", None, [], reasons)
        else:
            det = detector.predict([rgb])[0]
            res = decide(det, detector.names, defect_ids, (rgb.shape[1], rgb.shape[0]), fail_conf, review_low)
            annotate(rgb, res, detector.names).save(out / "annotated" / (p.stem + ".jpg"), quality=90)
        rows.append({"image": p.name, **res.to_dict()})
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    write_summary(rows, out / "summary.html", fail_conf, review_low)
    return rows


def write_summary(rows: list[dict], path: Path, fail_conf: float, review_low: float) -> None:
    """Write a self-contained HTML summary of an inspection run."""
    st = Counter(r["status"] for r in rows)
    cls = Counter(d["class"] for r in rows for d in r["defects"])
    lines = []
    for r in rows:
        score = "" if r["score"] is None else f"{r['score']:.2f}"
        detail = ", ".join(d["class"] for d in r["defects"]) or "; ".join(r["reasons"])
        lines.append(f"<tr><td>{html.escape(r['image'])}</td><td class='{r['status']}'>{r['status']}</td>"
                     f"<td>{score}</td><td>{html.escape(detail)}</td></tr>")
    trs = "\n".join(lines)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Tray inspection summary</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#222}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 8px}}.PASS{{color:#1a7f37}}.FAIL{{color:#c62828;font-weight:600}}
.REVIEW{{color:#b26a00}}.UNSUPPORTED_INPUT{{color:#666}}</style></head><body>
<h1>Tray inspection summary</h1>
<p>{len(rows)} trays &middot; FAIL if defect confidence &ge; {fail_conf}; REVIEW if &ge; {review_low}.</p>
<h2>Decisions</h2><ul>{''.join(f'<li>{k}: {v}</li>' for k, v in sorted(st.items()))}</ul>
<h2>Detected defects (&ge; review threshold)</h2><ul>{''.join(f'<li>{html.escape(k)}: {v}</li>' for k, v in sorted(cls.items())) or '<li>none</li>'}</ul>
<h2>All trays</h2><table><tr><th>Image</th><th>Status</th><th>Score</th><th>Defects / reasons</th></tr>
{trs}</table></body></html>"""
    path.write_text(page)
