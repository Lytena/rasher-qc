"""Procedural generator of annotated pork-rasher tray images (verification data).

Purpose
-------
The synthetic set is the project's *manufactured solution*: every image has
exactly known defect boxes, so the detector, the metric code and the tray
decision can be verified against a perfect ground truth, and the influence
of single factors (defect size, glare, exposure) can be isolated. It is
written in the same YOLO layout as the provided dataset (``data.yaml``,
``train/valid/test/{images,labels}``) so every script runs unchanged on
either. It is **not** evidence of performance on real trays.

Rendered scene (720 x 540 px, top view)
---------------------------------------
* Conveyor belt; a black (PET) or white tray with a rim/flange; a glossy
  heat-seal band where the lidding film is bonded to the flange.
* 4-6 streaky rashers laid shingled (overlapping) along the tray: pink lean
  with lengthwise fat bands, a fat/rind edge, mottling, edge shading and drop
  shadows on the rasher underneath.
* Tight lidding film: slight haze plus soft, broad light-bar **reflections**
  (present on defect-free trays too -- the main confounder for wrinkles).

Defect classes (each instance gets one bounding box)
----------------------------------------------------
0. ``loose-meat`` -- a rasher or fragment out of the shingle pattern, at a
   strongly different angle and/or overhanging the arrangement.
1. ``packaging-error`` -- breached lidding film: a jagged tear with bright
   curled film lips and a matte opening (no film gloss).
2. ``twisted-meat`` -- a half-twist along a rasher: width collapses, the fold
   is darker, and the fat edge swaps sides beyond the twist.
3. ``unsealed`` -- a section of flange without the seal band, with the film
   lifted: dark gap and a bright crumpled film edge.
4. ``wrinkle`` -- a sharp, narrow, curved bright crease with a parallel shadow.

Acquisition variation: exposure gain, saturation, colour temperature, sensor
noise and slight defocus, mirroring the exposure/saturation augmentation of
the provided Version 4 dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter

from . import config
from .dataset import write_yolo_split

W, H = config.NATIVE_WH


@dataclass
class Scene:
    """A rendered image with its annotations.

    Attributes:
        image: (H, W, 3) uint8 RGB.
        boxes: (N, 4) pixel ``x1, y1, x2, y2``.
        classes: (N,) class ids into :data:`~rasherqc.config.SYNTH_CLASSES`.
        glare: Strength of the film reflections (0-1), a nuisance factor.
    """

    image: np.ndarray
    boxes: np.ndarray
    classes: np.ndarray
    glare: float


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _noise(rng, shape, sigma):
    n = gaussian_filter(rng.standard_normal(shape), sigma)
    return (n - n.mean()) / (n.std() + 1e-9)


def _blend(img, colour, alpha):
    a = np.clip(alpha, 0, 1)[..., None]
    return img * (1 - a) + np.asarray(colour, float) * a


def _polyline_cov(pts: np.ndarray, width: float, pad: float | None = None) -> tuple[np.ndarray, tuple[slice, slice]]:
    """Anti-aliased coverage of a thick polyline, computed in its bounding window.

    Returns:
        ``(cov, (ys, xs))`` where ``cov`` is the coverage (0-1) inside the
        window given by the slices ``ys, xs`` of the full image.
    """
    pad = width + 3 if pad is None else pad
    x0 = int(max(np.floor(pts[:, 0].min() - pad), 0))
    x1 = int(min(np.ceil(pts[:, 0].max() + pad), W))
    y0 = int(max(np.floor(pts[:, 1].min() - pad), 0))
    y1 = int(min(np.ceil(pts[:, 1].max() + pad), H))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0)), (slice(0, 0), slice(0, 0))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)
    d = np.full(yy.shape, np.inf)
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        vx, vy = bx - ax, by - ay
        t = np.clip(((xx - ax) * vx + (yy - ay) * vy) / (vx * vx + vy * vy + 1e-9), 0, 1)
        d = np.minimum(d, np.hypot(xx - ax - t * vx, yy - ay - t * vy))
    return np.clip(width / 2 + 0.5 - d, 0, 1), (slice(y0, y1), slice(x0, x1))


def _curve(rng, start, n, step, turn, ang=None):
    pts = [np.asarray(start, float)]
    a = rng.uniform(0, 2 * np.pi) if ang is None else ang
    for _ in range(n):
        a += rng.normal(0, turn)
        pts.append(pts[-1] + step * np.array([np.cos(a), np.sin(a)]))
    return np.array(pts)


def _bbox(mask: np.ndarray, pad: int = 0) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 15:
        return None
    return [max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + 1 + pad, W), min(ys.max() + 1 + pad, H)]


# ----------------------------------------------------------------------------
# rasher
# ----------------------------------------------------------------------------
def _draw_rasher(img, rng, cx, cy, length, width, ang, twist=None, flip=False):
    """Draw one rasher (in place) and return its full mask and twist mask.

    Args:
        img: (H, W, 3) float image, modified in place.
        rng: Random generator.
        cx, cy: Centre (px).
        length, width: Rasher size (px).
        ang: Orientation of the long axis (rad).
        twist: ``None`` or ``(u0, half_len)``: position of a half-twist along
            the rasher (px from centre) and its half length.
        flip: Mirror the fat edge to the other side.

    Returns:
        ``(mask, twist_mask)`` full-image bool arrays.
    """
    r = int(length / 2 + width) + 4
    x0, x1 = max(int(cx - r), 0), min(int(cx + r), W)
    y0, y1 = max(int(cy - r), 0), min(int(cy + r), H)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)
    ca, sa = np.cos(ang), np.sin(ang)
    u = (xx - cx) * ca + (yy - cy) * sa  # along
    v = -(xx - cx) * sa + (yy - cy) * ca  # across
    bend = rng.uniform(2, 8) * np.sin(2 * np.pi * u / length * rng.uniform(0.6, 1.4) + rng.uniform(0, 6.3))
    hw = width / 2 * (1 + 0.06 * np.sin(u / rng.uniform(9, 16)) + 0.04 * np.sin(u / 5.3 + 1))
    wf = np.ones_like(u)
    side = np.ones_like(u)
    tmask_local = np.zeros(u.shape, bool)
    if twist is not None:
        u0, tl = twist
        s = (u - u0) / tl
        inside = np.abs(s) < 1
        wf = np.where(inside, np.maximum(np.abs(np.sin(np.pi * s / 2)), 0.22), 1.0)
        side = np.where(u > u0, -1.0, 1.0)
        tmask_local = np.abs(u - u0) < tl * 1.25
    hw_eff = hw * wf
    vn = (v - bend) / hw_eff * side * (-1 if flip else 1)
    end = length / 2 + 4 * np.sin(v / 7)
    mask = (np.abs(v - bend) <= hw_eff) & (np.abs(u) <= end)

    # drop shadow onto what is underneath (offset by a few px)
    sh = np.roll(np.roll(mask, 4, 0), 3, 1) & ~mask
    win = img[y0:y1, x0:x1]
    win[sh] *= 0.72

    lean = np.array([0.80, 0.42, 0.42]) * rng.uniform(0.92, 1.06, 3)
    fat = np.array([0.95, 0.88, 0.83])
    tex = lean * (1 + 0.07 * _noise(rng, u.shape, 2.5))[..., None]
    fe = rng.uniform(0.45, 0.65)  # fat edge position (normalised)
    fat_a = np.clip((vn - fe) * 8, 0, 1)
    for _ in range(rng.integers(1, 3)):  # interior fat streaks
        pos = rng.uniform(-0.7, 0.3)
        wid = rng.uniform(0.05, 0.12)
        line = pos + 0.06 * np.sin(u / rng.uniform(10, 25))
        fat_a = np.maximum(fat_a, np.clip(1 - np.abs(vn - line) / wid, 0, 1) ** 0.7)
    tex = _blend(tex, fat, fat_a * 0.92)
    tex = _blend(tex, [0.83, 0.68, 0.52], np.clip((vn - 0.9) * 10, 0, 1) * 0.6)  # rind
    shade = 1 - 0.18 * np.clip(np.abs(vn), 0, 1) ** 4
    if twist is not None:  # fold is darker, with a thin highlight on the crease
        shade = shade * np.where(tmask_local, 0.62 + 0.38 * wf, 1.0)
        crease = np.abs(u - twist[0]) < 2.0
        tex = _blend(tex, [1, 0.9, 0.88], 0.35 * crease)
    tex = tex * shade[..., None]
    edge = gaussian_filter(mask.astype(float), 0.6)
    win[:] = win * (1 - edge[..., None]) + tex * edge[..., None]

    full = np.zeros((H, W), bool)
    full[y0:y1, x0:x1] = mask
    tm = np.zeros((H, W), bool)
    tm[y0:y1, x0:x1] = mask & tmask_local
    return full, tm


# ----------------------------------------------------------------------------
# scene
# ----------------------------------------------------------------------------
def render(rng: np.random.Generator, defects: list[int] | None = None) -> Scene:
    """Render one tray image.

    Args:
        rng: Random generator (controls everything -> reproducible).
        defects: Defect class ids to include (repeats allowed). ``None``
            samples them: 35 % of trays defect-free, otherwise 1-2 defects.

    Returns:
        A :class:`Scene`.
    """
    if defects is None:
        defects = [] if rng.random() < 0.35 else list(rng.integers(0, 5, rng.integers(1, 3)))
    if any(d not in range(len(config.SYNTH_CLASSES)) for d in defects):
        raise ValueError(f"invalid defect ids {defects}")
    yy, xx = np.mgrid[0:H, 0:W].astype(float)

    # belt
    img = np.ones((H, W, 3)) * (np.array([0.30, 0.31, 0.33]) * rng.uniform(0.8, 1.2))
    img *= (1 + 0.05 * np.sin(xx * 0.9) * 0.5 + 0.05 * _noise(rng, (H, W), 8))[..., None]

    # tray geometry
    tcx, tcy = W / 2 + rng.uniform(-15, 15), H / 2 + rng.uniform(-12, 12)
    tw, th = rng.uniform(560, 620), rng.uniform(400, 440)
    rim = rng.uniform(18, 24)
    dx, dy = np.abs(xx - tcx), np.abs(yy - tcy)
    tray = (dx < tw / 2) & (dy < th / 2)
    inner = (dx < tw / 2 - rim) & (dy < th / 2 - rim)
    black = rng.random() < 0.7
    tcol = np.array([0.10, 0.10, 0.11]) if black else np.array([0.80, 0.80, 0.78])
    img[tray] = tcol * rng.uniform(0.9, 1.1)
    img[tray & ~inner] = tcol + (0.10 if black else -0.06)  # flange
    # glossy seal band in the middle of the flange
    dist_edge = np.minimum(tw / 2 - dx, th / 2 - dy)  # px inside tray outline
    seal = tray & (np.abs(dist_edge - rim / 2) < rim * 0.22)

    boxes: list[list[float]] = []
    classes: list[int] = []

    # unsealed: choose a flange segment where the seal is missing
    unsealed_regions = []
    for _ in range(defects.count(3)):
        side = rng.integers(4)
        seg = rng.uniform(70, 150)
        if side in (0, 1):  # top / bottom edge
            c = rng.uniform(tcx - tw / 2 + seg / 2 + 30, tcx + tw / 2 - seg / 2 - 30)
            region = (np.abs(xx - c) < seg / 2) & tray & (dy > th / 2 - rim - rng.uniform(12, 28)) & (np.sign(yy - tcy) == (-1 if side == 0 else 1))
        else:
            c = rng.uniform(tcy - th / 2 + seg / 2 + 30, tcy + th / 2 - seg / 2 - 30)
            region = (np.abs(yy - c) < seg / 2) & tray & (dx > tw / 2 - rim - rng.uniform(12, 28)) & (np.sign(xx - tcx) == (-1 if side == 2 else 1))
        unsealed_regions.append(region)
        seal &= ~region
    img = _blend(img, [0.92, 0.92, 0.95], 0.55 * seal)

    # rashers, shingled along the tray
    n = int(rng.integers(4, 7))
    length = (tw - 2 * rim) * rng.uniform(0.78, 0.88)
    width = rng.uniform(62, 80)
    span = th - 2 * rim - width - 20
    base_ang = rng.uniform(-0.05, 0.05)
    twist_target = [i for i in range(n)]
    rng.shuffle(twist_target)
    n_twist = defects.count(2)
    for i in range(n):
        cy = tcy - span / 2 + span * i / (n - 1) + rng.uniform(-4, 4)
        cx = tcx + rng.uniform(-12, 12)
        tw_spec = None
        if i in twist_target[:n_twist]:
            tw_spec = (rng.uniform(-0.3, 0.3) * length, rng.uniform(22, 34))
        _, tmask = _draw_rasher(img, rng, cx, cy, length, width, base_ang + rng.normal(0, 0.03), tw_spec)
        if tw_spec is not None:  # box resolved after the scene is complete (occlusion)
            twist_mask = tmask
            classes.append(2)
            boxes.append(twist_mask)  # type: ignore[arg-type]  # resolved below

    # loose meat: a rasher / fragment at a clearly different angle, on top
    for _ in range(defects.count(0)):
        frag = rng.uniform(0.3, 0.75) * length
        ang = base_ang + rng.choice([-1, 1]) * rng.uniform(0.5, 1.4)
        lx = rng.uniform(tcx - tw / 2 + frag / 3, tcx + tw / 2 - frag / 3)
        ly = rng.uniform(tcy - th / 2 + 40, tcy + th / 2 - 40)
        lmask, _ = _draw_rasher(img, rng, lx, ly, frag, width * rng.uniform(0.8, 1.0), ang, flip=rng.random() < 0.5)
        lmask &= tray | (rng.random() < 0.5)  # may overhang the tray
        classes.append(0)
        boxes.append(lmask)  # type: ignore[arg-type]

    # resolve twist boxes now that loose meat may occlude them
    loose_union = np.zeros((H, W), bool)
    for b, c in zip(boxes, classes):
        if c == 0:
            loose_union |= b  # type: ignore[operator]
    resolved_b, resolved_c = [], []
    for b, c in zip(boxes, classes):
        m = b if c == 0 else (b & ~loose_union)  # type: ignore[operator]
        bb = _bbox(m, pad=3)
        if bb is not None:
            resolved_b.append(bb)
            resolved_c.append(c)
    boxes, classes = resolved_b, resolved_c

    # unsealed appearance: dark gap + bright lifted film edge
    for region in unsealed_regions:
        g = gaussian_filter(region.astype(float), 2)
        img *= (1 - 0.35 * g)[..., None]
        ys, xs = np.nonzero(region)
        horiz = (xs.max() - xs.min()) > (ys.max() - ys.min())
        if horiz:
            yline = ys.mean() + (np.sign(tcy - ys.mean()) * rng.uniform(4, 12))
            pts = np.stack([np.linspace(xs.min(), xs.max(), 12), yline + rng.normal(0, 3, 12)], 1)
        else:
            xline = xs.mean() + (np.sign(tcx - xs.mean()) * rng.uniform(4, 12))
            pts = np.stack([xline + rng.normal(0, 3, 12), np.linspace(ys.min(), ys.max(), 12)], 1)
        cov, (sy, sx) = _polyline_cov(pts, rng.uniform(2.5, 4.5))
        img[sy, sx] = _blend(img[sy, sx], [0.96, 0.96, 1.0], 0.85 * cov)
        bb = _bbox(region, pad=4)
        if bb is not None:
            boxes.append(bb)
            classes.append(3)

    # packaging error: torn lidding film (jagged opening, curled bright lips)
    for _ in range(defects.count(1)):
        start = (rng.uniform(tcx - tw / 2 + 60, tcx + tw / 2 - 60), rng.uniform(tcy - th / 2 + 50, tcy + th / 2 - 50))
        pts = _curve(rng, start, int(rng.integers(4, 8)), rng.uniform(10, 18), 0.9)
        wgap = rng.uniform(4, 8)
        gap, (sy, sx) = _polyline_cov(pts, wgap, pad=wgap + 10)
        lips, _ = _polyline_cov(pts, wgap + 5, pad=wgap + 10)
        lips = np.clip(lips - gap, 0, 1)
        img[sy, sx] *= (1 - rng.uniform(0.2, 0.4) * gap)[..., None]
        img[sy, sx] = _blend(img[sy, sx], [0.97, 0.97, 1.0], rng.uniform(0.6, 0.9) * lips)
        full = np.zeros((H, W), bool)
        full[sy, sx] = (gap + lips) > 0.2
        bb = _bbox(full, pad=4)
        if bb is not None:
            boxes.append(bb)
            classes.append(1)

    # film haze over the tray
    img = _blend(img, [1, 1, 1], 0.05 * tray)

    # reflections (all trays): broad soft light bars + soft blobs  -> confounder
    glare = float(rng.uniform(0, 1))
    for _ in range(rng.integers(1, 4)):
        a = rng.uniform(0, np.pi)
        px, py = rng.uniform(tcx - tw / 3, tcx + tw / 3), rng.uniform(tcy - th / 3, tcy + th / 3)
        along = (xx - px) * np.cos(a) + (yy - py) * np.sin(a)
        across = -(xx - px) * np.sin(a) + (yy - py) * np.cos(a)
        bar = np.exp(-(across**2) / (2 * rng.uniform(6, 16) ** 2) - along**2 / (2 * rng.uniform(40, 120) ** 2))
        img = _blend(img, [1, 1, 1], glare * rng.uniform(0.25, 0.6) * bar * tray)

    # wrinkles: sharp narrow crease + parallel shadow
    for _ in range(defects.count(4)):
        start = (rng.uniform(tcx - tw / 2 + 30, tcx + tw / 2 - 30), rng.uniform(tcy - th / 2 + 30, tcy + th / 2 - 30))
        pts = _curve(rng, start, int(rng.integers(4, 9)), rng.uniform(12, 22), 0.25)
        pts[:, 0] = np.clip(pts[:, 0], tcx - tw / 2 + 5, tcx + tw / 2 - 5)
        pts[:, 1] = np.clip(pts[:, 1], tcy - th / 2 + 5, tcy + th / 2 - 5)
        sw = rng.uniform(1.6, 3.0)
        cov_s, (sy, sx) = _polyline_cov(pts + np.array([2.0, 2.5]), sw + 1)
        img[sy, sx] *= (1 - 0.35 * cov_s)[..., None]
        cov, (sy, sx) = _polyline_cov(pts, sw)
        img[sy, sx] = _blend(img[sy, sx], [1, 1, 1], rng.uniform(0.55, 0.85) * cov)
        full = np.zeros((H, W), bool)
        full[sy, sx] = cov > 0.2
        bb = _bbox(full, pad=5)
        if bb is not None:
            boxes.append(bb)
            classes.append(4)

    # acquisition: exposure, saturation, colour temperature, defocus, noise
    img = img * rng.uniform(0.8, 1.2)
    grey = img.mean(2, keepdims=True)
    img = grey + (img - grey) * rng.uniform(0.75, 1.25)
    t = rng.uniform(-0.04, 0.04)
    img = img * np.array([1 + t, 1, 1 - t])
    sb = rng.uniform(0, 0.8)
    if sb > 0.15:
        img = gaussian_filter(img, (sb, sb, 0))
    img = img + rng.normal(0, rng.uniform(0.005, 0.02), img.shape)
    out = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
    return Scene(out, np.array(boxes, float).reshape(-1, 4), np.array(classes, int), glare)


def generate_dataset(
    out_dir: str | Path,
    n_train: int = 600,
    n_valid: int = 150,
    n_test: int = 150,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Write a YOLO-format synthetic dataset identical in layout to the real one.

    Args:
        out_dir: Destination directory.
        n_train, n_valid, n_test: Images per split.
        seed: Master seed; same seed -> identical dataset.

    Returns:
        Per-image metadata (split, file, number of boxes per class, glare),
        also written to ``meta.csv``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for split, n in (("train", n_train), ("valid", n_valid), ("test", n_test)):
        for i in range(n):
            sc = render(rng)
            name = f"{split}_{i:04d}"
            write_yolo_split(out, split, name, sc.image, sc.boxes, sc.classes)
            row = {"split": split, "file": f"{name}.jpg", "glare": round(sc.glare, 3)}
            row.update({c: int((sc.classes == k).sum()) for k, c in enumerate(config.SYNTH_CLASSES)})
            rows.append(row)
    (out / "data.yaml").write_text(
        yaml.safe_dump({"names": list(config.SYNTH_CLASSES), "nc": len(config.SYNTH_CLASSES),
                        "train": "train/images", "val": "valid/images", "test": "test/images"}, sort_keys=False)
    )
    df = pd.DataFrame(rows)
    df.to_csv(out / "meta.csv", index=False)
    return df
