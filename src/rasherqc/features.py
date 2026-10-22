"""Hand-crafted image features for the classical (non-deep) baseline.

The baseline answers the tray-level questions *"should this tray FAIL?"* and
*"which defect types are present?"* from a fixed-length feature vector fed to
a random forest. It cannot localise defects; that is the price paid for a
model that is cheap, transparent and needs little data.

Feature groups and the defect signature each targets
----------------------------------------------------
* **Meat layout** (redness segmentation, CIE a* > threshold): meat coverage,
  number of meat blobs, meat outside the central tray area (loose meat
  overhanging), and the orientation statistics of meat edges -- in a correct
  shingle all rashers are parallel, so a loose rasher adds a second dominant
  orientation (lower coherence, broader orientation histogram).
* **Meat colour** (Lab percentiles): twisted rashers expose a dark fold and
  swap the fat edge; dark-red fraction and L* spread respond to that.
* **Thin bright / dark structures** (white and black morphological top-hats):
  wrinkles and torn film lips are thin bright lines; unsealed flanges show a
  dark gap next to a bright lifted edge.
* **Spatial pyramid** (3 x 3 grid): line and fold responses are pooled per
  cell and summarised by max and mean, so a small local defect is not
  averaged away over the whole tray.
* **Texture** (LBP histogram, GLCM statistics) and **gradient orientation**.

Two implementations exist so that optimisation can be measured honestly:
``mode="reference"`` is the first version written (each group recomputes its
own colour conversions, 256-level GLCM, larger top-hat footprint) and
``mode="fast"`` is the profiled, optimised version (shared conversions,
32-level GLCM, smaller footprint, top-hats computed once and reused by the
pyramid). Tests check both produce closely matching features.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2gray, rgb2lab
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.morphology import black_tophat, disk, white_tophat

#: Working resolution of the baseline (width, height); keeps the 4:3 aspect.
FEAT_WH: tuple[int, int] = (320, 240)
_PCTS = (5, 25, 50, 75, 95)
_GLCM = ("contrast", "homogeneity", "energy", "correlation")
_GRID = 3


def meat_mask(lab: np.ndarray) -> np.ndarray:
    """Segment meat (lean and attached fat) by redness.

    Lean pork is clearly red (a* > 18); fat streaks between lean are pale and
    are recovered by a small closing and hole filling.

    Args:
        lab: (H, W, 3) CIE-Lab image.

    Returns:
        (H, W) bool mask.
    """
    m = lab[..., 1] > 18.0
    m = ndi.binary_closing(m, structure=np.ones((5, 5)), iterations=1)
    m = ndi.binary_fill_holes(m)
    return ndi.binary_opening(m, iterations=1)


def _orientation_stats(grey: np.ndarray, region: np.ndarray, sigma: float):
    """Orientation histogram (8 bins), coherence and dominant-angle spread."""
    gx, gy = ndi.sobel(grey, 1), ndi.sobel(grey, 0)
    mag = np.hypot(gx, gy)
    ori = np.mod(np.arctan2(gy, gx), np.pi)
    w = mag[region]
    h, _ = np.histogram(ori[region], bins=8, range=(0, np.pi), weights=w)
    h = h / max(h.sum(), 1e-9)
    jxx, jyy, jxy = (ndi.gaussian_filter(v, sigma) for v in (gx * gx, gy * gy, gx * gy))
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / (jxx + jyy + 1e-6)
    # circular spread of doubled angles (orientation is axial, period pi)
    z = np.sum(w * np.exp(2j * ori[region])) / max(w.sum(), 1e-9)
    return h, float(np.average(coh[region], weights=w + 1e-9)) if region.any() else 0.0, float(1 - abs(z))


def _grid_pool(x: np.ndarray, stat) -> np.ndarray:
    """Apply ``stat`` to each cell of a GRID x GRID partition."""
    H, W = x.shape
    out = []
    for i in range(_GRID):
        for j in range(_GRID):
            out.append(stat(x[i * H // _GRID : (i + 1) * H // _GRID, j * W // _GRID : (j + 1) * W // _GRID]))
    return np.array(out)


def feature_names() -> list[str]:
    """Names of the features returned by :func:`extract_features`, in order."""
    n = ["meat_frac", "meat_blobs", "meat_border_frac", "meat_bbox_fill"]
    n += [f"meat_ori_h{i}" for i in range(8)] + ["meat_coherence", "meat_ori_spread"]
    n += [f"lab_{c}_p{p}" for c in "Lab" for p in _PCTS] + ["dark_red_frac", "pale_frac"]
    n += ["wth_meat_p99", "wth_bg_p99", "wth_bg_mean", "bth_meat_p99", "bth_bg_p99", "bth_meat_mean"]
    n += ["line_frac_meat", "line_frac_bg", "line_blobs", "line_maxlen"]
    n += ["grid_wth_max", "grid_wth_mean", "grid_wth_std", "grid_bth_max", "grid_bth_mean", "grid_bth_std"]
    n += ["grid_meat_frac_min", "grid_meat_frac_std", "grid_coh_min", "grid_coh_std"]
    n += [f"lbp_h{i}" for i in range(10)]
    n += [f"glcm_{p}" for p in _GLCM]
    n += [f"bg_ori_h{i}" for i in range(8)] + ["bg_coherence", "bg_ori_spread"]
    n += ["spec_frac", "spec_blobs", "spec_maxsize", "border_dark_frac", "border_bright_line_frac"]
    return n


def extract_features(rgb: np.ndarray, mode: str = "fast") -> np.ndarray:
    """Compute the feature vector of one tray image.

    Args:
        rgb: (H, W, 3) uint8 image at :data:`FEAT_WH` resolution.
        mode: ``"fast"`` (optimised, default) or ``"reference"`` (original).

    Returns:
        1-D float64 array of length ``len(feature_names())``.

    Raises:
        ValueError: Unknown ``mode`` or not an (H, W, 3) image.
    """
    if mode not in ("fast", "reference"):
        raise ValueError(f"unknown mode {mode!r}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) image, got {rgb.shape}")
    fast = mode == "fast"
    H, W = rgb.shape[:2]
    k = W / FEAT_WH[0]  # scale factor for footprints
    lab = rgb2lab(rgb)
    grey = rgb2gray(rgb)
    meat = meat_mask(lab)
    bg = ~meat
    f: list[float] = []

    # --- meat layout --------------------------------------------------------
    lbl, nb = ndi.label(meat)
    big = int(np.sum(ndi.sum(meat, lbl, range(1, nb + 1)) > 0.002 * H * W)) if nb else 0
    yy, xx = np.mgrid[0:H, 0:W]
    central = (np.abs(xx - W / 2) < 0.40 * W) & (np.abs(yy - H / 2) < 0.38 * H)
    f += [float(meat.mean()), float(big), float(meat[~central].mean())]
    if meat.any():
        ys, xs = np.nonzero(meat)
        f.append(float(meat.sum() / ((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))))
    else:
        f.append(0.0)
    edge_band = meat ^ ndi.binary_erosion(meat, iterations=2)
    h, coh, spread = _orientation_stats(grey if fast else rgb2gray(rgb), edge_band, 3 * k)
    f += list(h) + [coh, spread]

    # --- meat colour --------------------------------------------------------
    labm = (lab if fast else rgb2lab(rgb))[meat]
    if len(labm) < 10:
        labm = lab.reshape(-1, 3)
    for c in range(3):
        f += list(np.percentile(labm[:, c], _PCTS))
    f += [float(np.mean((labm[:, 0] < 35) & (labm[:, 1] > 15))), float(np.mean(labm[:, 0] > 75))]

    # --- thin structures ------------------------------------------------------
    r = max(1, round((3 if fast else 5) * k))
    g = grey if fast else rgb2gray(rgb)
    wth = white_tophat(g, disk(r))
    bth = black_tophat(g, disk(r))
    q = lambda a, m, p: float(np.percentile(a[m], p)) if m.any() else 0.0  # noqa: E731
    f += [q(wth, meat, 99), q(wth, bg, 99), float(wth[bg].mean()) if bg.any() else 0.0,
          q(bth, meat, 99), q(bth, bg, 99), float(bth[meat].mean()) if meat.any() else 0.0]
    lines = wth > 0.12
    f += [float(lines[meat].mean()) if meat.any() else 0.0, float(lines[bg].mean()) if bg.any() else 0.0]
    ll, nl = ndi.label(lines)
    if nl:
        sl = ndi.find_objects(ll)
        lens = [max(s[0].stop - s[0].start, s[1].stop - s[1].start) for s in sl]
        f += [nl / (H * W) * 1e3, max(lens) / W]
    else:
        f += [0.0, 0.0]

    # --- spatial pyramid ----------------------------------------------------------
    if not fast:  # reference version recomputed the top-hats per cell
        wth_cells = np.array([np.percentile(white_tophat(c, disk(r)), 99)
                              for c in _grid_pool_cells(g)])
        bth_cells = np.array([np.percentile(black_tophat(c, disk(r)), 99)
                              for c in _grid_pool_cells(g)])
    else:
        wth_cells = _grid_pool(wth, lambda c: np.percentile(c, 99))
        bth_cells = _grid_pool(bth, lambda c: np.percentile(c, 99))
    f += [wth_cells.max(), wth_cells.mean(), wth_cells.std(), bth_cells.max(), bth_cells.mean(), bth_cells.std()]
    mf = _grid_pool(meat.astype(float), np.mean)
    gx, gy = ndi.sobel(g, 1), ndi.sobel(g, 0)
    jxx, jyy, jxy = (ndi.gaussian_filter(v, 3 * k) for v in (gx * gx, gy * gy, gx * gy))
    cohmap = np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / (jxx + jyy + 1e-6)
    cc = _grid_pool(np.where(meat, cohmap, np.nan), lambda c: np.nanmean(c) if np.isfinite(c).any() else 1.0)
    f += [mf.min(), mf.std(), float(cc.min()), float(cc.std())]

    # --- texture ------------------------------------------------------------------
    g8 = (g * 255).astype(np.uint8)
    lbp = local_binary_pattern(g8, 8, 1, method="uniform")
    hl, _ = np.histogram(lbp[meat] if meat.any() else lbp, bins=10, range=(0, 10))
    f += list(hl / max(hl.sum(), 1))
    levels = 32 if fast else 256
    qg = (g8.astype(np.uint16) * levels // 256).astype(np.uint8)
    gl = graycomatrix(qg, [1], [0, np.pi / 2], levels=levels, symmetric=True, normed=True)
    f += [float(graycoprops(gl, p).mean()) for p in _GLCM]

    # --- background (film/tray) orientation and specular / flange cues ---------------
    h2, coh2, spread2 = _orientation_stats(g, bg & (wth > 0.05), 3 * k)
    f += list(h2) + [coh2, spread2]
    spec = (lab[..., 0] > 85) & (np.hypot(lab[..., 1], lab[..., 2]) < 12)
    ls, ns = ndi.label(spec)
    f += [float(spec.mean()), ns / (H * W) * 1e3, float(np.bincount(ls.ravel())[1:].max()) / (H * W) if ns else 0.0]
    border = ~central & bg
    f += [float(np.mean(bth[border] > 0.10)) if border.any() else 0.0,
          float(np.mean(lines[border])) if border.any() else 0.0]
    return np.nan_to_num(np.asarray(f, dtype=np.float64))


def _grid_pool_cells(x: np.ndarray):
    H, W = x.shape
    for i in range(_GRID):
        for j in range(_GRID):
            yield x[i * H // _GRID : (i + 1) * H // _GRID, j * W // _GRID : (j + 1) * W // _GRID]


def _extract_one(args):
    img, mode = args
    return extract_features(img, mode)


def extract_batch(images: list[np.ndarray], mode: str = "fast", n_jobs: int = 1) -> np.ndarray:
    """Extract features for many images, optionally in parallel processes.

    Args:
        images: (H, W, 3) uint8 images at :data:`FEAT_WH`.
        mode: Feature implementation, see :func:`extract_features`.
        n_jobs: Worker processes (1 = serial). Images are independent, so the
            work is embarrassingly parallel.

    Returns:
        (N, n_features) float64 array.
    """
    # Process pools need the "fork" start method when called from plain scripts;
    # on Windows/macOS ("spawn") the scripts would be re-imported, so run serially.
    if n_jobs <= 1 or multiprocessing.get_start_method(allow_none=False) != "fork":
        return np.stack([extract_features(x, mode) for x in images])
    with ProcessPoolExecutor(n_jobs) as ex:
        return np.stack(list(ex.map(_extract_one, ((x, mode) for x in images), chunksize=8)))
