"""Controlled image perturbations for sensitivity / robustness testing.

Each perturbation has one *severity* parameter so that performance can be
plotted against it. They model nuisance factors of a production line:

* ``exposure``: global gain (lighting drift, auto-exposure error);
* ``saturation``: colour-saturation scale (camera white balance, LED ageing);
* ``glare``: added soft specular light bars over the tray (reflections on
  the lidding film -- the brief asks about reflection errors);
* ``blur``: Gaussian defocus / motion (sigma in px at 720 x 540);
* ``noise``: additive Gaussian sensor noise (std in grey levels);
* ``jpeg``: JPEG compression (quality 100 -> 10).

All functions take and return (H, W, 3) uint8 RGB arrays and are
deterministic given ``seed``.
"""

from __future__ import annotations

import io as _io

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def exposure(img: np.ndarray, gain: float) -> np.ndarray:
    """Multiply intensities by ``gain`` (1 = unchanged), with clipping."""
    return np.clip(img.astype(float) * gain, 0, 255).astype(np.uint8)


def saturation(img: np.ndarray, scale: float) -> np.ndarray:
    """Scale colour saturation about the per-pixel grey value (1 = unchanged)."""
    f = img.astype(float)
    g = f.mean(2, keepdims=True)
    return np.clip(g + (f - g) * scale, 0, 255).astype(np.uint8)


def glare(img: np.ndarray, strength: float, seed: int = 0, n_bars: int = 2) -> np.ndarray:
    """Add ``n_bars`` soft elongated specular reflections of given strength (0-1)."""
    if strength <= 0:
        return img.copy()
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    f = img.astype(float) / 255
    for _ in range(n_bars):
        a = rng.uniform(0, np.pi)
        px, py = rng.uniform(0.25, 0.75) * w, rng.uniform(0.25, 0.75) * h
        # Rotated anisotropic Gaussian: long along the bar, thin across it,
        # like a reflection of a strip light on the lidding film.
        along = (xx - px) * np.cos(a) + (yy - py) * np.sin(a)
        across = -(xx - px) * np.sin(a) + (yy - py) * np.cos(a)
        bar = np.exp(-(across**2) / (2 * (0.02 * w) ** 2) - along**2 / (2 * (0.15 * w) ** 2))
        # Screen-style blend: brightens towards white without overshooting.
        f = f + (1 - f) * (strength * bar)[..., None]
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with standard deviation ``sigma`` px."""
    if sigma <= 0:
        return img.copy()
    return np.clip(gaussian_filter(img.astype(float), (sigma, sigma, 0)), 0, 255).astype(np.uint8)


def noise(img: np.ndarray, std: float, seed: int = 0) -> np.ndarray:
    """Add zero-mean Gaussian noise of standard deviation ``std`` grey levels."""
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(float) + rng.normal(0, std, img.shape), 0, 255).astype(np.uint8)


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    """Round-trip through JPEG at the given quality (1-100)."""
    buf = _io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


#: Perturbation sweeps used by ``scripts/04_robustness.py``: name -> (function, levels).
SWEEPS: dict[str, tuple] = {
    "exposure": (exposure, [0.5, 0.65, 0.8, 1.0, 1.2, 1.4, 1.6]),
    "saturation": (saturation, [0.0, 0.4, 0.7, 1.0, 1.3, 1.6]),
    "glare": (glare, [0.0, 0.2, 0.4, 0.6, 0.8]),
    "blur": (blur, [0.0, 1.0, 2.0, 3.0, 5.0]),
    "noise": (noise, [0.0, 5.0, 10.0, 20.0, 35.0]),
    "jpeg": (jpeg, [95, 60, 30, 15, 8]),
}
