"""Reading and validating single input images.

Top-level image contract
------------------------
* **File**: PNG/JPEG/BMP/TIFF readable by Pillow. EXIF orientation is applied
  (the provided dataset is already auto-oriented, so this is a no-op there).
* **Colour**: RGB, 8 bits per channel (0-255). RGBA is accepted (alpha
  discarded, recorded as a warning); palette/CMYK/16-bit images are
  converted. Greyscale is **rejected**: lean/fat colour is part of the
  evidence and the detector was trained on colour images.
* **Size**: each side within [:data:`~rasherqc.config.MIN_SIDE_PX`,
  :data:`~rasherqc.config.MAX_SIDE_PX`]. The dataset images are 720 x 540;
  other sizes are letter-boxed by the detector, which is supported but only
  validated for the 4:3 camera geometry of the dataset.
* **Exposure**: mean grey level within
  :data:`~rasherqc.config.MEAN_INTENSITY_RANGE` and at most
  :data:`~rasherqc.config.MAX_CLIPPED_FRACTION` of pixels clipped.
* **Scene**: one tray, viewed from above, roughly filling the frame.

Images outside this domain are not "repaired": :func:`validate_image`
returns ``ok=False`` with reasons and the inspection report marks the tray
``UNSUPPORTED_INPUT`` so that a person checks it. A tray is never passed on
the basis of an image the system cannot interpret.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from . import config


class InvalidInputError(ValueError):
    """Raised when an input cannot be processed at all (missing, unreadable)."""


@dataclass
class ValidationResult:
    """Outcome of :func:`validate_image`.

    Attributes:
        ok: ``True`` if the image is inside the supported input domain.
        reasons: Human-readable reasons for rejection (empty when ``ok``).
        warnings: Non-fatal issues that were corrected (e.g. alpha removed).
    """

    ok: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def read_image(path: str | Path) -> np.ndarray:
    """Read an image file into a uint8 array, applying EXIF orientation.

    Args:
        path: Path to the image file.

    Returns:
        ``uint8`` array of shape (H, W) for greyscale, or (H, W, C).

    Raises:
        InvalidInputError: If the file is missing or not a readable image.
    """
    path = Path(path)
    if not path.is_file():
        raise InvalidInputError(f"File not found: {path}")
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode not in ("L", "RGB", "RGBA"):
                im = im.convert("RGBA" if "transparency" in im.info else "RGB")
            arr = np.asarray(im)
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidInputError(f"Unreadable image {path}: {exc}") from exc
    return arr.astype(np.uint8, copy=False)


def validate_image(img: np.ndarray) -> tuple[np.ndarray | None, ValidationResult]:
    """Check an image against the supported input domain.

    Args:
        img: Array from :func:`read_image` (uint8, (H,W) or (H,W,C)).

    Returns:
        ``(rgb, result)``: ``rgb`` is an (H, W, 3) uint8 array when usable,
        otherwise ``None``; ``result`` lists the reasons and warnings.
    """
    warnings: list[str] = []
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return None, ValidationResult(False, ["empty or non-array input"])
    if img.dtype != np.uint8:
        return None, ValidationResult(False, [f"unsupported dtype {img.dtype}; expected uint8"])
    if img.ndim == 2:
        return None, ValidationResult(False, ["greyscale image; RGB colour is required"])
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        return None, ValidationResult(False, [f"unsupported shape {img.shape}; expected (H, W, 3)"])
    if img.shape[2] == 4:
        img = np.ascontiguousarray(img[..., :3])
        warnings.append("alpha channel discarded")

    reasons: list[str] = []
    h, w = img.shape[:2]
    if min(h, w) < config.MIN_SIDE_PX:
        reasons.append(f"image too small ({w}x{h}); minimum side {config.MIN_SIDE_PX}px")
    if max(h, w) > config.MAX_SIDE_PX:
        reasons.append(f"image too large ({w}x{h}); maximum side {config.MAX_SIDE_PX}px")
    if np.array_equal(img[..., 0], img[..., 1]) and np.array_equal(img[..., 1], img[..., 2]):
        reasons.append("channels identical (greyscale stored as RGB); colour is required")
    mean = float(img.mean())
    lo, hi = config.MEAN_INTENSITY_RANGE
    if not lo <= mean <= hi:
        reasons.append(f"exposure out of range (mean grey {mean:.1f}, allowed {lo}-{hi})")
    clipped = float(np.mean((img.max(axis=2) >= 255) | (img.min(axis=2) <= 0)))
    if clipped > config.MAX_CLIPPED_FRACTION:
        reasons.append(f"{clipped:.0%} of pixels clipped (limit {config.MAX_CLIPPED_FRACTION:.0%})")

    ok = not reasons
    return (img if ok else None), ValidationResult(ok, reasons, warnings)


def load_valid_image(path: str | Path) -> np.ndarray:
    """Read and validate one image, raising if it is outside the domain.

    Args:
        path: Image file path.

    Returns:
        (H, W, 3) uint8 RGB array.

    Raises:
        InvalidInputError: If the file cannot be read or fails validation.
    """
    rgb, res = validate_image(read_image(path))
    if not res.ok:
        raise InvalidInputError(f"{path}: " + "; ".join(res.reasons))
    return rgb


def resize_square(img: np.ndarray, size: int) -> np.ndarray:
    """Resize an RGB image to ``size`` x ``size`` (area averaging, no crop).

    Used by the classical baseline, whose features are defined on a fixed
    grid. Trays fill most of the 4:3 frame, so the mild anisotropic squeeze
    is acceptable and keeps the tray edges (where unsealed defects sit).

    Args:
        img: (H, W, 3) uint8 array.
        size: Output side length in pixels.

    Returns:
        (size, size, 3) uint8 array.
    """
    return np.asarray(Image.fromarray(img).resize((size, size), Image.BOX))
