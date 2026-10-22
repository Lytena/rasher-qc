"""Dataset loading (YOLO or COCO export), tray labels and leakage checks.

Supported dataset layouts
-------------------------
Both standard exports of the provided dataset are accepted::

    <root>/data.yaml                      # YOLO export: class names
    <root>/{train,valid,test}/images/*.jpg
    <root>/{train,valid,test}/labels/*.txt   # "cls cx cy w h", normalised 0-1

    <root>/{train,valid,test}/_annotations.coco.json   # COCO export
    <root>/{train,valid,test}/*.jpg

``val`` is accepted as an alias of ``valid``. A COCO export is converted once
to YOLO label files under ``<root>/_yolo/`` so a single detector code path is
used. Classes with no annotations at all (e.g. the dummy super-category some
exporters add) are dropped.

Annotation units: boxes are handled internally as ``(x1, y1, x2, y2)`` in
**pixels** of the original image, origin top-left.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from . import config
from .io import InvalidInputError

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SPLITS = ("train", "valid", "test")


@dataclass
class Record:
    """One annotated image.

    Attributes:
        path: Image file path.
        split: ``"train"``, ``"valid"`` or ``"test"``.
        width: Image width (px).
        height: Image height (px).
        boxes: (N, 4) float array of ``x1, y1, x2, y2`` in pixels.
        classes: (N,) int array of class indices.
    """

    path: Path
    split: str
    width: int
    height: int
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    classes: np.ndarray = field(default_factory=lambda: np.zeros(0, int))


@dataclass
class Dataset:
    """A loaded dataset.

    Attributes:
        root: Dataset directory.
        names: Class names, index = class id.
        records: All annotated images (every split).
        yolo_yaml: Path of a normalised ``data.yaml`` (absolute paths) that
            the detector trains from.
    """

    root: Path
    names: list[str]
    records: list[Record]
    yolo_yaml: Path

    def split(self, name: str) -> list[Record]:
        """Records of one split (``train``, ``valid`` or ``test``)."""
        return [r for r in self.records if r.split == name]

    @property
    def defect_ids(self) -> list[int]:
        """Class ids that count as defects (see :func:`is_defect_name`)."""
        return [i for i, n in enumerate(self.names) if is_defect_name(n)]


def is_defect_name(name: str) -> bool:
    """True unless ``name`` is listed in :data:`~rasherqc.config.NON_DEFECT_NAMES`."""
    norm = name.lower().replace(" ", "").replace("-", "").replace("_", "")
    return norm not in {n.replace("_", "") for n in config.NON_DEFECT_NAMES}


def tray_fail_label(rec: Record, defect_ids: list[int]) -> int:
    """Ground-truth tray decision: 1 (FAIL) if any defect box is annotated."""
    return int(np.isin(rec.classes, defect_ids).any())


def _split_dir(root: Path, split: str) -> Path | None:
    for cand in ((split,) if split != "valid" else ("valid", "val")):
        if (root / cand).is_dir():
            return root / cand
    return None


def _images_in(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


def _read_yolo_labels(txt: Path, w: int, h: int, nc: int) -> tuple[np.ndarray, np.ndarray]:
    """Parse a YOLO label file into pixel ``xyxy`` boxes and class ids."""
    if not txt.is_file():
        return np.zeros((0, 4)), np.zeros(0, int)  # no file = no objects
    rows = [ln.split() for ln in txt.read_text().splitlines() if ln.strip()]
    boxes, cls = [], []
    for r in rows:
        if len(r) != 5:  # polygon/segment rows are reduced to their bounding box
            vals = np.array(r[1:], float).reshape(-1, 2)
            x1, y1 = vals.min(0)
            x2, y2 = vals.max(0)
            cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        else:
            cx, cy, bw, bh = map(float, r[1:])
        c = int(float(r[0]))
        if not 0 <= c < nc:
            raise InvalidInputError(f"{txt}: class id {c} outside 0..{nc - 1}")
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        cls.append(c)
    b = np.clip(np.array(boxes, float).reshape(-1, 4), 0, [w, h, w, h])
    return b, np.array(cls, int)


def _load_yolo(root: Path) -> tuple[list[str], list[Record], dict[str, Path]]:
    meta = yaml.safe_load((root / "data.yaml").read_text())
    names = meta["names"]
    if isinstance(names, dict):  # {0: name, 1: name}
        names = [names[k] for k in sorted(names)]
    records, img_dirs = [], {}
    for split in SPLITS:
        d = _split_dir(root, split)
        if d is None:
            continue
        img_dir = d / "images" if (d / "images").is_dir() else d
        img_dirs[split] = img_dir
        for p in _images_in(img_dir):
            with Image.open(p) as im:
                w, h = im.size
            b, c = _read_yolo_labels(d / "labels" / (p.stem + ".txt"), w, h, len(names))
            records.append(Record(p, split, w, h, b, c))
    return list(names), records, img_dirs


def _load_coco(root: Path) -> tuple[list[str], list[Record], dict[str, Path]]:
    per_split = {}
    for split in SPLITS:
        d = _split_dir(root, split)
        if d is not None and (d / "_annotations.coco.json").is_file():
            per_split[split] = (d, json.loads((d / "_annotations.coco.json").read_text()))
    if not per_split:
        raise InvalidInputError(f"no data.yaml or _annotations.coco.json found under {root}")
    used = sorted({a["category_id"] for _, js in per_split.values() for a in js["annotations"]})
    cats = {c["id"]: c["name"] for _, js in per_split.values() for c in js["categories"]}
    remap = {cid: i for i, cid in enumerate(used)}
    names = [cats[c] for c in used]
    records, img_dirs = [], {}
    out_root = root / "_yolo"
    for split, (d, js) in per_split.items():
        anns: dict[int, list] = {}
        for a in js["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        img_out, lab_out = out_root / split / "images", out_root / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lab_out.mkdir(parents=True, exist_ok=True)
        img_dirs[split] = img_out
        for im in js["images"]:
            src = d / im["file_name"]
            w, h = im["width"], im["height"]
            bx, cl, lines = [], [], []
            for a in anns.get(im["id"], []):
                x, y, bw, bh = a["bbox"]  # COCO: top-left x, y, width, height (px)
                c = remap[a["category_id"]]
                bx.append([x, y, x + bw, y + bh])
                cl.append(c)
                lines.append(f"{c} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
            dst = img_out / Path(im["file_name"]).name
            if not dst.exists():
                try:
                    dst.symlink_to(src.resolve())
                except OSError:
                    shutil.copy(src, dst)
            (lab_out / (dst.stem + ".txt")).write_text("\n".join(lines))
            records.append(Record(dst, split, w, h, np.array(bx, float).reshape(-1, 4), np.array(cl, int)))
    return names, records, img_dirs


def load_dataset(root: str | Path) -> Dataset:
    """Load a YOLO- or COCO-format detection dataset.

    Args:
        root: Dataset directory (see module docstring for the layout).

    Returns:
        A :class:`Dataset`; also writes ``<root>/rasherqc_data.yaml`` with
        absolute paths for the detector.

    Raises:
        InvalidInputError: If the layout is not recognised or labels are
            inconsistent with the class list.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise InvalidInputError(f"dataset directory not found: {root}")
    names, records, img_dirs = _load_yolo(root) if (root / "data.yaml").is_file() else _load_coco(root)
    if "train" not in img_dirs or "valid" not in img_dirs:
        raise InvalidInputError(f"dataset needs at least train and valid splits; found {sorted(img_dirs)}")
    y = {"path": str(root), "names": {i: n for i, n in enumerate(names)}}
    y.update({"train": str(img_dirs["train"]), "val": str(img_dirs["valid"])})
    if "test" in img_dirs:
        y["test"] = str(img_dirs["test"])
    yaml_path = root / "rasherqc_data.yaml"
    yaml_path.write_text(yaml.safe_dump(y, sort_keys=False))
    return Dataset(root, names, records, yaml_path)


def write_yolo_split(out_dir: Path, split: str, name: str, img: np.ndarray, boxes: np.ndarray, classes: np.ndarray) -> None:
    """Write one image and its YOLO label file into ``out_dir/split/``.

    Args:
        out_dir: Dataset root.
        split: Split name.
        name: File stem.
        img: (H, W, 3) uint8 image.
        boxes: (N, 4) pixel ``xyxy`` boxes.
        classes: (N,) class ids.
    """
    h, w = img.shape[:2]
    (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_dir / split / "images" / f"{name}.jpg", quality=92)
    lines = [
        f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
        for (x1, y1, x2, y2), c in zip(boxes, classes)
    ]
    (out_dir / split / "labels" / f"{name}.txt").write_text("\n".join(lines))


# ----------------------------------------------------------------------------
# Video-frame structure of the provided dataset and leakage-aware splits
# ----------------------------------------------------------------------------
_FRAME_RE = re.compile(r"^(?P<prefix>.+?)_(?P<num>\d+)_(?:jpg|jpeg|png)\.rf\.[0-9a-f]+\.(?:jpg|jpeg|png)$", re.I)


def frame_number(path: str | Path) -> int | None:
    """Video frame number encoded in a Roboflow file name, else ``None``.

    Roboflow renames ``frame_with_red_10049.jpg`` to
    ``frame_with_red_10049_jpg.rf.<hash>.jpg``; augmented copies of the same
    source frame share the number and differ in the hash.
    """
    m = _FRAME_RE.match(Path(path).name)
    return int(m.group("num")) if m else None


def frame_gap_to_train(ds: Dataset) -> dict[str, np.ndarray]:
    """For each valid/test image, distance (in frames) to the nearest training frame.

    In a video, a small gap means the same physical tray seen moments later,
    i.e. the evaluation image is not independent of the training data.
    """
    tr = np.array(sorted({frame_number(r.path) for r in ds.split("train")} - {None}))
    out = {}
    for s in ("valid", "test"):
        f = np.array([frame_number(r.path) for r in ds.split(s)], dtype=float)
        out[s] = np.array([np.min(np.abs(tr - x)) if np.isfinite(x) and len(tr) else np.nan for x in f])
    return out


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy(src, dst)


def _write_subset(ds: Dataset, out: Path, assign: dict[str, list[Record]]) -> Path:
    """Materialise a dataset variant as symlinks + YOLO label files."""
    if out.exists():
        shutil.rmtree(out)
    for split, recs in assign.items():
        for r in recs:
            _link(r.path, out / split / "images" / r.path.name)
            lab = r.path.parent.parent / "labels" / (r.path.stem + ".txt")
            if lab.is_file():
                _link(lab, out / split / "labels" / lab.name)
            else:
                (out / split / "labels").mkdir(parents=True, exist_ok=True)
                (out / split / "labels" / (r.path.stem + ".txt")).write_text("")
    (out / "data.yaml").write_text(yaml.safe_dump({"names": ds.names, "nc": len(ds.names), "train": "train/images",
                                                   "val": "valid/images", "test": "test/images"}, sort_keys=False))
    return out


def one_copy_per_frame(records: list[Record]) -> list[Record]:
    """Keep one (deterministically chosen) augmented copy of every source frame."""
    seen: dict[int | str, Record] = {}
    for r in sorted(records, key=lambda r: r.path.name):
        key = frame_number(r.path)
        seen.setdefault(key if key is not None else r.path.name, r)
    return list(seen.values())


def make_reduced_split(ds: Dataset, out: str | Path) -> Path:
    """Predefined splits, but training reduced to one copy per source frame.

    The Version 4 export holds three exposure-augmented copies (+/-10 %) of
    every training frame. The detector applies its own, wider, on-the-fly
    HSV augmentation, so the copies mainly triple the cost of an epoch.
    """
    return _write_subset(ds, Path(out), {"train": one_copy_per_frame(ds.split("train")),
                                         "valid": ds.split("valid"), "test": ds.split("test")})


def make_block_split(
    ds: Dataset, out: str | Path, block: int = 300, gap: int = 60, frac: tuple[float, float] = (0.12, 0.12), seed: int = config.SEED
) -> tuple[Path, dict]:
    """Leakage-aware re-split: contiguous blocks of video frames per split.

    All source frames (from every original split, one copy each) are cut into
    blocks of ``block`` consecutive frame numbers. Whole blocks are assigned
    at random to test/valid/train (target fractions ``frac`` for test and
    valid). Training frames closer than ``gap`` frames to any test/valid frame
    are discarded so that no tray appears on both sides of the split.

    Args:
        ds: Loaded original dataset (frame numbers must be parseable).
        out: Output directory for the new variant.
        block: Block length in frames.
        gap: Guard band in frames.
        frac: Target fractions of frames for (test, valid).
        seed: Random seed for the block assignment.

    Returns:
        ``(path, info)`` where ``info`` reports frames per split and dropped frames.
    """
    recs = one_copy_per_frame(ds.records)
    recs = [r for r in recs if frame_number(r.path) is not None]
    fn = np.array([frame_number(r.path) for r in recs])
    blocks = fn // block
    ub = np.unique(blocks)
    rng = np.random.default_rng(seed)
    order = rng.permutation(ub)
    n_tot = len(recs)
    split_of_block: dict[int, str] = {}
    counts = {"test": 0, "valid": 0}
    for b in order:  # fill test, then valid, rest train
        n_b = int(np.sum(blocks == b))
        for s, f in (("test", frac[0]), ("valid", frac[1])):
            if counts[s] + n_b <= f * n_tot * 1.15 and counts[s] < f * n_tot:
                split_of_block[int(b)] = s
                counts[s] += n_b
                break
        else:
            split_of_block[int(b)] = "train"
    sp = np.array([split_of_block[int(b)] for b in blocks])
    held = fn[sp != "train"]
    keep = np.ones(len(recs), bool)
    for i in np.where(sp == "train")[0]:
        if len(held) and np.min(np.abs(held - fn[i])) < gap:
            keep[i] = False
    assign = {s: [r for r, ss, k in zip(recs, sp, keep) if ss == s and k] for s in ("train", "valid", "test")}
    info = {s: len(v) for s, v in assign.items()}
    info.update({"dropped_train_in_gap": int((~keep).sum()), "block": block, "gap": gap, "seed": seed})
    return _write_subset(ds, Path(out), assign), info


# ----------------------------------------------------------------------------
# Leakage check: near-duplicate images across splits
# ----------------------------------------------------------------------------
def dhash(path: str | Path, size: int = 8) -> int:
    """64-bit difference hash of an image (robust to resize/re-encode/exposure).

    The image is reduced to (size) x (size+1) grey pixels and each bit records
    whether a pixel is brighter than its right neighbour.
    """
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def cross_split_duplicates(ds: Dataset, max_hamming: int = 4) -> list[tuple[Path, Path, int]]:
    """Find test/valid images that are near-duplicates of training images.

    Near-duplicates across splits (the same tray photographed twice, or an
    augmented copy of a test image placed in training) inflate test scores.

    Args:
        ds: Loaded dataset.
        max_hamming: Maximum Hamming distance between dHashes to flag.

    Returns:
        List of ``(eval_image, train_image, distance)`` tuples.
    """
    tr = [(r.path, dhash(r.path)) for r in ds.split("train")]
    th = np.array([h for _, h in tr], dtype=np.uint64)
    out = []
    for r in ds.split("valid") + ds.split("test"):
        h = np.uint64(dhash(r.path))
        x = np.bitwise_xor(th, h)
        # popcount of 64-bit ints via byte view
        d = np.unpackbits(x.view(np.uint8)).reshape(len(th), 64).sum(1)
        j = int(np.argmin(d)) if len(d) else -1
        if j >= 0 and d[j] <= max_hamming:
            out.append((r.path, tr[j][0], int(d[j])))
    return out


def class_statistics(ds: Dataset) -> dict:
    """Counts of images, boxes and tray decisions per split and class.

    Returns:
        Nested dict ``{split: {"images": n, "fail_trays": n, "boxes": {name: n},
        "box_area_frac_median": {name: x}}}``.
    """
    stats = {}
    for split in SPLITS:
        recs = ds.split(split)
        if not recs:
            continue
        boxes = {n: 0 for n in ds.names}
        areas: dict[str, list] = {n: [] for n in ds.names}
        for r in recs:
            for b, c in zip(r.boxes, r.classes):
                boxes[ds.names[c]] += 1
                areas[ds.names[c]].append((b[2] - b[0]) * (b[3] - b[1]) / (r.width * r.height))
        stats[split] = {
            "images": len(recs),
            "fail_trays": int(sum(tray_fail_label(r, ds.defect_ids) for r in recs)),
            "boxes": boxes,
            "box_area_frac_median": {n: float(np.median(a)) if a else None for n, a in areas.items()},
        }
    return stats
