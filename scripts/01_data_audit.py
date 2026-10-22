"""Step 1 - audit the dataset before any modelling.

Outputs (results/<tag>/):
* tables/data_stats.json      images, FAIL trays, boxes and median box size per split/class
* tables/duplicates.json      valid/test images that are near-duplicates of training images
* figures/class_counts.png    box counts per class and split
* figures/box_sizes.png       distribution of box area fraction per class
* figures/examples.png        one annotated example per class

Usage: python scripts/01_data_audit.py --data data/<dataset>
"""

from __future__ import annotations

import numpy as np
from _common import out_dirs, parse, plt, save_json
from PIL import Image, ImageDraw

from rasherqc.dataset import class_statistics, cross_split_duplicates, load_dataset

a = parse(__doc__.split("\n")[0])
fig_dir, tab_dir, _ = out_dirs(a.tag)
ds = load_dataset(a.data)
stats = class_statistics(ds)
dups = cross_split_duplicates(ds)
save_json(stats, tab_dir / "data_stats.json")
save_json([{"eval_image": str(e), "train_image": str(t), "hamming": d} for e, t, d in dups], tab_dir / "duplicates.json")
print("classes:", ds.names)
for s, v in stats.items():
    print(f"{s:6s} images={v['images']:5d} fail_trays={v['fail_trays']:5d} boxes={v['boxes']}")
print(f"near-duplicates of training images in valid/test: {len(dups)}")

# class counts
fig, ax = plt.subplots(figsize=(6, 3))
splits = list(stats)
x = np.arange(len(ds.names))
for k, s in enumerate(splits):
    ax.bar(x + (k - 1) * 0.27, [stats[s]["boxes"][n] for n in ds.names], 0.27, label=s)
ax.set_xticks(x, ds.names, rotation=15)
ax.set_ylabel("boxes")
ax.set_title("Annotated defects per class and split")
ax.legend()
fig.savefig(fig_dir / "class_counts.png")

# box sizes
fig, ax = plt.subplots(figsize=(6, 3))
data = [[(b[2] - b[0]) * (b[3] - b[1]) / (r.width * r.height) * 100 for r in ds.records for b, c in zip(r.boxes, r.classes) if c == k]
        for k in range(len(ds.names))]
ax.boxplot([d if d else [np.nan] for d in data], tick_labels=ds.names, showfliers=False)
ax.set_yscale("log")
ax.set_ylabel("box area (% of image)")
ax.set_title("Defect size distribution")
fig.savefig(fig_dir / "box_sizes.png")

# examples: one per class
cols = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180)]
tiles = []
for k in range(len(ds.names)):
    rec = next((r for r in ds.split("train") if k in r.classes), None)
    if rec is None:
        continue
    im = Image.open(rec.path).convert("RGB")
    d = ImageDraw.Draw(im)
    for b, c in zip(rec.boxes, rec.classes):
        d.rectangle(list(b), outline=cols[c % 5], width=4)
        d.text((b[0] + 4, b[1] + 3), ds.names[c], fill=cols[c % 5])
    tiles.append((ds.names[k], im))
fig, axs = plt.subplots(1, len(tiles), figsize=(3.2 * len(tiles), 2.6))
for ax, (n, im) in zip(np.atleast_1d(axs), tiles):
    ax.imshow(im)
    ax.set_title(n)
    ax.axis("off")
fig.savefig(fig_dir / "examples.png")
print("saved figures to", fig_dir)
