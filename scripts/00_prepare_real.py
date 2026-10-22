"""Step 0 (real data only) - build the two dataset variants used in the experiments.

* ``<data>-reduced``: the predefined Roboflow splits, with the training split
  reduced to one of the three exposure-augmented copies of each source frame
  (cuts CPU training time by ~3x; the detector's own HSV augmentation covers
  the +/-10 % exposure range).
* ``<data>-block``: a leakage-aware re-split in which test and validation
  frames come from separate contiguous stretches of the video, with a guard
  band of frames removed from training (see rasherqc.dataset.make_block_split).

Also records how close (in video frames) each original valid/test image is to
a training frame: results/<tag>/tables/frame_gaps.json.

Usage: python scripts/00_prepare_real.py --data data/pork-rasher
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from _common import out_dirs, parse, plt, save_json

from rasherqc.dataset import class_statistics, frame_gap_to_train, load_dataset, make_block_split, make_reduced_split


def extra(ap):
    ap.add_argument("--block", type=int, default=300)
    ap.add_argument("--gap", type=int, default=60)


a = parse(__doc__.split("\n")[0], extra)
fig_dir, tab_dir, _ = out_dirs(a.tag)
ds = load_dataset(a.data)
gaps = frame_gap_to_train(ds)
summary = {s: {"median": float(np.nanmedian(g)), "frac_le_3": float(np.mean(g <= 3)), "frac_le_30": float(np.mean(g <= 30)),
               "frac_eq_0": float(np.mean(g == 0)), "max": float(np.nanmax(g))} for s, g in gaps.items()}
print("frame gap to nearest training frame:", summary)
fig, ax = plt.subplots(figsize=(5, 2.8))
for s, g in gaps.items():
    ax.hist(np.clip(g, 0, 40), bins=np.arange(-0.5, 41.5, 1), alpha=0.6, label=f"{s} (median {np.nanmedian(g):.0f})")
ax.set_xlabel("frames to nearest training frame (clipped at 40)")
ax.set_ylabel("images")
ax.set_title("Predefined split: evaluation frames vs training frames")
ax.legend(fontsize=7)
fig.savefig(fig_dir / "frame_gaps.png")

root = Path(a.data)
red = make_reduced_split(ds, root.parent / f"{root.name}-reduced")
blk, info = make_block_split(ds, root.parent / f"{root.name}-block", block=a.block, gap=a.gap)
out = {"frame_gaps": summary, "block_split": info,
       "reduced_stats": class_statistics(load_dataset(red)), "block_stats": class_statistics(load_dataset(blk))}
save_json(out, tab_dir / "dataset_variants.json")
for k in ("reduced_stats", "block_stats"):
    print(k)
    for s, v in out[k].items():
        print(f"  {s:6s} images={v['images']:5d} fail={v['fail_trays']:5d} boxes={v['boxes']}")
print("block split info:", info)
