# Data

Image data is **not** committed to this repository (size, and the dataset is distributed by its owner under its own licence).

## 1. Provided dataset (real images)

**Pork Rasher Error (Packaging)** — Roboflow Universe, workspace `hello-2aqe0`, project `pork-rasher-error-packaging`, Version 4.
Licence: **CC BY 4.0** (attribution required; see the main README).
<https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging>

* 1,955 original images; Version 4 = 3,549 images after augmentation of the training split (saturation and exposure)
* auto-oriented, resized to 720 × 540 px; predefined train / valid / test splits
* bounding boxes, 5 classes: `loose-meat`, `packaging-error`, `twisted-meat`, `unsealed`, `wrinkle`

To obtain it: open the link → *Dataset* → Version 4 → *Download Dataset* → format **YOLOv8** (COCO JSON also works) → *Download zip to computer*, then unzip to:

```
data/pork-rasher/
├── data.yaml
├── train/{images,labels}/
├── valid/{images,labels}/
└── test/{images,labels}/
```

Check it loads: `python scripts/01_data_audit.py --data data/pork-rasher`

## 2. Synthetic verification dataset

Generated, not downloaded — every image has exactly known boxes (see `src/rasherqc/synthetic.py`):

```bash
rasherqc synth --out data/synthetic          # 600 / 150 / 150 images, seed 42, ~15 min on 2 CPU cores
```

The same seed always reproduces the same images.
