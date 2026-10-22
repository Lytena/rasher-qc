# Automated Detection of Pork-Rasher Packaging and Placement Defects

**MMA3001 Numerical Methods and Machine Learning — Individual Project Report**
Author: Yijing · Repository: `rasher-qc` · Date: October 2026

> **Status of results.** All numbers in this version come from the *synthetic verification dataset* (Section 2.3), because the provided Roboflow dataset could not be processed in time on the available hardware. Every script runs unchanged on the real dataset (`DEVICE=0 ./run_all.sh data/pork-rasher`); Section 9 lists what to re-check once it has run. Numbers are copied from `results/synthetic/SUMMARY.md`, which the code generates.

---

## 1. Engineering problem

**Context.** Sliced bacon is packed by placing several rashers in a shingled (overlapping) pattern on a plastic tray, then heat-sealing a lidding film to the tray flange. Faults arise at both stages: rashers end up *loose* (out of the pattern, overhanging) or *twisted*; the film is *unsealed* along part of the flange, *wrinkled*, or otherwise defective (*packaging-error*). An unsealed or damaged pack leaks, loses its modified atmosphere and spoils before its use-by date; a messy pack is rejected by retailers. Inspection is currently largely manual: one person watching trays at 60+ per minute, which is tiring, inconsistent and does not leave a record.

**Problem addressed (narrowed scope).** From one top-view RGB photograph of a sealed tray:
1. **detect and localise** each defect of the five annotated types (bounding box + class + confidence);
2. **decide** whether the tray should PASS, FAIL, or be sent to a person (REVIEW);
3. **establish** whether the method is accurate *and* fast enough for in-line use.

**Why it matters.** A missed unsealed pack is a food-safety and recall risk; a false reject costs a re-check or re-pack. The method must therefore be judged on its ability to catch defective trays at an acceptable false-reject rate — not only on a detection score.

**Intended use.** Decision support at the end of the packing line: automatically pass clearly good trays, fail clearly defective ones, and route uncertain trays to an operator. Every tray gets a machine-readable record (`results.json`) for traceability.

**Limits of scope.** Only the five annotated defect types and the camera geometry of the dataset (fixed overhead camera, 720 × 540). No seal-strength or leak measurement (a visually intact seal can still leak), no weight/count compliance, no real-time hardware integration, and no claims about other products or tray types.

## 2. Inputs, outputs and data

### 2.1 Top-level inputs and outputs

| | Item | Format / units / domain | Meaning / use |
|---|---|---|---|
| In | Tray image | RGB, uint8 0–255, 720 × 540 px (accepted: each side 240–8192 px, letter-boxed); one tray, top view | The only runtime input |
| In | Training data | YOLO export (`data.yaml`, `labels/*.txt`: class, cx, cy, w, h normalised 0–1) or COCO JSON (pixels) | Training and evaluation only |
| Out | Decision | `PASS` · `REVIEW` · `FAIL` · `UNSUPPORTED_INPUT` | Action on the line |
| Out | Tray score | 0–1, highest defect confidence | Ranking / threshold tuning |
| Out | Defects | class, confidence 0–1, box `x1,y1,x2,y2` in px (origin top-left), box area as fraction of image | Localisation for the operator; severity ranking |
| Out | Run report | `results.json`, annotated JPEGs, `summary.html` | Traceability, supervisor overview |

**Invalid, missing or unsupported inputs** (`rasherqc.io.validate_image`): unreadable or missing files, greyscale images (colour is evidence), images whose three channels are identical, images too small to resolve a wrinkle (< 240 px side), grossly under/over-exposed images (mean grey outside 20–235 or > 25 % clipped pixels) are returned as `UNSUPPORTED_INPUT` with the reason. RGBA is accepted with the alpha dropped (warning recorded). Such trays are never passed automatically. Dataset loading rejects unknown class ids and inconsistent label files with an explicit error. Function-level inputs and outputs are documented in the docstrings and the generated HTML API documentation (`docs/api/`).

### 2.2 Provided dataset

*Pork Rasher Error (Packaging)* (Roboflow Universe, Version 4, CC BY 4.0): 1,955 original images, 3,549 after augmentation of the training split (saturation and exposure variation), auto-oriented and resized to 720 × 540, predefined train/valid/test splits, bounding boxes for `loose-meat`, `packaging-error`, `twisted-meat`, `unsealed`, `wrinkle`. Two properties matter for validation: (i) augmented copies exist only in training, and (ii) the same physical tray may have been photographed more than once, which would leak information across splits — `scripts/01_data_audit.py` checks this with a perceptual (difference) hash.

### 2.3 Synthetic verification dataset

Before trusting any number from real data, the pipeline was verified on a *manufactured* dataset (`rasherqc.synthetic`) where the ground truth is exact — the image-processing analogue of the method of manufactured solutions. The generator renders a belt, a black or white tray with flange and glossy seal band, 4–6 shingled streaky rashers (lean/fat bands, rind, drop shadows), film haze and broad soft reflections, and the five defects: an out-of-pattern rasher (loose-meat), a jagged film tear with bright lips (packaging-error), a half-twist where the rasher width collapses and the fat edge swaps sides (twisted-meat), a flange section without seal plus a lifted film edge (unsealed), and narrow bright creases with a shadow (wrinkle). Acquisition varies exposure, saturation, colour temperature, blur and noise. 900 images (600/150/150), same YOLO layout and class order as the real export, so every script runs on both. The reflections deliberately act as a confounder for wrinkles.

What the synthetic set **can** establish: that the code is correct (labels, metrics, thresholds, decision logic), that the detector can learn localised defects of these shapes, and how accuracy depends on isolated factors (defect size, glare, exposure). What it **cannot** establish: accuracy on real trays — real defects are more varied, labels are noisier and real reflections are more complex.

<!-- RESULTS-SECTIONS -->
