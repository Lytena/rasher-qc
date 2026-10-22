# Automated Detection of Pork-Rasher Packaging and Placement Defects

**MMA3001 Numerical Methods and Machine Learning — Individual Project Report**
Author: Yijing · Repository: `rasher-qc` · October 2026

> Every number in this report is copied from `results/<dataset>/SUMMARY.md`, which `scripts/06_summary.py` generates from the experiment outputs. GPU results were produced on an RTX 4070 laptop; anything measured on the 2-core cloud Xeon used during development is marked as such.

---

## 1. Engineering problem

**Context.** Sliced bacon is packed by laying several rashers in a tray and heat-sealing a lidding film onto the tray flange. Faults arise at both stages: rashers end up *loose* (out of position, overhanging) or *twisted*; the film is *unsealed* along part of the flange, *wrinkled*, or otherwise damaged (*packaging-error*). An unsealed pack leaks, loses its protective atmosphere and spoils before its use-by date; an untidy pack is rejected by the retailer. Inspection is still largely manual — one person watching trays pass at around one per second — which is tiring, inconsistent and leaves no record.

**Problem addressed (narrowed scope).** From one top-view RGB image of a sealed tray on the conveyor:
1. **detect and localise** each defect of the five annotated types (box, class, confidence);
2. **decide** whether the tray should PASS, FAIL, or go to a person (REVIEW);
3. **establish** whether this is accurate *and* fast enough for in-line use, and how far the accuracy figures can be trusted.

**Why it matters.** Missing an unsealed pack is a food-safety and recall risk; failing a good pack costs a re-check. The method is therefore judged mainly on how many defective trays it catches at an acceptable false-reject rate, not only on a detection score.

**Intended use.** Decision support at the end of the packing line: clearly good trays pass, clearly defective trays are diverted, uncertain ones are routed to an operator, and every tray gets a machine-readable record (`results.json`) for traceability. It is not intended to replace the operator for defect types that the data cannot validate (Section 5).

**Limits of the scope.** Only the five annotated defect types and the camera set-up of the dataset (fixed overhead camera, dark textured belt, clear trays, 720 × 540 images). No seal-strength or leak measurement (a seal that looks intact can still leak), no weight or slice-count checks, no hardware integration, and no claims about other products, tray types or lighting.

## 2. Inputs, outputs and data

### 2.1 Top-level inputs and outputs

| | Item | Format, units, domain | Meaning and use |
|---|---|---|---|
| In | Tray image | RGB, uint8 (0–255), 720 × 540 px; accepted range 240–8192 px per side; one tray, top view | The only input at run time |
| In | Training data | YOLO export (`data.yaml`; `labels/*.txt` = class, centre x, centre y, width, height, normalised 0–1) or COCO JSON (pixels) | Training and evaluation only |
| Out | Decision | `PASS`, `REVIEW`, `FAIL`, `UNSUPPORTED_INPUT` | Action on the line |
| Out | Tray score | 0–1: highest confidence among detected defects | Ranking; threshold setting |
| Out | Defect list | class; confidence 0–1; box `x1, y1, x2, y2` in pixels (origin top-left); box area as a fraction of the image | Shows the operator where to look; rough severity ranking |
| Out | Run report | `results.json`, annotated JPEGs, `summary.html` | Traceability and a supervisor overview |

**Invalid, missing or unsupported inputs** (`rasherqc.io.validate_image`). Missing or undecodable files, greyscale images (colour is part of the evidence), RGB images whose three channels are identical, images too small to resolve a crease (side < 240 px), and grossly under- or over-exposed images (mean grey level outside 20–235, or more than 25 % of pixels clipped) are reported as `UNSUPPORTED_INPUT` with the reason. They are never passed automatically. RGBA images are accepted with the alpha channel dropped, and a warning is recorded. When training data is loaded, unknown class ids or malformed label lines raise an explicit error rather than being skipped. The inputs and outputs of individual functions are documented in Google-style docstrings and in the generated HTML API documentation (`docs/api/index.html`).

### 2.2 The provided dataset — and what the audit found

*Pork Rasher Error (Packaging)* (Roboflow Universe, version 4, CC BY 4.0). It contains 3,549 images at 720 × 540: 3,349 training images (three exposure-augmented copies, ±10 %, of 1,085 source frames), 120 validation and 80 test images, with boxes for five classes. Before modelling, `scripts/01_data_audit.py` and `scripts/00_prepare_real.py` audited it. Three findings changed the project:

1. **The images are video frames, and the predefined split leaks.** File names keep the source frame number (`frame_with_red_<n>`). For each validation and test image the distance to the nearest training frame was computed: the median is **2.5 frames for test and 3 for validation**, 57 % of test images are within 3 frames of a training image, all are within 30, and 5 % share the exact frame number (they differ only by the exposure augmentation). A test tray is therefore usually the *same physical tray* seen a fraction of a second after a training image, so scores on the predefined test split will be optimistic. A perceptual-hash check confirmed 143 valid/test images that are near-duplicates of training images.
2. **Classes are heavily imbalanced.** In training there are 1,712 `packaging-error`, 1,055 `unsealed`, 72 `twisted-meat`, 66 `loose-meat` and 18 `wrinkle` boxes (all three augmented copies counted). The validation and test splits contain no `wrinkle` boxes at all and only 3–4 `loose-meat` and `twisted-meat` boxes each. Per-class accuracy can only be estimated for `packaging-error` and `unsealed`. A class is reported as *evaluable* only with at least 10 test boxes; below that, a single box changes AP by 10 points or more.
3. **Few good trays to test against.** Only 12 of the 80 test images contain no defect, so a false-reject rate measured on the test split has a wide confidence interval.

**Response.** Two dataset variants are built automatically (`rasherqc.dataset`):
* **Predefined split (`-reduced`)**: the given split, keeping one augmented copy per training frame. The detector applies its own, wider colour augmentation, so the copies mostly triple training time.
* **Leakage-aware block split (`-block`)**: all source frames are cut into blocks of 300 consecutive frame numbers; whole blocks go to test or validation, and training frames within 60 frames of any held-out frame are removed. The result is 872 training, 155 validation and 173 test frames, with 72 training frames removed by the guard band. The block split also happens to contain more test boxes for the two evaluable classes (98 `packaging-error`, 51 `unsealed`) and more good trays (39), so its estimates are less noisy.

Comparing the two splits measures how optimistic the predefined split is, instead of assuming it (Section 5.4).

### 2.3 Synthetic verification dataset

A procedural generator (`rasherqc.synthetic`) renders tray images in the same YOLO layout, with the same class names, and with **exactly known boxes**. This plays the role that manufactured solutions play in numerical methods: it verifies that the code is correct — labels, metrics, thresholds, decision logic — independently of noisy real labels, and it isolates single factors such as defect size and glare. It renders shingled streaky rashers on a tray with a sealed flange and film reflections, plus each defect type. It says nothing about accuracy on real trays. In fact, the real scenes (clear trays partly in view on a dark textured belt) look quite different, which is itself a reminder of why the synthetic set is used only for verification.

## 3. Computational solution

### 3.1 Pipeline

`image → validation (io) → YOLOv8n detector → per-class boxes + confidences → tray score s = max defect confidence → decision (PASS / REVIEW / FAIL) → report`

### 3.2 Main method: YOLOv8n object detector with transfer learning

**How it works.** YOLOv8 is a one-stage, anchor-free convolutional detector. A CSP-Darknet backbone with C2f blocks extracts feature maps at strides 8, 16 and 32. A path-aggregation neck fuses them, so small defects (a lifted film corner) and large ones (a loose rasher) are both represented. A decoupled head predicts, for every cell of each feature map, the class probabilities and the distances from the cell to the four box edges (each as a discrete probability distribution). Training minimises the weighted sum of three losses: binary cross-entropy for the classes, CIoU for box overlap, and distribution focal loss for the edge distributions. Ground-truth boxes are assigned to cells by a task-aligned assigner. The loss is minimised by mini-batch stochastic gradient descent with the AdamW optimiser, and the model is stored at the epoch with the best validation fitness (0.1·mAP@0.5 + 0.9·mAP@0.5:0.95). At inference, non-maximum suppression (IoU 0.6) removes duplicate boxes.

**Why it was selected.**
* **It localises.** The brief asks to detect *and localise*, and an operator needs to see where the fault is.
* **Speed.** One forward pass per image suits a line running at about one tray per second. The nano variant has 3.0 M parameters, so it can run on a CPU (Section 6).
* **Little data.** Starting from COCO-pretrained weights reuses generic edge, texture and shape features, which matters with only 1,085 distinct training frames. This is tested directly against training from scratch (Section 4).
* **The data matches.** The dataset is labelled with boxes, so no re-annotation is needed.
* **Reproducibility.** A widely used, documented library; the same code runs on CPU and GPU.

**Assumptions.**
1. Defects are visible in a single top-view RGB frame.
2. Box labels are consistent enough to learn from.
3. The training, validation and test images come from the same process: same camera, belt and trays.
4. A tray's status can be summarised by its most confident defect.

**Settings** (`rasherqc.config`, `rasherqc.detector.train`). Input 416 px on the long side; 60 epochs on GPU (20 on the CPU development run), batch 16, AdamW with the learning rate chosen automatically (≈1.1 × 10⁻³), early-stopping patience 15, seed 42 with `deterministic=True`. Augmentation: mosaic (switched off for the last 5 epochs), horizontal and vertical flips (a top-view camera has no preferred orientation), and HSV jitter. Rotation is disabled, because the trays always run along the belt.

**Implementation.** `rasherqc.detector` wraps Ultralytics 8.4. Our wrapper fixes one silent pitfall: Ultralytics expects OpenCV's BGR channel order, while the rest of the code uses RGB. A unit test checks that passing an image as an array or as a file gives identical detections.

### 3.3 Tray decision

With tray score *s*: **FAIL** if *s* ≥ *t*<sub>fail</sub>; **REVIEW** if *t*<sub>low</sub> ≤ *s* < *t*<sub>fail</sub>; otherwise **PASS**. Both thresholds are chosen on the **validation** split. *t*<sub>fail</sub> is the highest threshold that still catches at least 95 % of defective validation trays. *t*<sub>low</sub> is the highest that catches at least 99 %, so trays in between go to a person. The targets express the unequal costs of the two errors (Section 1). They are engineering choices to be agreed with the plant, not results.

### 3.4 Known limitations of the method

* Boxes are axis-aligned, so box area overstates the true defect area. It is useful for ranking severity, not for measuring it.
* The model cannot detect defect types it has (almost) never seen: there are 6 `wrinkle` training boxes in the reduced set.
* Confidence scores are not calibrated probabilities. Thresholds must be re-chosen if the camera, lighting or product changes.
* Ultralytics is licensed under AGPL-3.0. A closed-source commercial deployment would need a commercial licence or a different detector.

## 4. Alternative solutions

| | Method | What it tests | Localises? |
|---|---|---|---|
| A | **YOLOv8n, COCO-pretrained** (main) | — | yes |
| B | **YOLOv8n trained from scratch**: same architecture, data, settings and seed; random initial weights | whether transfer learning is worth it at this data size | yes |
| C | **Hand-crafted features + random forest**: 80 colour, texture, morphological and orientation features; 500 trees chosen by 5-fold cross-validation on the training split | whether a classical, interpretable, cheap pipeline is good enough for the PASS/FAIL decision | no (image-level only) |

The baseline's features were designed around the physics of each defect. In a correct pack the rashers lie parallel, so edge orientation is coherent; a loose rasher adds a second dominant orientation. Wrinkles and lifted film edges are thin, bright structures, picked up by a white top-hat filter. An unsealed flange shows a dark gap, picked up by a black top-hat filter. A 3 × 3 spatial pyramid pools these responses so that a small local defect is not averaged away. All three methods produce a 0–1 tray score and pass through the *same* threshold procedure (Section 3.3), so their tray decisions are directly comparable.

{{COMPARISON_TABLE}}

{{COMPARISON_DISCUSSION}}

## 5. Validation

### 5.1 What is validated, and how

| Question | Method | Metric / criterion |
|---|---|---|
| Is the code correct? | Verification tests against hand-calculated cases; synthetic data with exact ground truth; our metric code cross-checked against Ultralytics' validator | Tests pass; our mAP equals the library's to within interpolation differences |
| How well are defects localised? | Withheld test split | AP@0.5 per evaluable class, mAP@0.5, mAP@0.5:0.95 |
| How good is the tray decision? | Thresholds fixed on validation, applied **once** to test; 95 % bootstrap confidence intervals (1,000 resamples of test images) | FAIL recall (share of defective trays caught), false-reject rate, ROC AUC, escape and review rates |
| How optimistic is the predefined split? | Same pipeline on the leakage-aware block split | Difference in the metrics above |
| How robust is it? | Controlled perturbations of test images, thresholds kept fixed | Recall and false-reject rate against perturbation strength (Section 6.4) |

**Why these metrics.** mAP is the standard detection measure. It includes localisation (IoU ≥ 0.5) and ranks predictions at every confidence level. The line, however, acts on trays, so the decision metrics come first: recall of FAIL trays, because escapes are the costly error, and the false-reject rate, because it is the cost of operating the system. Bootstrap intervals are reported because the test set is small. **Influence on model selection:** validation data chose the best epoch, the thresholds and the baseline's hyper-parameters (by cross-validation on training data). The test split was used once, after every choice had been made.

### 5.2 Verification of the implementation

* **Metrics** (`tests/test_metrics.py`, 21 tests). IoU is checked for identical, half-overlapping, contained, touching and degenerate boxes. AP is checked against a three-prediction case worked out on paper: 0.5·1 + 0.5·⅔ = 0.833. ROC AUC is checked against scikit-learn, including ties, and duplicate detections are confirmed to count as false positives. Writing these tests exposed a real bug: the first COCO-style AP returned 0.0099 instead of 0 for a detector that never matches, because a sentinel point leaked into the 101-point interpolation. They also showed that Ultralytics' trapezoidal AP is capped at 0.995 for a perfect detector, which explains part of the small gap between our numbers and the library's.
* **Cross-check with the library.** On the real test split our mAP@0.5 for the pretrained detector is 0.344 (COCO 101-point rule) and 0.346 (trapezoid rule) against Ultralytics' 0.338; for the scratch model 0.241 / 0.243 against 0.256. On the synthetic set: 0.978 against 0.972. The remaining differences come from interpolation details and from the classes with 3–4 boxes, where a single detection moves AP by several points; on the well-populated `unsealed` class the two agree to within 0.015.
* **Data handling.** Boxes read back from YOLO files match the rendered boxes to within 0.01 px. Converting the YOLO data to COCO and loading it again gives identical annotations. A copy of a training image deliberately planted in the test split is flagged by the leakage check.
* **Synthetic end-to-end run.** After only 10 CPU epochs the pretrained detector reached test mAP@0.5 = 0.978 (per class 0.91–1.00) and a perfect tray ROC AUC on the synthetic set, catching 104 of 106 defective trays with no false rejects. The pipeline therefore can learn and score localised defects correctly when the labels are exact; the much lower real-data scores are a property of the real problem and data, not a bug. The baseline reached AUC 0.82 on the same data.
* **Input domain.** Twelve tests confirm that every class of unsupported input listed in Section 2.1 is rejected with a reason, and that an end-to-end inspection of a folder containing a good image, a greyscale image and a corrupt file gives FAIL/PASS, UNSUPPORTED, UNSUPPORTED respectively.

In total **80 tests** pass (`results/test_report.html`), and they run on every push via GitHub Actions.

### 5.3 Results on the real data

{{REAL_RESULTS}}

### 5.4 How optimistic is the predefined split?

{{LEAK_RESULTS}}

### 5.5 What the validation cannot establish

* Performance on `wrinkle`, `loose-meat` and `twisted-meat`. With 0–4 test boxes each, any AP for these classes is anecdotal. The tray decision includes them, but its recall is dominated by `packaging-error` and `unsealed`.
* Performance on other days, shifts, camera positions, lighting, trays or products. All frames appear to come from a single video session. Even the block split only separates moments within one session.
* Whether the labels are correct. Some test boxes look ambiguous (for example, what counts as `packaging-error` versus `unsealed`), and nobody re-annotated them independently.
* Physical seal integrity: the system only ever sees what the camera sees.

## 6. Optimisation and performance

{{PERF_SECTION}}

## 7. Limitations and lessons learned

{{LESSONS}}

## 8. Reflection on the use of AI

> **Note to myself before submission:** this section must describe my own use of AI honestly and in my own words. The facts below are taken from the project log; edit the evaluation to reflect what I actually checked and understood.

**How AI was used.** Claude (Anthropic), used as an agentic assistant with access to a cloud workspace and, later, to my laptop's project folder, wrote most of the code and tests, proposed the method and validation design, ran the CPU experiments and the synthetic verification, and drafted this report from the generated results. I selected the project, supplied the dataset, ran the GPU pipeline on my RTX 4070 laptop, reported errors back, and reviewed the outputs. The commit history records this split of the work; every commit made by the assistant is labelled as co-authored.

**Where the AI was wrong, and how it was caught.** These cases are the main reason I treated every output as unverified until tested:

| Problem | How it was caught | Fix |
|---|---|---|
| COCO-style AP returned 0.0099 instead of 0 for a detector that never matches (a sentinel point leaked into the interpolation) | hand-calculated unit test failed | AP computed from observed points only; test kept |
| Peak-memory measurement used `resource`, which exists only on Linux/macOS | pipeline crashed on my Windows laptop | `psutil` used instead |
| Process pools and data-loader workers re-import the script on Windows | anticipated after reading the multiprocessing docs; confirmed by the laptop logs | serial fallback / 0 workers on Windows |
| Set-up script assumed a Python version supported by PyTorch and scikit-image; my default Python 3.15 had no scikit-image build | install failed on my laptop | script selects Python 3.10–3.13 via the `py` launcher |
| Roofline compared GPU timings with a CPU compute ceiling (two different machines in one plot) | spotted when reviewing the first performance results | roofline uses CPU timings only |
| "Parallel" feature-extraction timing on Windows was actually a second serial run | inconsistent timings in the summary | flagged in the output |
| The synthetic tray images were designed before the real images were seen and look quite different | visual comparison with the dataset | synthetic data restricted to code verification |
| Several time estimates were wrong (e.g. "30–60 min" for a run that took about three hours) | observation | — |

The most valuable AI contribution was not code generation but the **data audit**: it noticed from the file names that the images are video frames and quantified the leakage of the predefined split, which I would not have checked.

**How the work was verified.** (i) Metric code tested against cases worked out by hand and against scikit-learn; (ii) detection metrics cross-checked against the independent Ultralytics validator; (iii) the full pipeline run on synthetic data with exact labels; (iv) a FLOP counter checked by hand calculation and against `thop`; (v) figures and error galleries inspected visually; (vi) results reproduced on two machines (cloud CPU, laptop GPU).

**Evaluation.** AI made it possible to complete a broader and more rigorous project than I could have done alone in the time, and it was consistently strongest at structure, testing discipline and spotting validation problems. Its weaknesses were environment-specific assumptions (operating system, Python versions, hardware) and overconfident estimates — exactly the things that only fail when code meets a real machine. The main risk for me is understanding: code I did not write is code I must be able to explain. To address this, before the presentation I am working through the parts most likely to be questioned — the AP calculation (Section 5.2), the YOLO loss terms and early stopping (Section 3.2), the block split (Section 2.2) and the roofline argument (Section 6) — until I can explain them without notes. *[Edit to state what I actually did.]* If I did the project again I would keep a dated log of prompts and decisions from the start rather than reconstructing it afterwards, and I would run the pipeline on the target operating system from the first day.

## 9. References

1. Jocher, G., Chaurasia, A., Qiu, J. *Ultralytics YOLOv8*, v8.x, 2023–2026. https://github.com/ultralytics/ultralytics (AGPL-3.0).
2. Redmon, J., Divvala, S., Girshick, R., Farhadi, A. "You Only Look Once: Unified, real-time object detection." *CVPR*, 2016.
3. Lin, T.-Y. et al. "Microsoft COCO: Common objects in context." *ECCV*, 2014. (Pretraining data; COCO AP definition.)
4. Everingham, M. et al. "The PASCAL Visual Object Classes (VOC) challenge." *IJCV* 88, 2010. (IoU ≥ 0.5 matching and AP.)
5. Zheng, Z. et al. "Distance-IoU loss: faster and better learning for bounding box regression." *AAAI*, 2020. (CIoU.)
6. Li, X. et al. "Generalized focal loss." *NeurIPS*, 2020. (Distribution focal loss.)
7. Breiman, L. "Random forests." *Machine Learning* 45, 2001.
8. Ojala, T., Pietikäinen, M., Mäenpää, T. "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns." *IEEE TPAMI* 24, 2002.
9. Haralick, R. M., Shanmugam, K., Dinstein, I. "Textural features for image classification." *IEEE TSMC* 3, 1973. (GLCM.)
10. Williams, S., Waterman, A., Patterson, D. "Roofline: an insightful visual performance model for multicore architectures." *CACM* 52(4), 2009.
11. Efron, B., Tibshirani, R. *An Introduction to the Bootstrap.* Chapman & Hall, 1993.
12. Kapoor, S., Narayanan, A. "Leakage and the reproducibility crisis in machine-learning-based science." *Patterns* 4(9), 2023.
13. Dataset: "Pork Rasher Error (Packaging)", v4, workspace hello-2aqe0, Roboflow Universe, 2025, CC BY 4.0. https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging
14. Pedregosa, F. et al. "Scikit-learn: Machine learning in Python." *JMLR* 12, 2011; van der Walt, S. et al. "scikit-image: image processing in Python." *PeerJ* 2, 2014.
