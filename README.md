# rasher-qc — automated inspection of packaged pork rashers

MMA3001 Numerical Methods and Machine Learning — individual project.

Detects and localises **placement and packaging defects** in top-view images of pork-rasher (bacon) trays — `loose-meat`, `twisted-meat`, `packaging-error`, `unsealed`, `wrinkle` — and turns the detections into a tray decision **PASS / REVIEW / FAIL** that a packing line can act on. The project also asks whether the method is accurate and fast enough for in-line use.

| | |
|---|---|
| **Input** | RGB tray image, uint8, 720 × 540 px (dataset format); one tray, top view |
| **Output** | per tray: decision, defect score (0–1), list of defects with class, confidence, box (px) and box area fraction; per run: `results.json`, annotated images, `summary.html` |
| **Main method** | YOLOv8n object detector, fine-tuned from COCO weights (transfer learning) |
| **Alternatives compared** | same detector trained from scratch; hand-crafted features + random forest (no localisation) |
| **Validation** | thresholds fixed on the validation split, reported once on the test split with bootstrap 95 % CIs; metric code verified against hand-computed cases and against Ultralytics; synthetic “manufactured-solution” dataset with exact ground truth |
| **Key result** | On unseen trays (leakage-aware split) the pretrained detector reaches AP@0.5 0.87 (`unsealed`) / 0.70 (`packaging-error`) and tray ROC AUC 0.86; catching ~90 % of defective trays costs ~41 % false rejects. Runs in 63 ms/tray on a laptop CPU (budget 1,000 ms). The predefined Roboflow split leaks (test frames are 2–3 video frames from training frames) — see report §2.2 and §5.4. |
| **Report** | [`docs/REPORT.md`](docs/REPORT.md) · results tables: [`results/<dataset>/SUMMARY.md`](results/) · API docs: [`docs/api/index.html`](docs/api/index.html) · test report: [`results/test_report.html`](results/test_report.html) |

---

## 1. Quick start

```bash
git clone https://github.com/<your-username>/rasher-qc.git
cd rasher-qc
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                   # installs the package + test/doc tools
mkdir -p models && curl -L -o models/yolov8n_coco_pretrained.pt \
     https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
pytest                                                    # ~1 min; detector tests skip until a model is trained
```

Tested with Python 3.12/3.13, PyTorch 2.11, Ultralytics 8.4 on Linux (cloud CPU) and Windows 11 (RTX 4070 laptop). PyTorch and scikit-image do not yet ship builds for Python 3.14+, so use 3.10–3.13. CPU only is fine for inference and for the synthetic set; train on the real dataset with a GPU (see §3).

### Inspect a folder of tray images

```bash
rasherqc inspect --weights models/pork-rasher-block/yolo_pretrained.pt \
                 --data data/pork-rasher --images path/to/trays --out inspection/ \
                 --fail-conf 0.079 --review-low 0.048     # thresholds chosen on validation, report §5.3
# -> inspection/results.json, inspection/annotated/*.jpg, inspection/summary.html
```

Images that are greyscale, too small, badly exposed, corrupt or unreadable are reported as `UNSUPPORTED_INPUT` with the reason — never passed automatically.

## 2. Data

Image data is not committed. See [`data/README.md`](data/README.md).

* **Real dataset:** *Pork Rasher Error (Packaging)*, Roboflow Universe, Version 4 — 1,955 original images (3,549 after training-set augmentation), 720 × 540, bounding boxes, predefined train/valid/test splits. **Licence CC BY 4.0** — © the dataset author (“Hello”, workspace `hello-2aqe0`), <https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging>. Download in YOLOv8 (or COCO) format and unzip to `data/pork-rasher/`.
* **Synthetic verification set:** procedurally rendered trays with exact boxes for the same five classes (`rasherqc synth`); used to verify the pipeline and to isolate factors such as defect size and glare. It is *not* evidence of real-world accuracy.

## 3. Reproducing the results

Everything is scripted; each step writes JSON tables and PNG figures to `results/<dataset>/`.

```bash
python run_pipeline.py --zip <roboflow-yolov8-export.zip> --device 0   # real data, GPU 0  (~3 h on an RTX 4070 laptop)
python run_pipeline.py --synthetic --device cpu                       # synthetic verification set
```

`run_pipeline.py` builds the two dataset variants (predefined split `-reduced`, leakage-aware `-block`), then trains, evaluates and profiles everything for both, runs the tests and builds the docs. Finished steps are skipped on restart (`--force` redoes them). **Windows + NVIDIA GPU:** put the Roboflow zip in `Downloads` and double-click `run_on_windows_gpu.bat` — it picks a supported Python, installs CUDA PyTorch and runs the pipeline. Linux/macOS: `DEVICE=0 ./run_all.sh <zip>`.

or step by step:

| step | script | produces |
|---|---|---|
| 0 | `scripts/00_prepare_real.py` | frame-gap leakage audit; builds `-reduced` and `-block` dataset variants |
| 1 | `scripts/01_data_audit.py` | class/box statistics, cross-split near-duplicate (leakage) check, example figure |
| 2 | `scripts/02_train_detectors.py` | `models/<dataset>/yolo_{pretrained,scratch}.pt`, training curves |
| 3 | `scripts/03_evaluate.py` | detection AP/mAP, tray decision with thresholds from validation, baseline, CIs, error gallery |
| 4 | `scripts/04_robustness.py` | performance under exposure, saturation, glare, blur, noise, JPEG |
| 5 | `scripts/05_performance.py` | latency vs resolution, FLOPs, arithmetic intensity, roofline, profiling, throughput |
| 6 | `scripts/06_summary.py` | `results/<dataset>/SUMMARY.md` — every number quoted in the report |
| 7 | `scripts/07_operating_points.py` | recall vs false-reject trade-off; resolution sweep on evaluable classes |

No GPU? [`notebooks/train_real_data_colab.ipynb`](notebooks/train_real_data_colab.ipynb) runs the whole pipeline on a free Colab GPU. [`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb) walks through the method and results interactively.

Determinism: all scripts use seed 42; Ultralytics runs with `deterministic=True`. CPU results are bit-reproducible on the same machine; GPU training can differ slightly between runs/hardware (non-deterministic cuDNN kernels). Evaluation is reproducible across machines: the laptop-GPU models re-evaluated on a Linux CPU give identical mAP.

The committed `results/` hold the runs reported: `pork-rasher` (data audit), `pork-rasher-reduced` (predefined split), `pork-rasher-block` (leakage-aware split, the headline), `synthetic` (verification).

## 4. Repository layout

```
rasher-qc/
├── src/rasherqc/          # the package (documented with Google-style docstrings)
│   ├── io.py              #   image reading + input-domain validation
│   ├── dataset.py         #   YOLO/COCO loading, tray labels, near-duplicate check
│   ├── synthetic.py       #   synthetic verification data with exact labels
│   ├── detector.py        #   YOLOv8 training / inference wrapper (main method)
│   ├── features.py        #   hand-crafted features (reference + optimised)
│   ├── baseline.py        #   random-forest baseline
│   ├── metrics.py         #   IoU, AP/mAP, tray metrics, bootstrap — written from scratch
│   ├── decision.py        #   PASS / REVIEW / FAIL rule, defect extent
│   ├── robustness.py      #   perturbations for sensitivity analysis
│   ├── performance.py     #   FLOPs, arithmetic intensity, roofline, timing
│   ├── report.py, cli.py  #   inspection report and command line
├── scripts/               # numbered experiment pipeline (see §3)
├── tests/                 # pytest suite (verification of metrics, I/O, generator, features, ...)
├── notebooks/             # walkthrough + Colab GPU notebook
├── docs/REPORT.md         # project report (also REPORT.docx/.pdf);  docs/api/ = generated HTML API docs
├── results/<dataset>/     # figures/, tables/, SUMMARY.md, training runs
├── models/<dataset>/      # trained pretrained-YOLO weights (6.5 MB each)
├── data/README.md         # how to obtain the data (data itself is git-ignored)
└── run_pipeline.py, run_on_windows_gpu.bat, run_all.sh, pyproject.toml, requirements.txt, LICENSE
```

## 5. Documentation and tests

```bash
pdoc src/rasherqc -o docs/api --docformat google              # HTML API docs
pytest --html=results/test_report.html --self-contained-html   # test report
pytest --cov=rasherqc                                          # coverage
```

The test suite is mainly *verification*: IoU/AP/AUC against hand-calculated cases and scikit-learn, the dataset loader against a YOLO↔COCO round trip, the synthetic generator against its own masks, the FLOP counter against a hand calculation, and the optimised feature code against the reference version. GitHub Actions (`.github/workflows/tests.yml`) runs the tests and builds the docs on every push.

## 6. Licences

* Code: MIT (see `LICENSE`).
* Dataset: CC BY 4.0 — attribution above; not redistributed here.
* `ultralytics` (YOLOv8) is AGPL-3.0. Using it as a library in this coursework is fine; a commercial, closed-source deployment of this code would need an Ultralytics enterprise licence or an alternative detector (e.g. torchvision's Apache-2.0 models). This is discussed in the report.
* COCO-pretrained weights are provided by Ultralytics under the same AGPL-3.0 terms.

## 7. Use of AI

Generative AI (Claude, Anthropic) was used in developing this project; how, what was checked and what was changed is described in the AI-use reflection in the report.
