"""Cross-platform driver for the whole experiment pipeline (Windows, Linux, macOS).

Examples::

    python run_pipeline.py --zip "C:/Users/me/Downloads/Pork_Rasher_Error-Packaging-_v4i_yolov8.zip" --device 0
    python run_pipeline.py --data data/pork-rasher --device 0 --epochs 60
    python run_pipeline.py --synthetic                      # verification set, CPU

For the real dataset it runs, in order: data audit, leakage-aware dataset
variants, then for both the predefined split (``-reduced``) and the block
split (``-block``): training of both detectors, evaluation, robustness,
performance and the summary; finally the tests and the API documentation.
Steps whose outputs already exist are skipped unless ``--force`` is given,
so an interrupted run can simply be restarted.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(*args: str) -> None:
    print("\n>>>", " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zip", help="Roboflow YOLOv8 export zip (unzipped to data/pork-rasher)")
    ap.add_argument("--data", default="data/pork-rasher")
    ap.add_argument("--synthetic", action="store_true", help="run on the synthetic verification set instead")
    ap.add_argument("--device", default="0", help="'0' = first GPU, 'cpu' = CPU")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--variants", nargs="+", default=["reduced", "block"])
    ap.add_argument("--force", action="store_true", help="redo steps whose outputs exist")
    a = ap.parse_args()

    weights = ROOT / "models" / "yolov8n_coco_pretrained.pt"
    if not weights.is_file():
        import urllib.request

        weights.parent.mkdir(parents=True, exist_ok=True)
        print("downloading COCO-pretrained YOLOv8n weights ...")
        urllib.request.urlretrieve("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt", weights)

    if a.synthetic:
        data = ROOT / "data" / "synthetic"
        if not (data / "data.yaml").is_file():
            run("-m", "rasherqc.cli", "synth", "--out", str(data))
        datasets = [data]
        run("scripts/01_data_audit.py", "--data", str(data))
    else:
        data = (ROOT / a.data).resolve()
        if a.zip and not (data / "data.yaml").is_file():
            print(f"unzipping {a.zip} -> {data}")
            data.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(a.zip) as z:
                z.extractall(data)
        if not (data / "data.yaml").is_file():
            sys.exit(f"dataset not found at {data}; pass --zip <Roboflow YOLOv8 zip>")
        run("scripts/01_data_audit.py", "--data", str(data))
        run("scripts/00_prepare_real.py", "--data", str(data))
        datasets = [data.parent / f"{data.name}-{v}" for v in a.variants]

    for d in datasets:
        tag = d.name
        mdir = ROOT / "models" / tag
        for which in ("pretrained", "scratch"):
            if a.force or not (mdir / f"yolo_{which}.pt").is_file():
                run("scripts/02_train_detectors.py", "--data", str(d), "--which", which, "--epochs", str(a.epochs),
                    "--batch", str(a.batch), "--device", a.device)
        tables = ROOT / "results" / tag / "tables"

        def done(name: str) -> bool:
            return not a.force and (tables / name).is_file()

        if not done("evaluation.json"):
            run("scripts/03_evaluate.py", "--data", str(d), "--device", a.device)
        if not done("robustness.json") or "rf_baseline" not in (tables / "robustness.json").read_text():
            run("scripts/04_robustness.py", "--data", str(d), "--device", a.device,
                "--methods", "yolo_pretrained", "yolo_scratch", "rf_baseline")
        if not done("performance.json"):
            run("scripts/05_performance.py", "--data", str(d), "--device", a.device)
        if not done("operating_points.json"):
            run("scripts/07_operating_points.py", "--data", str(d), "--device", a.device)
        run("scripts/06_summary.py", "--data", str(d))

    run("-m", "pytest", "--html=results/test_report.html", "--self-contained-html", "-q")
    run("-m", "pdoc", "src/rasherqc", "-o", "docs/api", "--docformat", "google")
    print("\nDone. Headline numbers: results/<dataset>/SUMMARY.md")


if __name__ == "__main__":
    main()
