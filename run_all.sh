#!/usr/bin/env bash
# Reproduce every result for one dataset.
#   ./run_all.sh data/pork-rasher            # real dataset (use a GPU: DEVICE=0 ./run_all.sh ...)
#   ./run_all.sh                             # synthetic verification set (generated if missing)
set -euo pipefail
DATA=${1:-data/synthetic}
DEVICE=${DEVICE:-cpu}
EPOCHS=${EPOCHS:-30}

if [ "$DATA" = "data/synthetic" ] && [ ! -f data/synthetic/data.yaml ]; then
  rasherqc synth --out data/synthetic
fi
python scripts/01_data_audit.py      --data "$DATA"
python scripts/02_train_detectors.py --data "$DATA" --epochs "$EPOCHS" --device "$DEVICE"
python scripts/03_evaluate.py        --data "$DATA" --device "$DEVICE"
python scripts/04_robustness.py      --data "$DATA" --device "$DEVICE"
python scripts/05_performance.py     --data "$DATA" --device "$DEVICE"
python scripts/06_summary.py         --data "$DATA"
pytest --html=results/test_report.html --self-contained-html
pdoc src/rasherqc -o docs/api --docformat google
