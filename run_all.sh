#!/usr/bin/env bash
# Reproduce every result (Linux/macOS). Thin wrapper around run_pipeline.py.
#   ./run_all.sh                                   # synthetic verification set, CPU
#   DEVICE=0 ./run_all.sh path/to/roboflow.zip     # real dataset (zip, YOLOv8 format) on GPU 0
set -euo pipefail
DEVICE=${DEVICE:-cpu}
EPOCHS=${EPOCHS:-60}
if [ $# -eq 0 ]; then
  python run_pipeline.py --synthetic --device "$DEVICE" --epochs "$EPOCHS"
else
  python run_pipeline.py --zip "$1" --device "$DEVICE" --epochs "$EPOCHS"
fi
