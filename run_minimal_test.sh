#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/minimal_test.yml"
DATA_ROOT=$(grep DATA_ROOT .env | cut -d= -f2 | tr -d "'\"")
RUNS_DIR="$DATA_ROOT/runs"

echo "=== Step 1: Calibrate Projector ==="
python -m hproj calibrate-projector --config "$CONFIG"

# extract the run_id from the most recent run directory
RUN_ID=$(ls -t "$RUNS_DIR" | head -1)
echo "Run ID: $RUN_ID"

echo "=== Step 2: Calibrate Classifier ==="
python -m hproj calibrate-classifier --run-id "$RUN_ID"

echo "=== Step 3: Estimate Curve ==="
python -m hproj estimate-curve --run-id "$RUN_ID"

echo "=== Step 4: Intrinsic Dims ==="
python -m hproj intrinsic-dims --run-id "$RUN_ID"

echo "=== Step 5: Thresholds ==="
python -m hproj thresholds --run-id "$RUN_ID"

echo "=== Step 6: Generalisation ==="
python -m hproj generalise --run-id "$RUN_ID"

echo "=== Pipeline complete ==="
echo "Results in: $RUNS_DIR/$RUN_ID"
