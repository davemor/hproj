#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/medium_test.yml"
DATA_ROOT=$(grep DATA_ROOT .env | cut -d= -f2 | tr -d "'\"")
RUNS_DIR="$DATA_ROOT/runs"

echo "=== Step 1: Calibrate Projector ==="
uv run hproj calibrate-projector --config "$CONFIG"

# extract the run_id from the most recent run directory
RUN_ID=$(ls -t "$RUNS_DIR" | head -1)
echo "Run ID: $RUN_ID"

echo "=== Step 2: Calibrate Classifier ==="
uv run hproj calibrate-classifier --run-id "$RUN_ID"

echo "=== Step 3: Estimate Curve ==="
uv run hproj estimate-curve --run-id "$RUN_ID"

echo "=== Step 4: Intrinsic Dims ==="
uv run hproj intrinsic-dims --run-id "$RUN_ID"

echo "=== Step 5: Thresholds ==="
uv run hproj thresholds --run-id "$RUN_ID"

echo "=== Step 6: Generalisation ==="
uv run hproj generalise --run-id "$RUN_ID"

echo "=== Step 7: Report ==="
uv run hproj report --run-id "$RUN_ID"

echo "=== Pipeline complete ==="
echo "Results in: $RUNS_DIR/$RUN_ID"
