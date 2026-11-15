#!/usr/bin/env bash
# Batch 1/2/4 curve for 005. Reuses the existing full-val batch2 report.
set -euo pipefail
cd "${PROJECT_DIR:-$HOME/mws}"
export CASE_DATA_DIR="${CASE_DATA_DIR:-$PWD/data/organizer_v2_1}"
export SCORER_VERSION=2.1
export ALLOW_LEGACY_CONTRACT=1
export CATALOG_PATH="$CASE_DATA_DIR/catalog.json"
export TEMPORAL_CATALOG_PATH="$CASE_DATA_DIR/catalog.json"
export MAX_INPUT_TOKENS=12288 MAX_SEQUENCE_TOKENS=16384 MAX_NEW_TOKENS=4096
export PROMPT_LOOKUP_TOKENS=0 JSON_CONSTRAINED=1
export JSON_GRAMMAR_BACKEND=torch_native JSON_GRAMMAR_THREADS=8
export TEMPORAL_NORMALIZATION=case_v2 FACT_RECOVERY=1
export TEMPORAL_SERIES_HISTORY=1 SAVE_RAW_RESPONSES=1 RETRY_ON_TOKEN_LIMIT=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="$PWD/out/merged-4090-continue-005-noise-001"
BATCH2="$PWD/out/full-val-005-noise-v2-001/report.json"
CURVE="$PWD/out/batch-curve-005-noise-001"
test -f "$MODEL/model.safetensors"
test -f "$BATCH2"
test ! -e "$CURVE"
mkdir -p "$CURVE"

for batch in 1 4; do
  "$HOME/mws/.venv/bin/python" -u src/benchmark.py \
    --input "$CASE_DATA_DIR/val.jsonl" --model "$MODEL" \
    --device cuda --batch-size "$batch" --warmup-batches 1 \
    --output-dir "$CURVE/cuda-batch$batch"
done

"$HOME/mws/.venv/bin/python" src/compare_runs.py \
  --reports "$BATCH2" "$CURVE/cuda-batch1/report.json" "$CURVE/cuda-batch4/report.json" \
  --baseline-pred "$CASE_DATA_DIR/baseline_pred.jsonl" \
  --gold "$CASE_DATA_DIR/val.jsonl" \
  --output-dir "$CURVE/comparison"
echo "DONE: $CURVE/comparison"
