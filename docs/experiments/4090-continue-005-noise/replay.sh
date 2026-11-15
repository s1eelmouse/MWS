#!/usr/bin/env bash
# Replay the 005-noise cycle from the checked-in code snapshot.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$HERE/code"
PROJECT="${PROJECT_DIR:-$(cd "$HERE/../../.." && pwd)}"
cd "$PROJECT"
export CASE_DATA_DIR="${CASE_DATA_DIR:-$PROJECT/data/organizer_v2_1}"
export BASE_MODEL=Qwen/Qwen3-0.6B
export MAX_INPUT_TOKENS=12288 MAX_SEQUENCE_TOKENS=16384 MAX_NEW_TOKENS=4096
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MWS_CODE_REVISION=snapshot-4090-continue-005-noise

RUN="${RUN_ID:-4090-continue-005-noise-replay}"
mode="${1:-help}"
case "$mode" in all|train|evaluate|test) ;; *)
  echo "Usage: bash replay.sh {all|train|evaluate|test}; activate the matching .venv first."
  exit 2;; esac

mkdir -p out
log="out/$RUN.$mode.log"
if [[ -e "$log" ]]; then echo "Log exists: $log. Choose a new RUN_ID."; exit 1; fi
export PYTHONPYCACHEPREFIX="$PROJECT/out/$RUN-pycache"
exec > >(tee "$log") 2>&1

if [[ "$mode" == test ]]; then
  python -m unittest discover -s "$CODE/tests" -v
  python -m compileall -q "$CODE/src" "$CODE/tests"
  exit 0
fi

if [[ "$mode" == all || "$mode" == train ]]; then
  python -u "$CODE/src/train.py" \
    --train "$CASE_DATA_DIR/train.jsonl" \
    --val "$CASE_DATA_DIR/val.jsonl" \
    --catalog "$CASE_DATA_DIR/catalog.json" \
    --train-variant noise \
    --init-adapter-dir "${INIT_ADAPTER_DIR:-out/adapter-003}" \
    --allow-legacy-contract --epochs 1 --batch-size 1 \
    --learning-rate 1e-4 --save-steps 50 --max-sequence-tokens 16384 \
    --output-dir "out/$RUN"
fi

if [[ "$mode" == all || "$mode" == evaluate ]]; then
  python -u "$CODE/src/merge.py" --adapter-dir "out/$RUN" \
    --output "out/merged-$RUN" --allow-legacy-contract
  python -u "$CODE/src/benchmark.py" \
    --input "$CASE_DATA_DIR/val.jsonl" --model "out/merged-$RUN" \
    --device cuda --batch-size 2 --warmup-batches 1 \
    --output-dir "out/full-val-$RUN"
  python -u "$CODE/src/benchmark.py" \
    --input out/offline-preflight-v21/val-noisy-015.jsonl \
    --model "out/merged-$RUN" --device cuda --batch-size 2 --warmup-batches 1 \
    --output-dir "out/noisy-val-$RUN"
fi
