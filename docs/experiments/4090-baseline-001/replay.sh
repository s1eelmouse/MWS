#!/usr/bin/env bash
# Reconstruction for replay; this launcher itself was NOT the historical script.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$HERE/code"
PROJECT="${PROJECT_DIR:-$(cd "$HERE/../../.." && pwd)}"
cd "$PROJECT"
export BASE_MODEL=Qwen/Qwen3-0.6B
export MAX_INPUT_TOKENS=12288 MAX_SEQUENCE_TOKENS=16384 MAX_NEW_TOKENS=4096
export PROMPT_LOOKUP_TOKENS=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export MWS_CODE_REVISION=snapshot-4090-baseline-001
RUN="${RUN_ID:-4090-baseline-001-replay}"
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
  python -u "$CODE/src/train.py" --epochs 1 --batch-size 1 --seed 42 --save-steps 50 --output-dir "out/$RUN"
fi
if [[ "$mode" == all || "$mode" == evaluate ]]; then
  python -u "$CODE/src/merge.py" --adapter-dir "out/$RUN" --output "out/merged-$RUN"
  python -u "$CODE/src/predict.py" --input data/organizer/val.jsonl --model "out/merged-$RUN" --batch-size 2 --output "out/pred-$RUN.jsonl"
  python "$CODE/src/eval.py" --gold data/organizer/val.jsonl --pred "out/pred-$RUN.jsonl" --json "out/eval-$RUN.json" --errors "out/errors-$RUN.jsonl"
fi
