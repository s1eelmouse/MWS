#!/usr/bin/env bash
# Evaluation only. No training, adapter loading for training, or merge.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"
source "${MWS_VENV:-/home/ubuntu/mws/.venv}/bin/activate"
export MODEL_DIR="${MODEL_DIR:-/home/ubuntu/mws/out/merged-4090-continue-002}"
export VAL_FILE="${VAL_FILE:-/home/ubuntu/mws/data/organizer/val.jsonl}"
export CATALOG_PATH="${CATALOG_PATH:-/home/ubuntu/mws/data/organizer/catalog.json}"
export MAX_INPUT_TOKENS=12288 MAX_SEQUENCE_TOKENS=16384 MAX_NEW_TOKENS=4096
export MAX_TEXT_CHARS=0 PROMPT_LOOKUP_TOKENS=0
export JSON_CONSTRAINED=1 JSON_GRAMMAR_BACKEND=torch_native JSON_GRAMMAR_THREADS=8
export TEMPORAL_NORMALIZATION=task SAVE_RAW_RESPONSES=1
export RETRY_ON_TOKEN_LIMIT=1 RETRY_REPETITION_PENALTY=1.2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export MWS_CODE_REVISION=7b0f585-plus-full-val-003
if [[ "${1:-run}" == check ]]; then
  python -u "$HERE/preflight.py"
  exit 0
fi
RUN="${RUN_ID:-continue-002-full-val-003}"
OUT="/home/ubuntu/mws/out/$RUN"
LOG="/home/ubuntu/mws/out/$RUN.log"
if [[ -e "$OUT" || -e "$LOG" ]]; then
  echo "Output/log exists. Preserve it and choose a new RUN_ID."; exit 1
fi
exec > >(tee "$LOG") 2>&1
echo "EVALUATION ONLY; model=$MODEL_DIR; full val; XGrammar=torch_native; dates=task; catalog enabled."
python -u "$HERE/preflight.py"
python -u src/benchmark.py --input "$VAL_FILE" --model "$MODEL_DIR" \
  --device cuda --batch-size 2 --warmup-batches 1 --output-dir "$OUT"
python -u src/compare_temporal.py --gold "$VAL_FILE" \
  --before "$OUT/predictions-before-series.jsonl" --after "$OUT/predictions.jsonl" \
  --output-dir "$OUT/series-date-comparison"
cp "$HERE/run.sh" "$OUT/run.sh"
if [[ -f EVALUATION_TRANSFER_MANIFEST.json ]]; then
  cp EVALUATION_TRANSFER_MANIFEST.json "$OUT/code-transfer-manifest.json"
fi
echo "DONE: full 383-call evaluation complete; no training. Results: $OUT"
