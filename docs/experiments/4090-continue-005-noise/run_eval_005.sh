#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/mws"
export CASE_DATA_DIR="$PWD/data/organizer_v2_1"
export SCORER_VERSION=2.1
export ALLOW_LEGACY_CONTRACT=1
export CATALOG_PATH="$CASE_DATA_DIR/catalog.json"
export TEMPORAL_CATALOG_PATH="$CASE_DATA_DIR/catalog.json"
export MAX_INPUT_TOKENS=12288
export MAX_SEQUENCE_TOKENS=16384
export MAX_NEW_TOKENS=4096
export PROMPT_LOOKUP_TOKENS=0
export JSON_CONSTRAINED=1
export JSON_GRAMMAR_BACKEND=torch_native
export JSON_GRAMMAR_THREADS=8
export TEMPORAL_NORMALIZATION=case_v2
export FACT_RECOVERY=1
export TEMPORAL_SERIES_HISTORY=1
export SAVE_RAW_RESPONSES=1
export RETRY_ON_TOKEN_LIMIT=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="$PWD/out/merged-4090-continue-005-noise-001"
CLEAN="$PWD/out/full-val-005-noise-v2-001"
NOISY="$PWD/out/noisy-val-005-noise-v2-001"
test -f "$MODEL/model.safetensors"
test ! -e "$CLEAN"
test ! -e "$NOISY"
nvidia-smi
sha256sum "$MODEL/model.safetensors"
"$HOME/mws/.venv/bin/python" -u src/benchmark.py \
  --input "$CASE_DATA_DIR/val.jsonl" --model "$MODEL" \
  --device cuda --batch-size 2 --warmup-batches 1 \
  --output-dir "$CLEAN"
"$HOME/mws/.venv/bin/python" src/compare_temporal.py \
  --gold "$CASE_DATA_DIR/val.jsonl" --scorer-version 2.1 \
  --catalog "$CASE_DATA_DIR/catalog.json" \
  --before "$CLEAN/predictions-before-series.jsonl" \
  --after "$CLEAN/predictions.jsonl" \
  --output-dir "$CLEAN/history-date-comparison"
"$HOME/mws/.venv/bin/python" -u src/benchmark.py \
  --input out/offline-preflight-v21/val-noisy-015.jsonl --model "$MODEL" \
  --device cuda --batch-size 2 --warmup-batches 1 \
  --output-dir "$NOISY"
"$HOME/mws/.venv/bin/python" - <<'PY'
import json
for name, path in [('clean', 'out/full-val-005-noise-v2-001/report.json'),
                   ('noisy', 'out/noisy-val-005-noise-v2-001/report.json')]:
    r = json.load(open(path))
    print(name, json.dumps(r['quality']['official'], ensure_ascii=False))
    print(name, 'failed', r['failed_calls'], 'format', r['quality']['diagnostics']['format_valid_rate'])
PY
echo "DONE: 005 clean + noisy val"
