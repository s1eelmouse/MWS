#!/usr/bin/env bash
# GPU A/B for NOISE_ROBUST_RECOVERY on the selected 005 profile (docs/NOISE_ROBUST_RECOVERY.md).
#
#   MODEL=out/merged-4090-continue-005-noise-001 bash scripts/run_noise_robust_ab.sh
#
# Runs, in priority order (each ~45 min on RTX 4090; stop any time, the table is
# printed after every run):
#   B  noisy +0.15, new rules ON      <- candidate
#   A  noisy +0.15, new rules OFF     <- control, same generation settings
#   K  clean val,   new rules ON      <- must not regress on clean
#   C  noisy +0.30, new rules ON      <- stress test for a noisier hidden set
# RUNS="B A" limits the list. Nothing existing is overwritten.
set -euo pipefail
REPO="${REPO:-$PWD}"
cd "$REPO"
PY="${PY:-$REPO/.venv/bin/python}"
: "${MODEL:?set MODEL to the merged 005 model directory}"
export CASE_DATA_DIR="${CASE_DATA_DIR:-$REPO/data/organizer-v2.1}"
RUNS="${RUNS:-B A K C}"
OUT="${OUT:-$REPO/out/noise-robust-ab-$(date +%Y%m%d-%H%M%S)}"

# Selected 005 profile (docs/UNIFIED_RULES_005.md) + run settings of run_eval_005.sh
export SCORER_VERSION=2.1 ALLOW_LEGACY_CONTRACT=1
export CATALOG_PATH="$CASE_DATA_DIR/catalog.json" TEMPORAL_CATALOG_PATH="$CASE_DATA_DIR/catalog.json"
export TEMPORAL_NORMALIZATION=case_v2 TEMPORAL_SERIES_HISTORY=1 FACT_RECOVERY=1 TARIFF_BOUNDARY_POLICY=explicit
export JSON_CONSTRAINED=1 JSON_GRAMMAR_BACKEND=torch_native JSON_GRAMMAR_THREADS="${THREADS:-8}"
export MAX_INPUT_TOKENS=12288 MAX_SEQUENCE_TOKENS=16384 MAX_NEW_TOKENS=4096 PROMPT_LOOKUP_TOKENS=0
export SAVE_RAW_RESPONSES=1 RETRY_ON_TOKEN_LIMIT=1
export OMP_NUM_THREADS="${THREADS:-8}" MKL_NUM_THREADS="${THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test -f "$CASE_DATA_DIR/val.jsonl" && test -f "$CASE_DATA_DIR/catalog.json"
test -d "$MODEL"
test ! -e "$OUT"
mkdir -p "$OUT/data"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv | tee "$OUT/gpu.txt"
for LEVEL in 0.15 0.30; do
  "$PY" src/make_noisy_val.py --input "$CASE_DATA_DIR/val.jsonl" \
    --output "$OUT/data/noisy-$LEVEL/val.jsonl" --level "$LEVEL" --seed 42 > "$OUT/data/noisy-$LEVEL.manifest.json"
done

summary() {
  "$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
names = {"B": "noisy+0.15 ON ", "A": "noisy+0.15 OFF", "K": "clean      ON ", "C": "noisy+0.30 ON "}
print("\nrun             |   T    |   S    |   P    |   R    |   G    | fail | tariff S | ticket S | service S | verdict")
for run in "BAKC":
    path = out / f"run-{run}" / "report.json"
    if not path.is_file():
        continue
    report = json.loads(path.read_text(encoding="utf-8"))
    q = report["quality"]; o = q["official"]; pa = q["per_attribute"]
    ok = "PASS" if o["temporal_recall"] >= 0.4506 and o["state_accuracy"] >= 0.5145 else "below target"
    print(f"{names[run]}  | {o['temporal_recall']:.4f} | {o['state_accuracy']:.4f} | {o['fact_precision']:.4f} | "
          f"{o['fact_recall']:.4f} | {o['grounding_turn']:.4f} | {report['failed_calls']:4d} | "
          f"{pa['tariff']['state_accuracy']:.4f}   | {pa['ticket_status']['state_accuracy']:.4f}   | "
          f"{pa['service_state']['state_accuracy']:.4f}    | {ok}")
print("targets: T >= 0.4506, S >= 0.5145 (623 states: +0.0016 per correct state)")
PY
}

for RUN in $RUNS; do
  case "$RUN" in
    B) INPUT="$OUT/data/noisy-0.15/val.jsonl"; FLAG=1 ;;
    A) INPUT="$OUT/data/noisy-0.15/val.jsonl"; FLAG=0 ;;
    K) INPUT="$CASE_DATA_DIR/val.jsonl";       FLAG=1 ;;
    C) INPUT="$OUT/data/noisy-0.30/val.jsonl"; FLAG=1 ;;
    *) echo "unknown run $RUN" >&2; exit 2 ;;
  esac
  echo "=== run $RUN: $INPUT NOISE_ROBUST_RECOVERY=$FLAG  $(date +%H:%M:%S)"
  NOISE_ROBUST_RECOVERY=$FLAG "$PY" -u src/benchmark.py --input "$INPUT" --model "$MODEL" \
    --device cuda --batch-size 2 --warmup-batches 1 --output-dir "$OUT/run-$RUN" 2>&1 | tee "$OUT/run-$RUN.log"
  summary | tee "$OUT/summary.txt"
done
echo "DONE: $OUT"
