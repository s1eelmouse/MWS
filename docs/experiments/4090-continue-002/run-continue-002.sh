#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/mws-measurements-002
source /home/ubuntu/mws/.venv/bin/activate
export BASE_MODEL=Qwen/Qwen3-0.6B
export MAX_INPUT_TOKENS=12288 MAX_SEQUENCE_TOKENS=16384 MAX_NEW_TOKENS=4096
export PROMPT_LOOKUP_TOKENS=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export MWS_CODE_REVISION=working-tree-adapter-continuation-002
adapter=out/4090-continue-002
merged=out/merged-4090-continue-002
benchmark=out/benchmark-4090-continue-002
log=out/4090-continue-002.pipeline.log
for target in "$adapter" "$merged" "$benchmark" "$log"; do
    if [[ -e "$target" ]]; then
        echo "Output already exists: $target. Preserve it; use a new run name."
        exit 1
    fi
done
exec > >(tee "$log") 2>&1
echo "NEW cycle from baseline adapter weights; fresh optimizer/schedule; LR=1e-4; one additional epoch."
python -u src/train.py --init-adapter-dir out/4090-baseline-001 \
    --learning-rate 0.0001 --epochs 1 --batch-size 1 --seed 42 --save-steps 50 \
    --output-dir "$adapter"
python -u src/merge.py --adapter-dir "$adapter" --output "$merged"
python -u src/benchmark.py --input data/organizer/val.jsonl --model "$merged" \
    --device cuda --batch-size 2 --output-dir "$benchmark"
echo "DONE: training, merge and full-val benchmark complete. Results: $benchmark/report.json"
