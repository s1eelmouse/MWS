#!/usr/bin/env bash
# Continue dependency installation in the existing V100 environment; no model transfer.
set -euo pipefail
umask 077
ROOT="${1:-$HOME/team_shpana}"
cd "$ROOT"
ROOT="$(pwd)"
HERE="$ROOT/docs/migrations/4090-to-v100"
VENV="${MWS_VENV:-$ROOT/.venv-v100}"
PY="$VENV/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Existing Python environment not found: $VENV"; exit 1
fi
mkdir -p "$ROOT/out"
OUT="$ROOT/out/v100-install-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir "$OUT"
exec > >(tee "$OUT/setup.log") 2>&1
trap 'echo "INSTALL STOPPED. Evidence: $OUT"' ERR
cp "${BASH_SOURCE[0]}" "$OUT/resume-install.sh"
echo "Resuming installation in $VENV; evidence: $OUT"
uv --version
"$PY" --version
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 JSON_GRAMMAR_THREADS=4
export JSON_GRAMMAR_BACKEND=torch_native
"$PY" - <<'PY'
from pathlib import Path
import hashlib
import json
import sys
assert sys.version_info[:2] == (3, 12), 'Expected existing Python 3.12 environment'
root = Path.cwd()
manifest = json.loads((root / 'CODE_TRANSFER_MANIFEST.json').read_text())
for name, expected in manifest['files'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, f'Code checksum mismatch: {name}'
print('Existing transferred code checksums verified')
PY
# uv's torch backend routes only PyTorch packages to its dedicated index.
# Generic packages (including requests) are resolved from PyPI.
uv pip install --python "$PY" -r "$ROOT/requirements.txt" -c "$HERE/constraints-v100.txt" \
  --default-index https://pypi.org/simple --torch-backend cu126
uv pip check --python "$PY"
uv pip freeze --python "$PY" > "$OUT/requirements.freeze.txt"
"$PY" - <<'PY'
import torch
assert torch.__version__ == '2.14.1+cu126', f'Unexpected PyTorch build: {torch.__version__}'
assert torch.version.cuda == '12.6', f'Unexpected CUDA build: {torch.version.cuda}'
print('Verified PyTorch build:', torch.__version__, 'CUDA:', torch.version.cuda)
PY
"$PY" -u "$HERE/probe-v100.py" --json "$OUT/compatibility.json"
"$PY" -u "$ROOT/src/doctor.py" --json "$OUT/doctor.json"
"$PY" -m unittest discover -s "$ROOT/tests" -v > "$OUT/tests.log" 2>&1
"$PY" -m compileall -q "$ROOT/src" "$ROOT/tests"
echo "INSTALL AND TESTS COMPLETE. Environment: $VENV; evidence: $OUT"
