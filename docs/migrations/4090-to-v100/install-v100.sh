#!/usr/bin/env bash
set -euo pipefail
trap 'echo "INSTALL STOPPED: inspect this log and out/v100-install-*/tests.log"' ERR
umask 077
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"
uname -a
cat /etc/os-release
df -h "$HOME"
python3 - <<'PY'
from pathlib import Path
import hashlib
import json
root = Path.cwd()
manifest = json.loads((root / 'CODE_TRANSFER_MANIFEST.json').read_text())
for name, expected in manifest['files'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, f'Code checksum mismatch: {name}'
print('Code package checksums verified')
PY
VENV="${MWS_VENV:-$ROOT/.venv-v100}"
if [[ -e "$VENV" ]]; then
  echo "Environment exists: $VENV. Preserve it; set a NEW MWS_VENV to retry."; exit 1
fi
command -v uv
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo 'nvidia-smi is not on PATH; driver access will be checked through PyTorch.'
fi
uv python install 3.12
uv venv --seed --python 3.12 "$VENV"
export MWS_VENV="$VENV"
bash "$HERE/resume-v100-install.sh" "$ROOT"
