#!/usr/bin/env bash
# Запуск финальной модели 005 v2 в Docker одной командой.
#   ./scripts/start.sh [путь/к/005-v2-final-merged.tar.gz]
# Архив по умолчанию ищется в корне репозитория. Скрипт распаковывает веса в
# models/005-v2-final/merged, открывает их на чтение пользователю контейнера
# (UID 10001), сверяет SHA256, собирает образ (GPU, если есть nvidia-smi, иначе CPU)
# и ждёт /health.
set -euo pipefail
cd "$(dirname "$0")/.."

ARCHIVE="${1:-005-v2-final-merged.tar.gz}"
MODEL_ROOT="models/005-v2-final"
MODEL_DIR="$MODEL_ROOT/merged"
EXPECTED_SHA="6b4e1f2d2c48284188714ad0acb31e6a1aac89d63b660a65c8bfce0e64557617"
PORT="${PORT:-8000}"

if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
  [ -f "$ARCHIVE" ] || { echo "Нет $ARCHIVE: скачайте asset 005-v2-final-merged.tar.gz из Release final-005-v2" >&2; exit 1; }
  mkdir -p "$MODEL_ROOT"
  echo "Распаковка $ARCHIVE -> $MODEL_ROOT"
  tar -xzf "$ARCHIVE" -C "$MODEL_ROOT"
fi

# В архиве права 0600/0700: без этого UID 10001 в контейнере не прочитает веса.
chmod -R a+rX models

echo "Проверка SHA256 весов"
echo "$EXPECTED_SHA  $MODEL_DIR/model.safetensors" | sha256sum -c -

FILES=(-f docker-compose.yml)
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  FILES+=(-f docker-compose.gpu.yml); echo "Режим: GPU"
else
  echo "Режим: CPU (GPU/nvidia-smi не найдены; звонок обрабатывается ~1-4 мин)"
fi

docker compose "${FILES[@]}" up -d --build

echo -n "Ожидание /health"
for _ in $(seq 1 120); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null; then
    echo; curl -s "http://localhost:$PORT/health"; echo
    echo "Готово: POST http://localhost:$PORT/extract, Swagger: http://localhost:$PORT/docs"
    exit 0
  fi
  echo -n "."; sleep 5
done
echo; echo "Сервис не ответил за 10 минут. Логи:" >&2
docker compose "${FILES[@]}" logs --tail 50 extractor >&2
exit 1
