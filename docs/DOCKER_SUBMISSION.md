# Docker: финальная сдача (final-005-v2)

Модель: GitHub Release `final-005-v2`, asset `005-v2-final-merged.tar.gz`
(standalone merged Qwen3-0.6B + tokenizer, адаптер не нужен).
Веса и каталог в образ не копируются — монтируются read-only.

Профиль в `docker-compose.yml` совпадает с финальным прогоном
(clean val, 383 calls, 0 failures: T 0.5158, S 0.5682, P 0.4973, R 0.5913, G 0.8002):
constrained JSON (XGrammar, torch_native, 4 threads), FP16 на CUDA, batch 4,
`TEMPORAL_NORMALIZATION=case_v2`, `FACT_RECOVERY=1`
(policy `unified-v1-guarded-tariffs-explicit`), series history,
`PROMPT_LOOKUP_TOKENS=0`, retry on token limit (rep. penalty 1.2),
каталог `case_data/catalog.json` (sha256 b7579c70…).

## 0. Одной командой

Положите `005-v2-final-merged.tar.gz` (Release `final-005-v2`) в корень репозитория и:

```bash
./scripts/start.sh            # или ./scripts/start.sh /путь/к/архиву.tar.gz
```

Скрипт распаковывает веса, открывает их на чтение пользователю контейнера,
сверяет SHA256, собирает образ (GPU при наличии `nvidia-smi`, иначе CPU) и ждёт
`/health`. Требования: Docker + Compose v2; для GPU — NVIDIA Container Toolkit,
драйвер с поддержкой CUDA 12.6.

Ниже — те же шаги вручную.

## 1. Модель

```bash
mkdir -p models/005-v2-final
tar -xzf 005-v2-final-merged.tar.gz -C models/005-v2-final
ls models/005-v2-final/merged   # config.json, model.safetensors, tokenizer*
chmod -R a+rX models          # в архиве права 0600: иначе контейнер (UID 10001) не прочитает веса
sha256sum models/005-v2-final/merged/model.safetensors
# 6b4e1f2d2c48284188714ad0acb31e6a1aac89d63b660a65c8bfce0e64557617
```

Если модель лежит в другом месте: `export MODEL_PATH=/abs/path/to/merged`.

## 2. Сборка и запуск (GPU, NVIDIA Container Toolkit)

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
docker compose logs -f extractor     # ждать "Application startup complete"
```

Драйвер старее CUDA 12.6: `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124`.
CPU-режим (только для проверки, медленно): `docker compose up -d --build`.

## 3. Проверка

```bash
curl -s localhost:8000/health
# один звонок во входном формате v2.1 (поле facts = gold, его не отправляем)
head -n1 case_data/val.jsonl | python3 -c "import json,sys;d=json.loads(sys.stdin.read());d.pop('facts',None);print(json.dumps(d,ensure_ascii=False))" > /tmp/call.json
curl -s -X POST localhost:8000/extract -H 'Content-Type: application/json' --data-binary @/tmp/call.json
```

`/extract` — один звонок, `/extract_batch` — список звонков, `/docs` — Swagger.
В `/health` поле `contract` должно содержать signature
`43666ebeed4459ce492ba224670921624bffb72f8e4a85011c71d5e08f7206ad`.

## 4. Остановка

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
```
