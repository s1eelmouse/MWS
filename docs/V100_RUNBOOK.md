# V100: проверка нового стенда

Для сокомандника, у которого есть доступ к V100. Наша задача — проверить, что
код, QLoRA/4-bit и XGrammar работают на Volta (sm_70), и получить независимый
прогон val 2.1. Числа V100 — отдельный стенд, с RTX 4090 их не сравнивать.

## 1. Что перенести

Нужны:

- код из `main` (после последнего коммита);
- `data/organizer_v2_1/` (train/val/catalog, ~34 МБ);
- одна merged-модель:
  - `out/merged-4090-continue-003` — основной вариант в готовом bundle; на
    сохранённых предсказаниях с recovery даёт T 0.4663, S 0.5152;
  - `out/merged-4090-continue-004-v21` — если 004 удалось забрать с 4090;
  - `out/merged-4090-continue-002` — старый fallback.

Если есть доступ с V100 к 4090:

```bash
rsync -az ubuntu@<4090-host>:/home/ubuntu/mws-v21/ ~/mws-v21/
# внутри уже есть data/organizer_v2_1 и out/merged-...
```

Иначе использовать готовый `v100-bundle-003.tar.gz` (код + data 2.1 +
merged-003; 1019 МБ, SHA256
`bf62eefcd9265dea47ad3ebc2476f7ef2194742f200c5642edd29b4a2cf85728`,
локально `/home/user/hackathon/artifacts-4090/`) и распаковать:

```bash
mkdir -p ~/mws-v21 && tar xzf v100-bundle-003.tar.gz -C ~/mws-v21
cd ~/mws-v21
```

Bundle содержит gold train/val; передавать только внутри команды, не публиковать.

## 2. Окружение

```bash
nvidia-smi
python3 --version
df -h .
```

Нужен Python 3.10–3.12, CUDA-драйвер, поддерживающий колёса PyTorch. Для V100
подходит CUDA 12.1 (sm_70):

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r requirements.txt
```

Если драйвер старее — заменить `cu121` на `cu118`. Если интернета нет, взять
готовое окружение с 4090 нельзя (разные CUDA/архитектуры); нужен локальный wheel
cache или образ.

Проверка:

```bash
.venv/bin/python src/doctor.py --json v100-doctor.json
cat v100-doctor.json
```

Ожидаемо: `compute_capability: 7.0`, `bf16_supported: false`, VRAM 16 или 32 ГиБ.

## 3. Данные и короткий QLoRA

```bash
.venv/bin/python src/train.py --check-data \
  --train data/organizer_v2_1/train.jsonl \
  --val data/organizer_v2_1/val.jsonl
```

Короткий smoke (30 шагов, 4-bit):

```bash
MAX_INPUT_TOKENS=12288 MAX_SEQUENCE_TOKENS=16384 MAX_NEW_TOKENS=4096 \
.venv/bin/python -u src/train.py \
  --train data/organizer_v2_1/train.jsonl \
  --val data/organizer_v2_1/val.jsonl \
  --limit 64 --max-steps 30 --batch-size 1 --seed 42 \
  --output-dir out/v100-smoke-001
```

Если bitsandbytes/4-bit падает на Volta, повторить с `--no-4bit` (0,6B LoRA в
FP16 должен поместиться в 16 ГиБ, но будет медленнее). Не запускать полный цикл,
пока smoke не завершился.

## 4. Preview 20 звонков

```bash
export MAX_INPUT_TOKENS=12288 MAX_NEW_TOKENS=4096 MAX_SEQUENCE_TOKENS=16384
export JSON_CONSTRAINED=1 JSON_GRAMMAR_BACKEND=torch_native
export TEMPORAL_NORMALIZATION=task FACT_RECOVERY=1
export CATALOG_PATH=data/organizer_v2_1/catalog.json
.venv/bin/python -u src/benchmark.py \
  --input data/organizer_v2_1/val.jsonl --limit 20 \
  --model out/merged-4090-continue-003 \
  --device cuda --batch-size 2 \
  --output-dir out/v100-preview-001
```

Проверить: `failed_calls = 0`, `format_valid_rate = 1.0`,
`temporal_normalization = task`. Модель по умолчанию — 003 из bundle. Если есть
004, заменить `--model` на `out/merged-4090-continue-004-v21`.

## 5. Полный val, если preview живой

```bash
.venv/bin/python -u src/benchmark.py \
  --input data/organizer_v2_1/val.jsonl \
  --model out/merged-4090-continue-003 \
  --device cuda --batch-size 2 \
  --output-dir out/v100-full-001
```

Шумный val — только после чистого:

```bash
.venv/bin/python src/make_noisy_val.py \
  --input data/organizer_v2_1/val.jsonl \
  --output data/organizer_v2_1/val-noisy-015.jsonl --level 0.15 --seed 42
.venv/bin/python -u src/benchmark.py \
  --input data/organizer_v2_1/val-noisy-015.jsonl \
  --model out/merged-4090-continue-003 \
  --device cuda --batch-size 2 \
  --output-dir out/v100-noisy-001
```

## 6. Что прислать обратно

Минимально:

- `v100-doctor.json` и `nvidia-smi`;
- `out/v100-smoke-001.log` + `experiment.json`/`status.json`;
- `out/v100-preview-001/report.json` и `metrics.json`;
- `pip freeze` (файл).

Если получился полный val — `report.json`, `metrics.json`, `predictions.jsonl`.

Собрать всё в один архив:

```bash
.venv/bin/pip freeze > v100-pip-freeze.txt
tar czf v100-results.tar.gz --ignore-failed-read v100-doctor.json v100-pip-freeze.txt \
  out/v100-smoke-001.log out/v100-smoke-001/experiment.json \
  out/v100-preview-001/report.json out/v100-preview-001/metrics.json \
  out/v100-full-001/report.json out/v100-full-001/metrics.json
```

Прислать `v100-results.tar.gz` и `sha256sum v100-results.tar.gz`.
Не нужно присылать веса и большие логи. Достаточно отчётов и хешей.

## 7. Ограничения

- V100 не поддерживает BF16; код уже выбирает FP16.
- Triton/XGrammar на sm_70 может не собраться — используем `torch_native`.
- vLLM/FlashAttention на V100 не гарантированы; сервис — Transformers.
- 1.7B не тренировать без письменного разрешения и проверки лимита 1,5B.
- Не смешивать скорость V100 с 4090; качество на одном и том же val должно совпасть.
- 4090 дважды уходил в ребут при работе с Docker; не запускать Docker-сборку на
  нём до сдачи. Docker уже собран и проверен на `/health`/коротком `/extract`;
  для повторной проверки использовать V100 или отдельную машину.
