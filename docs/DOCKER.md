> **Обновление 14.11: текущие данные и оценщик — 2.1.** [Правила, метрики и команды нового запуска](CASE_V21_MIGRATION.md).
> Числа, команды и серверные прогоны прежних этапов ниже сохранены как история v1; они не подтверждают прохождение новой базы.

# Docker: запуск HTTP-контракта

В образе нет весов модели и данных организаторов. `data/` и `out/` исключены
через `.dockerignore`; готовая merged-модель монтируется в `/models/merged`
только для чтения. Контейнер работает от пользователя `app` (UID 10001).

Статус: файлы подготовлены, но в среде проверки Docker отсутствовал. Образы
не собирались, CPU/GPU-запуск и стек vLLM здесь не проверены. См.
[VALIDATION.md](VALIDATION.md).

## 1. Подготовить модель

Docker не обучает и не сливает адаптер. Для финальной модели:

    python src/merge.py --adapter-dir out/4090-continue-002 --output out/merged-4090-continue-002
    docker run --rm -p 8000:8000 \
      -v "$PWD/out/merged-4090-continue-002:/models/merged:ro" compactfactextraction:cpu

`out/merged-4090-continue-002` должен содержать `case_contract.json`; иначе
`serve.py` откажется загружать модель как несовместимую с текущей схемой.
Рецепт и метрики — [MODEL.md](MODEL.md). Для быстрой проверки цепочки можно
собрать smoke-адаптер (`--limit 64 --max-steps 30`), но он не является
финальной моделью.

## 2. CPU

    docker compose up --build -d
    docker compose ps
    curl -fsS http://127.0.0.1:8000/health

Первый запуск собирает образ и загружает модель; `/health` отвечает 200 только
после загрузки. Логи:

    docker compose logs -f extractor

Проверка запроса (эталонные `facts` из примера удаляются и не отправляются):

    python -c "import json; r=json.loads(open('data/sample_case.jsonl').readline()); r.pop('facts', None); print(json.dumps(r, ensure_ascii=False))" > /tmp/call.json
    curl -fsS -X POST http://127.0.0.1:8000/extract \
      -H 'Content-Type: application/json' --data-binary @/tmp/call.json

Альтернатива без compose:

    docker build --target extractor -t compactfactextraction:cpu .
    docker run --rm -p 8000:8000 \
      -v "$PWD/out/merged-4090-continue-002:/models/merged:ro" compactfactextraction:cpu

## 3. NVIDIA GPU

Требуются драйвер NVIDIA и NVIDIA Container Toolkit. Проверка на хосте:

    nvidia-smi
    docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

Сборка с CUDA-сборкой PyTorch и запуск:

    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
    docker compose exec extractor python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

По умолчанию используется индекс `cu126`. Он должен соответствовать драйверу;
другой индекс задаётся при сборке, например для более старого драйвера:

    TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
      docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d

Список доступных индексов: https://pytorch.org/get-started/locally/ .
`serve.py` выбирает `cuda`, если PyTorch её видит, и загружает merged-модель в
float16. Это не заменяет измерение памяти и скорости на RTX 4060.

## 4. Необязательный vLLM

В отдельном compose-файле vLLM обслуживает merged-модель, а маленький API-образ
(`serve_vllm.py`) пересылает запросы по тому же HTTP-контракту:

    docker compose -f docker-compose.vllm.yml up --build -d

Стек требует GPU. Образ `vllm/vllm-openai` большой, тег по умолчанию `latest`
не закреплён; для воспроизводимости задайте `VLLM_IMAGE` с конкретным тегом или
дайджестом. `--max-model-len` по умолчанию 16384; он должен покрывать бюджет
входа и ответа из `src/config.py`. Реальный vLLM в этом репозитории не проверен.

## 5. Переменные

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `MODEL_PATH` | `./out/merged-4090-continue-002` | каталог merged-модели на хосте, монтируется только для чтения |
| `PORT` | `8000` | порт API на хосте |
| `MAX_BATCH` | `1` | размер пачки для backend |
| `WITH_SCHEMA` | `0` | `1` добавляет JSON-схему в prompt; не включать для модели, обученной без неё |
| `JSON_CONSTRAINED` | `1` | `0` отключает XGrammar-ограничение только для A/B; в vLLM-режиме схема задаётся через `response_format` |
| `JSON_GRAMMAR_BACKEND` | `auto` | масочное ядро XGrammar; `torch_native` — если Triton JIT не собирается |
| `TAG` | `cpu`/`cuda`/`vllm-api` | тег локального образа |
| `TORCH_INDEX_URL` | CPU или `cu126` | индекс PyTorch при сборке |
| `VLLM_IMAGE` | `vllm/vllm-openai:latest` | образ vLLM |
| `VLLM_MAX_MODEL_LEN` | `16384` | контекст vLLM |

Остановка:

    docker compose down

## 6. Что важно помнить

- Веса и данные не попадают в Git, образ и build context. Mount модели —
  единственный источник весов.
- API не принимает эталонные `facts`: лишние поля отклоняются с HTTP 422.
- Ошибки backend возвращаются как 502/503, а не как пустой успешный ответ.
- Для `docker compose` нужен Compose v2; проверка `create_host_path: false`
  требует сравнительно свежую версию. Если каталога модели нет, запуск должен
  завершиться ошибкой, а не создать пустой каталог.
- Каталог модели должен быть читаем UID 10001; обычно достаточно прав `755`
  на каталоги и `644` на файлы.
- Версии внутри собранного образа фиксируйте отчётом:

      docker compose exec extractor python src/doctor.py --json /tmp/doctor.json

  и сохраняйте вывод вместе с тегом/дайджестом образа.
- Сборка на другой машине не доказывает совместимость с RTX 4060: нужны
  отдельные замеры памяти, времени и качества.
