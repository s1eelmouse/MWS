# Обучение и замеры по пункту 6 PLAN.md

Ветка добавляет сбор результатов. Реальные CPU/GPU-замеры и Docker-прогон на сервере
не считаются выполненными до получения отчётов. Официальный оценщик не изменяется.
Куда переносить числа из файлов прогона для отчёта — [RESULTS.md](RESULTS.md).

## Что собирается

`src/run_experiment.py` запускает последовательно программные тесты, compileall,
проверку данных, обучение, merge, три режима инференса и сравнение с базой.
Ошибка этапа останавливает цепочку. Старые каталоги не перезаписываются.
Каждый режим запускается в новом процессе, чтобы модель предыдущего режима не
оставалась в памяти. По умолчанию: CUDA batch=1, CUDA batch=4, CPU batch=1.
CPU использует Transformers float32; ONNX/CTranslate2 пока не реализованы.

```
out/experiment-001/
  environment.json, settings.json     окружение, версия/хеши кода, настройки
  stages.jsonl, status.json           команды, коды завершения, итоговый статус
  tests.log, compile.log, train.log   вывод этапов
  adapter/                           адаптер и два последних checkpoint
    attempt-*/                       отдельный журнал каждого train/resume
      events.jsonl                   шаг, эпоха, loss/eval_loss, RSS и CUDA память
      loss-history.json              история Trainer до конца попытки
      environment.json, settings.json, status.json
  merged/                            модель после слияния
  benchmark/{cuda-batch1,cuda-batch4,cpu-batch1}/
    environment.json, report.json    оборудование, конфигурация, сводные замеры
    predictions.jsonl                ответы и явные ошибки с call_id
    calls.jsonl                      токены, время завершения, ошибка каждого звонка
    batches.jsonl                    фактический размер и время каждой пачки
    memory.csv                       выборки RSS процесса каждые 50 мс
    metrics.json, errors.jsonl        официальное качество и диагностика
  comparison/
    baseline.json                    пересчитанная база на том же срезе val
    comparison.csv, comparison.json  общая таблица и отклонения от базы
    per_attribute.csv               пять типов фактов, база и каждый режим
    resources-quality.svg            экспортируемый график времени × качества
```

`loss` логируется с частотой Trainer (обычно каждые 10 шагов), `eval_loss` — после
эпохи. Официальное качество считается после самостоятельной генерации, а не из
loss. В `events.jsonl` нет текстов диалогов, ответов или эталонов.

## Единая методика

- Один исходный val и один порядок call_id; файлы и выбранные ID закреплены SHA256.
  Для другого токенизатора входные длины могут отличаться — записываются реальные.
- Один прогревочный batch, такой же размер, как у измеряемых пачек. Прогрев и загрузка
  модели измеряются отдельно, исключены из основной скорости. Некорректные ответы
  при прогреве сохраняются в `warmup_errors` и не мешают оценивать слабую модель.
  Авария выполнения прогрева, например OOM, останавливает запуск и создаёт failed report.
- Измеряется локальная обработка: подготовка промпта/токенизация, генерация, разбор
  и проверка ответа, с CUDA synchronize. Это не HTTP-задержка: сеть, очередь,
  сохранение файлов и расчёт качества не входят в измеряемый интервал.
- `completion_latency_seconds` — время от подачи всей пачки до готовности её
  результатов. Для batch>1 у звонков пачки одинаковое время; оно НЕ делится на batch.
  p50/p95 используют метод ближайшего ранга. График использует среднюю стоимость
  `measurement_seconds / calls`, отличную от задержки запроса при batch>1.
- `calls_per_hour` включает все попытки, в том числе ошибки; отдельно сохраняется
  `successful_calls_per_hour`. Ошибки не удаляются из качества. Поля
  `format_valid_rate` и `failed_calls` не позволяют выдать быстрые отказы за успех.
- `gpu_seconds_per_1000_calls = measurement_seconds * 1000 / calls` для одной
  выделенной GPU. Это время занятой карты, НЕ сумма времени CUDA-ядер. Для CPU —
  null, а не ноль. Измерять при отсутствии посторонних задач на выбранной GPU.
- GPU память: пики allocator allocated/reserved PyTorch после прогрева, включая
  уже загруженную модель. Это память данного процесса, не память всех процессов
  драйвера. CPU память: абсолютный максимум RSS процесса по выборкам 50 мс;
  короткий пик между выборками возможен. Пики загрузки модели не входят в это окно.
- `input_tokens` — фактический полный промпт без padding; `output_tokens` — токены
  генерации до первого EOS включительно, без padding. Неизвестные длины явно null.
  Достижение лимита ответа без EOS — `output_token_limit`, даже если частичный
  ответ похож на JSON. Диалоги не обрезаются.
- В benchmark запрещено автоматическое уменьшение пачки после OOM. Отказ
  фиксируется для всей пачки. Обычный сервис сохраняет прежний механизм деления.
- Сравнение требует >=3 различных конфигураций, одинаковых данных, среза, схемы,
  лимитов, прогрева и версии оценщика. Ошибки генерации остаются в сравнении;
  аварийно прерванные прогоны исключаются с явной ошибкой.
- Базовые предсказания организаторов дают качество, но не время их получения.
  GPU-секунды и ускорение относительно этой базы без отдельного замера неизвестны.
  На графике база — горизонтальная пунктирная линия, не точка с выдуманной скоростью.

Сводка CSV включает число параметров, размер локальных файлов весов
(`model_weights_bytes`, сумма `.safetensors` / `.bin` без конфигурации),
время загрузки, p50/p95 токенов, скорость успешных звонков и счётчики сбоев.
Для удалённой модели без локальных файлов размер неизвестен. В старом отчёте
размер можно восстановить из `model_artifacts`; недостающие параметры и счётчики
остаются пустыми. В `report.json` поле `error_counts` считает ошибки по звонкам,
исключая прогрев; нулевые стандартные типы и дополнительные типы сохраняются.
CSV содержит стандартные `errors_*` и полный JSON в колонке `error_counts`.
Отклонения от базы — модель минус база для всех пяти официальных метрик и
валидности схемы; null не заменяется нулём.
`per_attribute.csv` содержит строки для базы и каждого режима, тип атрибута
(временной/статический), количества фактов, precision/recall и их отклонения, а также
типы ошибок из `per_attribute` оценщика: `matched_facts`, `missed_facts`, `extra_facts`,
`valid_from_errors`, `valid_to_errors`, `date_errors`, `source_turn_errors` и
`temporal_recall` / `grounding_turn` по типу. Ошибки дат и источников считаются среди
совпавших фактов официальным сопоставлением, даты по дню, поэтому в сумме они
воспроизводят официальные `temporal_recall` и `grounding_turn`.
Существующие отчёты можно заново обработать `compare_runs.py` в новый каталог:
повторять обучение для расширенного экспорта не требуется. Счётчики типов ошибок
появятся только в отчётах, снятых после этого изменения оценщика.

### Устойчивость к шуму

`src/make_noisy_val.py` делает детерминированную копию val с дополнительным ASR-шумом
вместо скрытого adversarial-подмножества; исходный файл не меняется:

```bash
python src/make_noisy_val.py --input data/organizer/val.jsonl --output out/val-noisy-015.jsonl --level 0.15 --seed 42
python src/run_experiment.py --output-dir out/noisy-001 --model out/merged --val out/val-noisy-015.jsonl
```

Меняются только тексты реплик; факты и `turn_idx` прежние, span и цитата обнуляются.
База на шумной копии не пересчитывается по смыслу: её предсказания сделаны по чистому
тексту, поэтому `comparison.csv` для такого прогона сравнивает с базой на чистом val.

## Запуск на Linux без Docker

Все команды из корня проекта. Установить совместимую CUDA-сборку PyTorch и
`requirements.txt`, `requirements-cpu.txt`. Рекомендуемый способ:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install --torch-backend=auto torch -r requirements.txt -r requirements-cpu.txt
python src/prepare_case.py --source-dir /path/to/organizer-package
export MAX_INPUT_TOKENS=11000
export BASE_MODEL=Qwen/Qwen3-0.6B
python src/run_experiment.py --output-dir out/experiment-001
```

Данные должны быть в `data/organizer`. Папка `case_data` исключена из Docker build
context. В `.gitignore` она тоже есть, но в `main` её файлы уже закоммичены, а Git
продолжает отслеживать закоммиченные файлы: правило защищает только от новых.
Оригиналы не перемещаются.
Длительный запуск выполнять в tmux. Логи сохраняются на диске независимо от терминала.

Короткая проверка новой среды (не итоговая оценка):

```bash
python src/run_experiment.py --output-dir out/smoke-001 \
  --train-limit 64 --max-steps 30 --eval-limit 20
```

Если модель уже обучена, повторное обучение не требуется:

```bash
python src/run_experiment.py --output-dir out/measurements-002 \
  --model /absolute/path/to/merged-model
```

Одиночный замер, например после исправления неудачного режима:

```bash
python src/benchmark.py --model out/experiment-001/merged --device cuda \
  --batch-size 4 --output-dir out/extra-gpu4-001
```

Продолжить прерванное обучение: запустить `src/train.py` напрямую с исходными
настройками и `--resume-from-checkpoint out/experiment-001/adapter/checkpoints/checkpoint-N`.
Тот же `--output-dir out/experiment-001/adapter`; создать новый файл stdout-лога.
Сохраняются все `attempt-*`, но checkpoint по-прежнему остаются только два последних.
После завершения выполнить merge в новый каталог и новый `run_experiment --model`.
Весь pipeline автоматически с середины не возобновляется.

## Docker на подготовленном NVIDIA-сервере

Используется отдельный `Dockerfile.train`, образ сервиса не изменён. Данные
монтируются read-only, `out` — writable, кеш моделей — постоянный Docker volume.
Требуются Docker Compose v2 и NVIDIA Container Toolkit. Один GPU выдаётся контейнеру.

```bash
mkdir -p out
# Предоставьте UID 10001 право записи в out (или настройте ACL для пользователя контейнера).
sudo chown 10001:10001 out
export CODE_REVISION=$(git rev-parse HEAD)
export RUN_NAME=experiment-001
# Индекс выбирается под GPU и драйвер; cu128 — пример, не гарантия совместимости.
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
docker compose -f docker-compose.train.yml build
docker compose -f docker-compose.train.yml up -d
docker compose -f docker-compose.train.yml logs -f
```

Сборка фиксирует установленные версии в `/image-packages.txt`, а каждый запуск
сохраняет версии ML-библиотек и хеши исходников. Для повторения сохраняйте ID образа:

```bash
docker image inspect compactfactextraction-train:local --format '{{.Id}}'
docker compose -f docker-compose.train.yml ps -a
```

Проверка GPU и тестов до полного запуска:

```bash
docker compose -f docker-compose.train.yml run --rm --entrypoint python experiment src/doctor.py
docker compose -f docker-compose.train.yml run --rm --entrypoint python experiment -m unittest discover -s tests -v
```

Для измерения существующей модели (она должна лежать внутри смонтированного out):

```bash
docker compose -f docker-compose.train.yml run -d --name mws-measure-002 experiment \
  --output-dir /app/out/measurements-002 --model /app/out/merged-existing
docker logs -f mws-measure-002
docker inspect mws-measure-002 --format '{{.State.ExitCode}}'
```

Остановка контейнера не является автоматическим resume. Не удаляйте постоянный диск
до копирования каталога эксперимента. Для скачивания из PowerShell:

```powershell
scp -P PORT -r USER@HOST:/workspace/MWS/out/experiment-001 C:\Users\ivanr\PycharmProjects\MWS\out\
```

## Что остаётся для сдачи

Фактические прогоны CPU/GPU, проверка Docker в чистой серверной среде, достижение
100% корректных ответов и целевого качества, интерпретация ошибок, презентация и
демо с резервным видео. Автоматическое создание таблицы не подтверждает эти пункты.
Для HTTP-задержки используйте существующий `src/eval.py --url`: его p50/p95 включают
HTTP и очередь, поэтому их нельзя смешивать с локальными числами `benchmark.py`.
