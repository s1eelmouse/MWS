# CompactFactExtraction: рабочий контур кейса

Статус: адаптированный код и CPU-проверки. Настоящее обучение на RTX 4060,
качество дообученной модели и ресурсные показатели GPU ещё не измерены.

Основной вход: call_id, started_at и turns с idx/speaker/text.
Выход: call_id и список facts по схеме организаторов. При ошибке модели API
возвращает ошибочный HTTP-статус, а файловый запуск сохраняет явное поле error.

## 1. Проверки без видеокарты

Все команды выполняются из корня проекта. На Windows можно обращаться к
интерпретатору напрямую, не меняя политику выполнения PowerShell:

    py -3.12 -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
    .venv\Scripts\python.exe -m unittest discover -s tests
    .venv\Scripts\python.exe src\doctor.py --json out\doctor-first.json

Далее python означает интерпретатор этого окружения. Если оно не активировано,
заменяйте python на .venv\Scripts\python.exe в Windows или .venv/bin/python в POSIX.

На macOS/Linux:

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements-cpu.txt
    .venv/bin/python -m unittest discover -s tests

Тесты API используют подставной backend и проверяют настоящий HTTP-контракт.
Дополнительные ML-тесты используют крошечную случайную модель на CPU, если
установлены torch/transformers/peft/accelerate. Они не измеряют качество извлечения.

## 2. Подготовить пакет организаторов

Укажите папку, в которой лежат выданные train.jsonl, val.jsonl,
baseline_pred.jsonl, baseline_metrics.json, catalog.json, schema.md,
case_metrics.py и README.md:

    python src/prepare_case.py --source-dir "ПУТЬ_К_ПАКЕТУ"

Файлы копируются без изменения байтов в data/organizer. train и val не
перемешиваются. Источники остаются нетронутыми. При несовпадении существующего
файла подготовка останавливается вместо перезаписи. Хеши и результаты проверки:
data/organizer/package_manifest.json. Эта папка исключена из Git.

    python src/check_data.py data/organizer/train.jsonl
    python src/train.py --check-data

Для официального набора action/agreement часто привязаны к клиенту.
Значения нормализованные, а не обязательно дословные цитаты. Небуквальные
gold-цитаты отмечаются предупреждением, поскольку ASR изменил исходный текст.
Допустимый в данных type=other поддерживается явно.

## 3. Воспроизвести базовую оценку

    python src/eval.py --gold data/organizer/val.jsonl --pred data/organizer/baseline_pred.jsonl --reference-metrics data/organizer/baseline_metrics.json --json out/baseline-first.json --errors out/baseline-first-errors.jsonl

Файлы результатов создаются эксклюзивно: повторный запуск требует нового имени.
Сопоставление выполняется по call_id, не по порядку строк. Дубликаты и неизвестные
ID отклоняются; отсутствующие предсказания учитываются как пропуски.

Официальный скрипт сохранён без исправлений в src/official_case_metrics.py.
Диагностика схемы выводится отдельно. В частности, обратные интервалы дат в
базовых предсказаниях отмечаются как нарушение контракта, но не выбрасываются из
официального расчёта: иначе изменилась бы база сравнения.

Известные ограничения официальной оценки:
- state_accuracy смешивает объекты; val против самого себя даёт 0.8957.
- Проверка названий объектов может принимать неверные объекты.
- temporal_recall и grounding условны по уже сопоставленным фактам.
- Времена сравниваются по дням.
- Опубликованные baseline_metrics не совпадают с пересчётом приложенным скриптом.

## 4. Короткий запуск на ПК с NVIDIA

Сначала установите CUDA-сборку PyTorch, совместимую с драйвером, по официальному
селектору https://pytorch.org/get-started/locally/ . Если uv уже установлен,
можно использовать uv pip install torch --torch-backend=auto.

    python -m pip install -r requirements.txt
    python src/doctor.py --json out/doctor-gpu-first.json

В отчёте должно быть cuda_available=true. Модель по умолчанию:
Qwen/Qwen3-0.6B. Это старт для технической проверки, не обещание итогового качества.
Общее число параметров проверяется перед обучением относительно 1.5B.

    python src/train.py --limit 64 --max-steps 30 --batch-size 1 --output-dir out/smoke-001

Обучение сохраняет checkpoint каждые 10 шагов, оставляет два последних,
записывает модель, seed, хеши данных и схемы в experiment.json. В конце сохраняет
адаптер, токенизатор и отдельный отчёт времени и памяти. Существующий непустой
каталог без явного продолжения не перезаписывается.

Продолжение прерванного запуска с теми же данными и параметрами:

    python src/train.py --limit 64 --max-steps 30 --batch-size 1 --output-dir out/smoke-001 --resume-from-checkpoint out/smoke-001/checkpoints/checkpoint-20

Длинные примеры НЕ обрезаются. Если вход или полный обучающий пример не
помещается в выбранный бюджет токенов, скрипт сообщает call_id и длины и
останавливается. В этом первом этапе автоматические окна длинных диалогов
ещё не реализованы. Не увеличивайте лимит без проверки видеопамяти.

Для полного эксперимента выбирайте новую папку и убирайте --limit/--max-steps.
На 8 ГБ сначала оцените скорость и память короткого запуска.

## 5. Слияние, предсказания и HTTP

После удачного обучения:

    python src/merge.py --adapter-dir out/smoke-001 --output out/merged-smoke-001
    python src/predict.py --input data/organizer/val.jsonl --limit 20 --model out/merged-smoke-001 --output out/pred-smoke-001.jsonl
    python src/eval.py --gold data/organizer/val.jsonl --limit 20 --pred out/pred-smoke-001.jsonl --json out/eval-smoke-001.json

Это оценка короткого технического запуска, а не итоговая проверка качества.
Предсказания сохраняются после каждой пачки; готовый файл не перезаписывается.
Прерванный файл содержит только завершённые вызовы, что явно видно по call_id.

MODEL_DIR задаётся переменной окружения. PowerShell:

    $env:MODEL_DIR = "out/merged-smoke-001"
    python -m uvicorn serve:app --app-dir src --host 127.0.0.1 --port 8000

В POSIX-оболочке:

    MODEL_DIR=out/merged-smoke-001 python -m uvicorn serve:app --app-dir src --host 127.0.0.1 --port 8000

Документация: http://127.0.0.1:8000/docs .
POST /extract принимает только вход звонка, без эталонного facts.
POST /extract_batch принимает объект calls со списком звонков.
GET /health отражает готовность загруженной модели.
Пример без внешнего запроса: data/sample_case.jsonl, из которого нужно убрать facts.

    python src/eval.py --gold data/organizer/val.jsonl --limit 20 --url http://127.0.0.1:8000 --json out/http-smoke-001.json

Для исходной модели без обучения используйте --model Qwen/Qwen3-0.6B --with-schema.
Код serve_vllm.py поддерживает тот же контракт, но реальный vLLM здесь не проверен.

## 6. Docker

Образ предназначен только для инференса: обучение и merge выполняются на
хосте. Веса и данные организаторов в образ не копируются: `.dockerignore`
исключает `data/` и `out/`, а merged-модель монтируется в `/models/merged`
только для чтения. Сначала соберите её командой из раздела 5, затем:

    docker compose up --build -d
    curl -fsS http://127.0.0.1:8000/health

По умолчанию compose ищет `out/merged`; другой каталог задаётся `MODEL_PATH`.
CPU-образ ставит CPU-сборку PyTorch. Для NVIDIA нужны драйвер и NVIDIA
Container Toolkit:

    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d

`TORCH_INDEX_URL` выбирает CUDA-сборку PyTorch под драйвер. Необязательный
стек vLLM:

    docker compose -f docker-compose.vllm.yml up --build -d

Контейнер работает от непривилегированного пользователя, проверка `/health`
встроена. Переменные, ограничения и команды: [docs/DOCKER.md](docs/DOCKER.md).
В среде подготовки Docker отсутствовал, поэтому образы здесь не собирались.

## 7. Что ещё предстоит измерить и реализовать

- Реальный CUDA/QLoRA запуск, память и время на RTX 4060.
- Полное обучение и оценка на исходной валидации.
- Отдельная детерминированная обработка дат и переходов состояний.
- Окна для диалогов, не помещающихся в бюджет токенов.
- Измерения CPU/GPU и минимум три режима ресурсы/качество.
- Сборка и проверка Docker-образов на CPU и GPU; финальная демонстрация.

Текущий prompt просит модель извлекать абсолютные даты. Это промежуточное
решение; требование вынести нормализацию дат из модели ещё не закрыто.

## 8. Сохранность и история

См. docs/SAFE_CHANGES.md. Прежняя версия README сохранена как README_2.md.
Legacy-схема, учебные примеры, teacher-скрипт, старые метрики и документация
удалены из репозитория. Для размеченного пакета организаторов повторная
разметка учителем не требуется.
