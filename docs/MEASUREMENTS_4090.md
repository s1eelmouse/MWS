# Замеры и диагностика готовой модели на RTX 4090

Основа: `codex/experiment-measurements` (58f4d7b), включённая в
`additional_metrics_PLAN` (6c1d139). Подготовлено под фактический 4090-baseline-001.
Первый отчёт и его приложения не изменяются. Новые измерения — отдельные эксперименты.
Промпт, целевой формат, схема Facts и подпись контракта существующей модели сохранены.

## Что изменено относительно веток

- Лимит входа по умолчанию 12288, полная последовательность 16384, ответ 4096.
- Для нового обучения по умолчанию одна эпоха. Полный pipeline требует явного
  выбора `--model ...` (без обучения) или `--train-new` (новый адаптер).
- Docker по умолчанию измеряет существующую модель, CUDA wheel index — cu130.
  Сборка контейнера создаёт новое окружение; не заменяет проверенную серверную .venv.
- Ошибки синтаксиса JSON, схемы, источника и достижения лимита различаются.
  `failures.jsonl` сохраняет call_id, raw_response, подробности ошибки и длины.
  Raw-ответы сохраняются только для отказов в benchmark; обычный HTTP-ответ их не раскрывает.
- Локальная генерация ограничена схемой `Facts` через XGrammar. В `report.json`
  появились `constrained_json` и `grammar_compile_seconds`. Для A/B можно один раз
  повторить срез с `JSON_CONSTRAINED=0`; в остальных прогонах оставлять включённым.
- `benchmark --failures FILE` проверяет только call_id со значением error из старого
  JSONL. Такой диагностический срез запрещено включать в итоговую кривую.
- Сравнение контролирует также сведения об оборудовании и число потоков внутри
  каждой группы CPU/GPU. Зафиксированные версии пакетов и исходников остаются в environment.json.

## Подготовка на существующем сервере

Команды выполняются вручную в `/home/ubuntu/mws`, в активной `.venv` и tmux.
Код нужно предварительно обновить; старые out не удалять и не перезаписывать.
На сервере уже есть psutil 7.2.2. Переустановка всего requirements не требуется.

```bash
source .venv/bin/activate
export BASE_MODEL=Qwen/Qwen3-0.6B
export MAX_INPUT_TOKENS=12288
export MAX_SEQUENCE_TOKENS=16384
export MAX_NEW_TOKENS=4096
export PROMPT_LOOKUP_TOKENS=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
python -c "import psutil; print(psutil.__version__)"
python -c "import importlib.metadata as m; print(m.version('xgrammar'))"
python -m unittest discover -s tests -v
```

Если `xgrammar` ещё нет, поставить только его, не переустанавливая torch/transformers:
`pip install "xgrammar>=0.2.8,<1"`. При первом запуске `benchmark.py` или сервиса
грамматика компилируется один раз; время попадёт в `grammar_compile_seconds`.

Пакет psutil нужен для RSS памяти процесса. Это не сервис и не внешний аккаунт.
На сервере версии PyTorch/Transformers не менять ради подключения метрик.

## 1. Пересчёт уже сохранённых ответов — без генерации и обучения

```bash
python src/eval.py --gold data/organizer/val.jsonl \
  --pred out/pred-4090-baseline-001-full.jsonl \
  --json out/eval-4090-baseline-001-detailed-002.json \
  --errors out/errors-4090-baseline-001-detailed-002.jsonl
```

Официальные метрики должны совпасть с первым отчётом. Новые поля per_attribute
детализируют пропуски, несовпавшие предсказания, даты начала/конца и источники.
Несовпавший факт не автоматически галлюцинация; совпадения считаются официальным matcher.
Разделение temporal/static по атрибуту — группировка, а не полный анализ переходов состояний.

## 2. Диагностический повтор отказавших звонков

```bash
python -u src/benchmark.py --input data/organizer/val.jsonl \
  --failures out/errors-4090-baseline-001-full.jsonl \
  --model out/merged-4090-baseline-001 --device cuda --batch-size 2 \
  --output-dir out/diagnose-4090-baseline-001-002
```

Читать failures.jsonl (причины и исходные ответы), metrics.json и calls.jsonl.
Если повторный ответ корректен, он не попадёт в failures.jsonl; он сохранится в
predictions.jsonl. Новые пачки отличаются от старого полного val, поэтому это новый
диагностический эксперимент: он не восстанавливает историческую причину каждого отказа.
Ошибки стадии прогрева хранятся отдельно; OOM прогрева останавливает прогон.
Для близкого сравнения со старым batch=2 можно повторить полный val с новым output-dir.

## 2a. Полный val с constrained decoding

Тот же merged-адаптер, но генерация ограничена схемой `Facts`. Новый каталог
обязателен: старые предсказания не перезаписываются.

```bash
export MAX_INPUT_TOKENS=12288 MAX_NEW_TOKENS=4096 MAX_SEQUENCE_TOKENS=16384
export PROMPT_LOOKUP_TOKENS=0 JSON_CONSTRAINED=1 JSON_GRAMMAR_BACKEND=torch_native
python -u src/benchmark.py --input data/organizer/val.jsonl \
  --model out/merged-4090-baseline-001 --device cuda --batch-size 2 \
  --output-dir out/json-constrained-4090-baseline-001-full-001
```

`JSON_GRAMMAR_BACKEND=torch_native` нужен на этом сервере: Triton JIT не собирается
без Python-заголовков. На машине с `python3-dev` можно оставить `auto`, чтобы
использовать быстрый CUDA-путь XGrammar. Проверять в `report.json`:
`constrained_json`, `error_counts.invalid_json`, `normalization_counts`,
`quality.diagnostics.format_valid_rate`.

Результат прогона: 0 отказов, `format_valid_rate = 1.0`, 383 звонка за 2 942 с,
7,68 с на звонок, ответ p50/p95 432/635 токенов, VRAM reserved 6,2 ГиБ.
Нормализовано 8 speaker, 10 невозможных дат, 3 обратных интервала. После удаления
111 точных дублей precision / recall 0.3625 / 0.3908, `temporal_recall` 0.3315,
`state_accuracy` 0.3467, grounding 0.7656. На шумном val targeted retry 1.2
восстановил 3 звонка с `output_token_limit`: формат 100 %, precision / recall
0.3445 / 0.3631, `temporal_recall` 0.3194, `state_accuracy` 0.3114.

## 3. Короткая проверка замеров

```bash
python -u src/benchmark.py --input data/organizer/val.jsonl \
  --model out/merged-4090-baseline-001 --device cuda --batch-size 2 --limit 20 \
  --output-dir out/benchmark-4090-preview-002
```

Это техническая проверка на первых 20 звонках, не итоговый результат.

## 4. Три режима одной модели без переобучения

```bash
python -u src/run_experiment.py --model out/merged-4090-baseline-001 \
  --output-dir out/measurements-4090-baseline-002 \
  --modes cuda:1 cuda:4 cpu:1
```

Перед полным запуском проверить CPU на тех же первых 20 звонках отдельным benchmark:
он может быть значительно медленнее GPU. Для пилотного pipeline есть `--eval-limit 20`;
такую таблицу явно подписывать как пилотную. Для сдачи запланировать полный согласованный срез.
Все сравниваемые режимы должны использовать одинаковые val, порядок, лимиты и прогрев.

Будут сохранены report.json, predictions.jsonl, calls.jsonl, batches.jsonl,
failures.jsonl, memory.csv; затем comparison.csv, per_attribute.csv и resources-quality.svg.
Память CUDA — пики PyTorch allocated/reserved после прогрева; RSS — выборки процесса,
не суммарная RAM машины. Краткие пики RSS и память драйвера этим способом не покрыты.
p50/p95 — время завершения всей пачки для звонка, без сети/очереди.
GPU-секунды — одна выделенная карта × измеренное время, не время CUDA-ядер.
Не сравнивать скорость напрямую с прежним predict.run.json: там другое окно таймера.

## Что этот код не доказывает

Факт наличия сборщика не означает выполнение критериев. Нужны реальные отчёты CPU/GPU,
три измеренные точки, исправление дат и корректная схема на всех примерах.
Ресурсы и число параметров учителя нужно получить отдельно у организаторов.
Сохранить веса и полный checkpoint вне арендуемой машины — отдельная задача.

make_noisy_val.py — локальный стресс-тест, не замена скрытому adversarial-срезу.
Он может удалить отрицание или дату, сохранив старый gold: часть деградации тогда
объясняется потерей информации. Просмотреть примеры вручную; не выдавать это за
измерение только устойчивости к опечаткам. Не сравнивать с baseline-предсказаниями
по исходному тексту как с повторным запуском учителя на шумном тексте.

Все raw-генерации и предсказания оставлять в игнорируемом out; не включать их в Git
или публичные приложения. Для защиты переносить сводные таблицы и выбранные разрешённые примеры.
