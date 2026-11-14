# Ветка новой разметки и state_accuracy

Ветка `codex/organizer-v21-state-accuracy` создана от `d2ddc5e`. В этой версии main
уже приняты оценщик 2.1, пустые имена объектов, совместимость старых адаптеров и
`FACT_RECOVERY`. Здесь добавлены подготовка нового архива и единый каталог данных
по умолчанию, чтобы обучение/проверка не использовали прежний val незаметно.

## Подготовка данных

В активном Python-окружении из корня проекта:

```powershell
python src/prepare_v21.py --archive "C:\Users\ivanr\Downloads\Данные кейса.zip"
```

На Linux передайте путь к тому же архиву. Поддерживаются внешний архив и вложенный
`CompactFactExtraction_participants.zip`. Проверяются контрольные суммы файлов,
точная версия оценщика, схема, уникальность звонков и разделение аккаунтов train/val.
Оригинальные файлы сохраняются побайтно. Известное несовпадение собственной суммы
`CHECKSUMS.txt` выводится предупреждением; ошибки других файлов останавливают подготовку.

Результат находится под игнорируемым `data/organizer/v2_1/`; данные в Git не добавляются.
Архив на другой машине нужно подготовить отдельно. Старые `case_data/`, веса,
checkpoint и отчёты остаются историческими. Подготовка отказывается перезаписывать
отличающиеся файлы.

Путь для всех основных CLI задаётся одним параметром окружения `ORGANIZER_DIR`.
По умолчанию это абсолютный путь к `data/organizer/v2_1` данного checkout.
Явные `--train`, `--val`, `--input`, `--gold` по-прежнему поддерживаются.

## Проверка state_accuracy без GPU

```bash
python src/train.py --check-data
python src/eval.py --pred data/organizer/v2_1/baseline_pred.jsonl --reference-metrics data/organizer/v2_1/baseline_metrics.json --json out/v21-baseline-001.json
```

Ожидаемый официальный baseline: precision 0.3348, recall 0.3805,
temporal_recall 0.4706, **state_accuracy 0.5345**, grounding_turn 0.7835.
Это проверка данных/оценщика, а не новое качество наших весов.

Сначала пересчитайте сохранённые предсказания каждой модели в новый отчёт:

```bash
python src/eval.py --pred /path/to/saved/predictions.jsonl --json out/v21-model-rescore-001.json
```

Затем выполните новую генерацию на исправленном val: у 24 звонков изменены входные
поля, поэтому пересчёт исторических ответов является только диагностикой.

```bash
python src/benchmark.py --model /path/to/merged-model --device cuda --batch-size 2 --output-dir out/v21-model-benchmark-001
```

Сравнивайте одинаковый набор правил: `FACT_RECOVERY=0/1`, даты и каталог.
Механизм восстановления добавляет пропущенные текущие тарифы/статусы; его вклад
нужно проверять по state_accuracy, precision И recall. Рост state_accuracy
не следует приписывать одному обучению или одной новой разметке.

## Обучение при необходимости

Только после проверки существующих весов на новых данных:

```bash
python src/train.py --epochs 1 --init-adapter-dir /path/to/previous-adapter --learning-rate 0.0001 --output-dir out/v21-new-cycle-001
```

Это новый цикл с новым оптимизатором. Не используйте resume старого эксперимента
для подмены его train/val. Val остаётся только оценочной выборкой; эталоны не
попадают в промпты или HTTP-запросы.

Официальный `src/official_case_metrics.py` остаётся побайтно исходным оценщиком 2.1.
Известная ошибка `recover_dates` на «через две недели» / «позапрошлый» не скрывается
и не исправляется локальной подменой официального скрипта.

## Локальная проверка ветки

После объединения `feat/case-v2-dates` доступен дополнительный экспериментальный
режим `TEMPORAL_NORMALIZATION=case_v2`. По умолчанию остаётся `task`, а путь
к новому пакету по-прежнему задаёт `ORGANIZER_DIR`. Для отдельного A/B:

```bash
python src/postprocess_dates.py --input data/organizer/v2_1/val.jsonl --pred /path/to/dates-off-predictions.jsonl --mode case_v2 --date-catalog data/organizer/v2_1/catalog.json --output-dir out/v21-case-v2-001
```

Используйте ответы до календарных правил; подробности — [DATES_V2_1.md](DATES_V2_1.md).
Путь `data/organizer_v2_1` в исторических примерах той ветки здесь заменяется на
`data/organizer/v2_1` либо на явно выбранный `ORGANIZER_DIR`.

Подготовлены все 3926 train- и 383 val-звонка без ошибок схемы и источников.
Пересчёт baseline через `src/eval.py` совпал со всеми опубликованными числовыми
показателями 2.1; отчёт — локальный `out/v21-baseline-branch.json`.
Полный набор: 160 тестов, 159 прошли, один пропущен из-за отсутствия XGrammar.
`compileall` и `git diff --check` прошли. Новое обучение, GPU-инференс и проверка
реальных весов на этой машине не выполнялись.
После объединения веток: 178 тестов, 177 прошли, один пропущен (XGrammar);
compileall и проверка конфигурации `case_v2` с новым каталогом прошли.
