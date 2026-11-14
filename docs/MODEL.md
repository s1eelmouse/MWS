# Финальная модель

## Идентичность

| Поле | Значение |
|---|---|
| База | `Qwen/Qwen3-0.6B`, 596 049 920 параметров |
| Адаптер | `4090-continue-002` (QLoRA, 4 бита, r=16, все линейные слои) |
| Старт второго цикла | `4090-baseline-001` (одна эпоха QLoRA) |
| Второй цикл | LR `1e-4`, +1 эпоха, seed 42, batch 1, max sequence 16 384 |
| Объединённая модель | `out/merged-4090-continue-002` |
| Контракт | `case_contract.json`, подпись `3ee403f3…a274` |
| Формат | constrained JSON по `Facts.model_json_schema()`, XGrammar |

Хеши данных и стартового адаптера записаны в `out/4090-continue-002/experiment.json`:
train `041dacaa…d6a`, val `a2cd8a6e…19e`, schema `63118114…050`.
Веса в Git не входят: они лежат в `out/` на сервере и исключены из репозитория.

## Инференс

```bash
export MAX_INPUT_TOKENS=12288
export MAX_NEW_TOKENS=4096
export MAX_SEQUENCE_TOKENS=16384
export JSON_CONSTRAINED=1
export JSON_GRAMMAR_BACKEND=torch_native   # если Triton JIT не собирается
export TEMPORAL_NORMALIZATION=task         # по умолчанию
python -u src/benchmark.py \
  --input data/organizer/val.jsonl \
  --model out/merged-4090-continue-002 \
  --device cuda --batch-size 2 \
  --output-dir out/json-constrained-continue-002-full-001
```

`benchmark.py` после генерации сам применяет пакетный шаг `task`: переносит
даты внутри серий и пересчитывает официальные метрики по уже исправленным
предсказаниям. В каталоге прогона появляется `temporal-audit.jsonl`, а в
`report.json` — `temporal_postprocess`. Поэтому строка «`task` + серии» в таблице
получается одним запуском benchmark, без ручной операции.

Для аудита уже сохранённых предсказаний тот же шаг доступен отдельно:

```bash
python src/postprocess_dates.py \
  --input data/organizer/val.jsonl \
  --pred out/json-constrained-continue-002-full-001/predictions.jsonl \
  --output-dir out/continue-002-task \
  --mode task
python src/eval.py --gold data/organizer/val.jsonl \
  --pred out/continue-002-task/predictions.jsonl --json out/continue-002-task/eval.json
```

HTTP-сервис (`src/serve.py`) применяет пофактные правила `task`; перенос дат
между звонками серии доступен только пакетному benchmark/постобработке, потому
что сервис обрабатывает один звонок.

## Качество на полном val

383 звонка, официальный оценщик:

| Вариант | temporal_recall | state_accuracy | precision / recall | grounding | формат |
|---|---:|---:|---:|---:|---:|
| База glm-5.3 (пересчёт) | 0.4205 | 0.3194 | 0.219 / 0.249 | 0.691 | 91 % |
| `continue-002`, без дат | 0.3566 | 0.3852 | 0.4008 / 0.4119 | 0.7987 | **100 %** |
| `continue-002`, `task` на звонок | 0.4000 | 0.3852 | 0.4008 / 0.4119 | 0.7987 | **100 %** |
| `continue-002`, `task` + серии | **0.4237** | **0.3852** | **0.4008 / 0.4119** | **0.7987** | **100 %** |

Скорость: 6,90 с на звонок (batch 2, RTX 4090), 521 звонок в час, VRAM reserved
6,2 ГиБ, 0 отказов. Режим `task` распознаёт типичные ASR-искажения cue-слов
(«вчира», «чирез», «сегадня»); для `action` используются только точные cue.
Шумный val (`--level 0.15 --seed 42`) для baseline с `task` даёт
`temporal_recall` 0.3194 → 0.3806; прогон этой модели на шумном val — отдельный
замер.

## Ограничения

- `service_state` и `tariff` остаются слабыми: историческое начало или день
  внутри «месяца назад» не выводятся из текущей реплики.
- `agreement` использует эвристику «время события при явном будущем времени»;
  на train её точность 0.383, потому что корпус смешивает время события и
  время заявления.
- Перенос даты «принята» между звонками серии работает только в пакетном
  режиме и только внутри одного `account_id`.
- Это измерение на val и локальном шумном стрессе, не на скрытом
  adversarial-срезе.
