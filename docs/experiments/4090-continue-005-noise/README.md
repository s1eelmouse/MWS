# 4090-continue-005-noise: дообучение на шумном train

- **Дата:** 15.11.2026
- **Стенд:** RTX 4090 24 ГБ, 8 vCPU, 31 ГиБ RAM, Ubuntu 26.04
- **Старт:** `out/adapter-003`
- **Данные:** train 2.1 + `--train-variant noise` (7 852 звонка)
- **Результат:** адаптер `out/continue-005-noise-001`, merged `out/merged-4090-continue-005-noise-001`
- **SHA256 merged:** `cef9fcde9f1f2a812e42c6122a0f5094fbaa456f6077596cf03ebb86e4d552f6`

## Зачем

`003 + recovery v2` проходит обе цели на чистом val, но на шумном val S падает
до 0.4141. Эта эпоха проверяет, помогает ли дообучение на шумных копиях train
поднять устойчивость состояний, не ломая чистый val.

## Обучение

Одна эпоха, 491 шаг, batch 1, accumulation 16, LR 1e-4, новый оптимизатор,
4-bit QLoRA, BF16, max sequence 16 384. Время Trainer — 4 113,28 с
(68 мин 34 с), `train_loss` 0.10879, `eval_loss` 0.12284, пик reserved
3,88 ГиБ. Подробности — [TRAINING.md](TRAINING.md).

## Оценка

Чистый и шумный val 2.1 выполняются отдельным `run_eval_005.sh` с теми же
`case_v2`, recovery v2, историей и batch 2. Результаты будут добавлены сюда
после завершения. Сравнение — с `003 + recovery v2`:

| Прогон | T | S |
|---|---:|---:|
| 003 + recovery v2, clean | 0.4807 | 0.5201 |
| 003 + recovery v2, noisy | 0.4582 | 0.4141 |
| 005 + recovery v2, clean | — | — |
| 005 + recovery v2, noisy | — | — |

## Файлы

- `TRAINING.md` — параметры, loss и ресурсы обучения;
- `replay.sh` — повторение из снимка кода;
- `code/` — снимок `src/` и `tests/` на коммите `1d9f63c`;
- `code-manifest.json` — SHA256 файлов снимка;
- `attachments/training/` — `experiment.json`, attempt, events, settings, status;
- `attachments/logs/` — stdout обучения и eval;
- `attachments/full-val/`, `attachments/noisy-val/` — отчёты и метрики после eval.

Веса и полные предсказания в Git не добавляются; они сохраняются на сервере и в
локальном `artifacts-4090/4090-continue-005-noise/`.
