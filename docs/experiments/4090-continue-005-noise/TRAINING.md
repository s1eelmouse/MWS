# 4090-continue-005-noise: обучение

- **Дата:** 15.11.2026
- **Стенд:** RTX 4090 24 ГБ, 8 vCPU, 31 ГиБ RAM, Ubuntu 26.04, драйвер 580.178.04
- **База:** `Qwen/Qwen3-0.6B`, 596 049 920 параметров
- **Стартовый адаптер:** `out/adapter-003`, SHA256 `982a8843905f76439fad72e3c9f32f96721f9aea4bd66a5ed1b92174e892a2ec`
- **Данные:** train 2.1, `--train-variant noise`; 3 926 исходных звонков → **7 852** после шумных копий
- **Val:** 383 звонка, не участвует в градиентах

## Параметры

| Параметр | Значение |
|---|---:|
| Эпох | 1 |
| Шагов | 491 |
| Batch / accumulation | 1 / 16 |
| Seed / data_seed | 42 / 42 |
| Learning rate | 1e-4, новый оптимизатор |
| Квантизация | 4-bit NF4 + double quantization |
| Compute dtype | BF16 |
| LoRA | r=16, alpha=32, dropout=0.05, all-linear |
| Обучаемые параметры | 10 092 544 |
| Max sequence | 16 384 |
| Gradient checkpointing | включён, `use_reentrant=False` |
| Checkpoints | каждые 50 шагов, сохранены `checkpoint-450`, `checkpoint-491` |

## Результат обучения

| Показатель | Значение |
|---|---:|
| `train_loss` | 0.10879 |
| `eval_loss` | 0.12284 |
| Время Trainer | 4 113,28 с (68 мин 34 с) |
| Пик CUDA allocated | 3 856 249 856 Б = 3,59 ГиБ |
| Пик CUDA reserved | 4 169 138 176 Б = 3,88 ГиБ |
| Статус | `complete`, 491 шаг |

`eval_loss` — teacher-forced loss на val, а не официальное качество. T/S/P/R/G
считаются отдельной генерацией в `full-val` и `noisy-val`.

## Выходы

- адаптер: `out/continue-005-noise-001`;
- merged: `out/merged-4090-continue-005-noise-001`, SHA256 `cef9fcde9f1f2a812e42c6122a0f5094fbaa456f6077596cf03ebb86e4d552f6`;
- лог обучения: `attachments/logs/training.log`;
- журнал попытки: `attachments/training/attempt-*/`;
- `experiment.json`: `attachments/training/experiment.json`.

## Воспроизведение

```bash
bash docs/experiments/4090-continue-005-noise/replay.sh test
bash docs/experiments/4090-continue-005-noise/replay.sh all
```

Скрипт использует снимок кода в `code/` и требует новый `RUN_ID`, чтобы не
перезаписать этот прогон. Старый адаптер `adapter-003` и старые результаты не
изменяются.
