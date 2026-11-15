# Вложения прогона 4090-continue-003-v2

Компактный комплект полного val 2.1 на 4090. Веса, `raw-answers.jsonl` и
`predictions.jsonl` в Git не добавляются; они остаются на сервере.

| Файл | SHA256 |
|---|---|
| `report.json` | `244576a202ff639456f14a30e027ca26777aac7e8e8ee4cea59fb14acc961b04` |
| `metrics.json` | `7a7fe40a146791e33a1949893a5d49c62a5553d17cd892900ac0444d04ee4c26` |
| `environment.json` | `3a19766736f59eea5fe1b6aa9c827813681de1902cfa64e1ae3add8cbda2a1fe` |
| `noisy/report.json` | `642e1838bef665afea7c047f3e41487dbcc0f570633e71060a5f15c45768ad17` |
| `noisy/metrics.json` | `0ae73d34ff96809ca96490f3f4394325db0e09171b3504b71ee1e4aef59148f3` |
| `noisy/environment.json` | `3a19766736f59eea5fe1b6aa9c827813681de1902cfa64e1ae3add8cbda2a1fe` |

- `report.json` — официальные метрики, ресурсы, настройки, история, ошибки.
- `metrics.json` — метрики и разбивка по атрибутам.
- `environment.json` — окружение и версии пакетов.
- `batches.jsonl` — время и размеры пачек.
- `history-date-comparison/comparison.json` — вклад истории дат.
- `noisy/` — `report.json`, `metrics.json`, `environment.json`, `calls.jsonl`
  для шумного val 2.1 (`--level 0.15 --seed 42`); `batches.jsonl` в компактный
  комплект не входит.

Модель: `merged-4090-continue-003`, SHA256
`ad82a0a33e5826944214e14d8574110c737238437811fdad07de6c81ab8944c9`.
