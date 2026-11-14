# Демо для защиты

Сценарий на 2–3 минуты. Не отправлять эталонные `facts` из val и не показывать
скрытый тест. Используем синтетический звонок из [EXAMPLES.md](EXAMPLES.md).

## Подготовка

```bash
export MODEL_DIR=out/merged-4090-continue-003   # или 002, если 003 не подтверждён
export JSON_CONSTRAINED=1
export JSON_GRAMMAR_BACKEND=torch_native
export TEMPORAL_NORMALIZATION=task
python -m uvicorn serve:app --app-dir src --host 0.0.0.0 --port 8000
```

Если сервис уже поднят в Docker:

```bash
curl -fsS http://127.0.0.1:8000/health
```

Ожидаемо: `status: ok`, `constrained_json: true`, путь к модели.

## Звонок с относительными датами

Сохранить в `/tmp/demo.json`:

```json
{
  "call_id": "demo:relative-dates",
  "started_at": "2026-03-11T10:00:00",
  "turns": [
    {"idx": 0, "speaker": "client",
     "text": "Вчера пропал интернет, а со вторника не работает телевизор."},
    {"idx": 1, "speaker": "operator",
     "text": "Заявку приняли сегодня. Мастер приедет в пятницу после двух."},
    {"idx": 2, "speaker": "client", "text": "Да, в пятницу подходит."}
  ]
}
```

```bash
curl -fsS -X POST http://127.0.0.1:8000/extract \
  -H 'Content-Type: application/json' --data-binary @/tmp/demo.json
```

Что показать в ответе:

- `service_state` «нет доступа в интернет» → `valid_from = 2026-03-10`;
- `ticket_status` «принята» → `valid_from = 2026-03-11`;
- `agreement` «выезд специалиста» → `valid_from = 2026-03-13`;
- `source.turn_idx` — исходная реплика, не позиция в массиве.

Точный набор фактов зависит от модели; на защите комментируем даты и источники,
а не обещаем дословный JSON. Если модель вернула меньше фактов — это ожидаемо
для 0,6B, показываем то, что есть, и переходим к метрикам полного val.

## Метрики

Открыть `report.json` полного прогона:

- `quality.official.temporal_recall = 0.4377` (continue-003, чистый val);
- `state_accuracy = 0.3949`;
- `format_valid_rate = 1.0`, `failed_calls = 0`;
- шумный `continue-002`: T 0.4074, S 0.3451, формат 100 %.

Важно: пакетный результат с переносом серий не равен HTTP-ответу на один звонок.
Для демо одного звонка это нормально; не приписывать HTTP-сервису пакетные 0.4377.

## Если демо не идёт

1. Показать `/health` и сохранённый `report.json`.
2. Включить резервное видео (2–3 минуты), записанное заранее.
3. Не тратить защиту на отладку сети; перейти к слайдам и Q&A.

## Чек-лист перед выходом

- [ ] модель загружена, `/health` отвечает 200;
- [ ] синтетический звонок отправляется без `facts`;
- [ ] в ответе есть `call_id` и валидные `facts`;
- [ ] открыт `report.json` с финальными числами;
- [ ] видео-резерв лежит локально и открывается без сети.
