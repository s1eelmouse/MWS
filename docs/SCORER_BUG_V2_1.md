# Баг официального оценщика 2.1

Оценщик: `case_metrics.py`, `VERSION = "2.1"`, SHA256
`863f87737dd1108e772250967f43697fa2411b3394c190d192b7ff66666259c3`.
Код организаторов **не изменялся**; файл в `src/official_case_metrics_v21.py`
байт-в-байт.

## Репродьюсер

`gold.jsonl`:

```json
{"call_id":"bug:1","started_at":"2026-03-11T10:00:00","turns":[{"idx":0,"speaker":"client","text":"Мастер приедет через две недели."}],"facts":[{"object":{"id":"x","type":"client","name":"клиент"},"attribute":"agreement","value":"выезд","polarity":1,"valid_from":"2026-03-25T00:00:00","valid_to":null,"source":{"turn_idx":0}}]}
```

`pred.jsonl` — те же факты (достаточно скопировать строку).

```bash
python case_metrics.py gold.jsonl pred.jsonl
```

## Ошибка

```text
File "case_metrics.py", line 263, in recover_dates
    out.update(call_date + timedelta(days=13), call_date + timedelta(days=14), call_date + timedelta(days=15))
TypeError: 'datetime.date' object is not iterable
```

В `set.update` переданы три `date` как отдельные аргументы вместо одного
итерируемого объекта; ветка «через две недели». По аудиту 2.1 также затронут
«позапрошл-».

## Почему это важно

`recover_dates` вызывается для транскрипта каждого сопоставленного временного
факта. Если в скрытом срезе встретится такая фраза, официальный оценщик упадёт
до подсчёта метрик. Это не наша ошибка и не повод менять scorer: нужно сообщить
организаторам и дождаться исправленной версии.

## Что делать локально

- Не патчить `official_case_metrics_v21.py`; сохранять байт-в-байт и SHA256.
- Для диагностики можно отметить такие звонки и не включать их в локальный
  прогон, но финальную оценку считать только официальным скриптом.
- В отчёте указать, что баг воспроизведён и передан организаторам.
