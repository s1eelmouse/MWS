# Приложения к эксперименту 4090-baseline-001

Оригинальные серверные файлы сохранены побайтно. Пути под attachments повторяют пути относительно /home/ubuntu/mws. Исходные записи не редактировались; выводы и преобразования находятся в отчёте и verification.json уровнем выше.

| Файлы | Назначение |
|---|---|
| collection_manifest.json | Размеры и SHA256 21 файла, хеши исходников/архива кода, revision кеша, список checkpoint и исключений |
| out/4090-baseline-001.log | Полный лог основной эпохи, включая финальные loss и ресурсные замеры |
| out/4090-baseline-001/training_stats*.json | Точные время, шаги и пики GPU-памяти обучения |
| out/4090-baseline-001/experiment.json | Модель, seed, хеши train/val/schema, длина и batch |
| out/4090-baseline-001/adapter_config.json | Фактическая конфигурация сохранённого LoRA |
| out/4090-baseline-001/case_contract.json | Подпись контракта вывода |
| out/4090-baseline-001/checkpoints/checkpoint-{200,246}/trainer_state.json | Шаг, эпоха, история loss, параметры состояния Trainer |
| out/4090-baseline-001/checkpoints/checkpoint-{200,246}/training_args.bin | Оригинальные сериализованные TrainingArguments; читаемая проверенная часть в training-arguments-verified.json уровнем выше |
| out/doctor-4090-001.json | Версии, GPU, VRAM и поддержка BF16 |
| out/requirements-4090-001.txt | Полный список установленных пакетов |
| out/repro-4090-baseline-001/* | Копии окружения, revision базы и описание команды |
| data/organizer/package_manifest.json | Хеши данных, счётчики и результаты валидации сплитов; строк датасета здесь нет |
| out/pred-4090-baseline-001-full.run.json | Модель, batch, число вызовов/отказов и время инференса |
| out/eval-4090-baseline-001-full.json | Полная официальная оценка дообученной модели |
| out/eval-organizer-baseline-full-001.json | Полная официальная оценка выданных baseline-предсказаний |
| out/errors-4090-baseline-001-full.jsonl | 383 call_id: счётчики совпадений либо код ошибки; текстов разговоров и raw-генераций нет |
| out/eval-4090-baseline-001-preview.json | Предварительная оценка первых 20 звонков |

Не входят: веса, строки датасета, полные предсказания, ключи SSH и .env. Архив отчётности не заменяет резервную копию модели и полного checkpoint.
