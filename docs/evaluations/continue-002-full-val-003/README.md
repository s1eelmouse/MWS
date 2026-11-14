# Полная проверка второй модели — continue-002-full-val-003

Только инференс на всех 383 звонках val. Ни train.py, ни merge.py не вызываются.
Используется существующая FP16-модель /home/ubuntu/mws/out/merged-4090-continue-002.
До запуска проверяются её байты по SHA256 из предыдущего полного benchmark,
подпись контракта и исходный val. Оценка использует gold, но backend получает только CallInput.

Основа кода: синхронизированный main 7b0f585 с интеграциями команды. Локально добавлены
progress failed, сохранение raw и учёт времени заключительной обработки. Полные
SHA256 файлов находятся в EVALUATION_TRANSFER_MANIFEST.json переносимого комплекта.
Снимки предыдущих тренировок не изменялись.

Локальная проверка подготовки: 141 компонентный тест, 136 успешно, 5 пропущены
без ML-зависимостей. Семь HTTP-тестов исключены после таймаута локального TestClient,
поэтому успешный полный HTTP-прогон не заявляется. Компиляция Python и bash -n прошли.
Настоящая CUDA/XGrammar-генерация в этой подготовке локально не выполнялась;
проверка компиляции грамматики выполняется через server preflight.

## Включённые компоненты

- JSON_CONSTRAINED=1, JSON_GRAMMAR_BACKEND=torch_native, 8 потоков компилятора.
- TEMPORAL_NORMALIZATION=task: обработка дат внутри звонка и финальная обработка серий.
- CATALOG_PATH указывает на организаторский catalog.json: включена нормализация каталога.
- Нормализация speaker по source и точная дедупликация из текущего prediction.py.
- RETRY_ON_TOKEN_LIMIT=1, penalty=1.2, один повтор. Время повтора включено в benchmark.
- SAVE_RAW_RESPONSES=1: все исходные генерации в raw-answers.jsonl, в том числе успешные.
- Greedy, thinking выключен, FP16, batch2, один прогревочный batch.
- Бюджеты 12288/16384/4096, без молчаливой обрезки; prompt lookup выключен.

XGrammar ограничивает генерацию и не является частью teacher-forced loss при обучении.
Этот запуск оценивает продукт целиком, не вклад только XGrammar или только дат.

## Установка на сервер

Код распаковывать в новый ~/mws-val-003. Веса и данные остаются в ~/mws;
используется /home/ubuntu/mws/.venv. Не обновлять общую папку, в которой работает другая смена.

```bash
mkdir ~/mws-val-003
tar -xzf ~/mws-full-val-003.tar.gz -C ~/mws-val-003
bash ~/mws-val-003/docs/evaluations/continue-002-full-val-003/run.sh check
```

check проверяет код, модель, валидатор и компиляцию схемы, не выполняет генерацию.
Если он падает, основной запуск не выполнять. XGrammar требуется >=0.2.8,<1 согласно
requirements; сначала проверить уже установленный пакет. При необходимости посмотреть
план `python -m pip install --dry-run 'xgrammar>=0.2.8,<1'` из активной серверной .venv.
Не менять torch/transformers без согласования; их версии попадут в environment.json.

## Запуск

Убедиться через nvidia-smi, что GPU свободен, затем:

```bash
tmux new-session -d -s mws-val-003 \
  'bash /home/ubuntu/mws-val-003/docs/evaluations/continue-002-full-val-003/run.sh'
```

Внутри другой tmux-сессии: `tmux switch-client -t mws-val-003`.
Из обычного SSH: `tmux attach -t mws-val-003`. Отсоединение: Ctrl+B, затем D.
Есть поток Completed 2/383; failed: 0, затем 4/383 и так далее.
Счётчик показывает ошибки исполнения/формата, а не несовпадения фактов с эталоном.
Качество по эталону считается после получения полного набора.

Лог: /home/ubuntu/mws/out/continue-002-full-val-003.log.
Выход: /home/ubuntu/mws/out/continue-002-full-val-003/.
При существовании лога/выхода запуск прекращается. Для нового повтора передать новый RUN_ID.
Завершение: DONE: full 383-call evaluation complete; no training.

## Что сохраняется

| Файл | Содержимое |
|---|---|
| report.json | Настройки, backend XGrammar, время, память, токены, сбои, нормализации и качество |
| environment.json | Оборудование, версии, хеши исходников |
| predictions.jsonl | Итоговые факты после обработки серий |
| predictions-before-series.jsonl | Факты до обработки серий; уже после XGrammar и нормализации внутри звонка |
| raw-answers.jsonl | Сырые генерации; при retry также initial_raw_response |
| calls.jsonl | Токены, длительности, retry, решения нормализации по звонкам |
| batches.jsonl | Фактические размеры/время пачек |
| failures.jsonl | Причины отказов и raw |
| temporal-audit.jsonl | Финальные изменения дат, если были решения |
| metrics.json / errors.jsonl | Официальные метрики, срезы по атрибутам, диагностика |
| memory.csv / postprocess-memory.csv | Выборки RSS при инференсе и при финальной обработке |
| series-date-comparison/ | Исправления/регрессии дат финальной обработки серий на тех же фактах |
| run.sh / code-transfer-manifest.json | Сценарий и версия использованного кода |

measurement_seconds и основные p50/p95 сохраняют прежнее определение: локальная
подготовка, генерация и нормализация внутри звонка; без загрузки, прогрева, сети,
записи файлов, финальной обработки серий и оценки качества.
Дополнительно сохраняются temporal_postprocess_seconds, processing_seconds_including_postprocess,
calls_per_hour_including_postprocess, gpu_seconds_per_1000_calls_including_postprocess.
Для целого продукта использовать показатели including_postprocess; для сравнения
с прежними benchmark учитывать, что финальная обработка раньше могла быть вне таймера.
Новая raw-запись может влиять на стенд через I/O, хотя она вне измеренных интервалов.

Пиковый RSS инференса — memory.process_rss_sampled_peak_bytes, финальной обработки —
postprocess_memory; общий выборочный максимум — combined_process_rss_sampled_peak_bytes.
CUDA allocated/reserved — пики аллокатора, не вся память драйвера и не GPU utilization.
Это одна ресурсная точка на GPU, не готовая кривая из трёх режимов и не CPU-замер.

## Получение результатов

После DONE на сервере:

```bash
tar -czf ~/continue-002-full-val-003-results.tar.gz -C ~/mws/out \
  continue-002-full-val-003 continue-002-full-val-003.log
```

Скачать архив в server-results вне репозитория. Сырые ответы и датасет не коммитить.
Основной код, флаги и сценарий готовы для последующего коммита владельцем;
эта подготовка не включает Git-операции или запуск на сервере.
