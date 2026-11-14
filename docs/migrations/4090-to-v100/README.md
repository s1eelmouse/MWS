# Перенос рабочей модели в team_shpana

Дата подготовки: 14.11.2026. Старый сервер: RTX 4090, `ubuntu@81.94.159.247`.
Новый: V100 32 GB, SSH alias `inh-compute`, пользователь `al-galiullin`.
Цель — `~/team_shpana`; промежуточные модель и данные на ПК — только
`D:\MWS\team_shpana-transfer`. Полный архив сервера и поиск всех старых логов
отложены по решению команды. Эти операции не запускались.

## Состав переноса

- Текущий код и тесты, включая XGrammar, правила дат и общую память заявок.
- Готовая merged-модель `4090-continue-002`: FP16-веса, конфигурации,
  токенизатор, `case_contract.json` и остальные файлы её каталога.
- Сохранённый адаптер второго цикла, если он найден: только файлы в корне
  `out/4090-continue-002`, без дерева checkpoints и состояния оптимизатора.
- Организаторские `train.jsonl`, `val.jsonl`, `catalog.json`.
- Новое изолированное окружение Python 3.12 с PyTorch CUDA 12.6.

Исторические серверные логи, все `out/`, кэши Hugging Face, старая `.venv`,
SSH-ключи и личные настройки в этот пакет не входят. Уже имеющиеся отчёты
репозитория входят в небольшой пакет кода. Для инференса готовая merged-модель
самодостаточна; скачивать базовую модель заново не требуется.
Если начнём новый цикл QLoRA с адаптера, понадобятся исходные базовые веса
`Qwen/Qwen3-0.6B`; их получим отдельно из старого кэша или HF. Повторно накладывать
тот же адаптер на merged-модель нельзя: это повторно добавит его поправки.

## Совместимость и зависимости

V100 — Volta, compute capability 7.0 (`sm_70`). Нужна не просто надпись CUDA 12.x,
а сборка PyTorch, содержащая kernels для `sm_70`. Выбран кандидат
**torch 2.14.1 + cu126**: это та же версия PyTorch, что была на 4090, с другой
CUDA-сборкой. Наличие wheel подтверждено официальной страницей PyTorch;
пригодность именно на этом сервере подтвердит реальный probe.

Python 3.14 в системе сохраняется. `uv` устанавливает отдельный Python 3.12
и окружение `~/team_shpana/.venv-v100`. Ни драйверы, ни глобальная CUDA, ни
системные библиотеки других команд не меняются. Для готовых PyTorch wheels
полный CUDA Toolkit/nvcc не нужен; выбран `JSON_GRAMMAR_BACKEND=torch_native`.
`torchvision`, `torchaudio`, Docker, vLLM и FlashAttention-2 для этой схемы
не устанавливаются. Остальные пакеты — из `requirements.txt` с ограничениями
`constraints-v100.txt`; после успешной установки сохраняется точный freeze.

На V100 обучение должно выбирать FP16. В проект добавлен `gpu_precision.py`,
который исключает BF16-эмуляцию на GPU старше Ampere; `doctor.py` показывает
compute capability и архитектуры установленной сборки. Совместимость QLoRA NF4
проверяется маленькой реальной операцией bitsandbytes, а не только импортом.
Выбранные версии — кандидат, пока установка и probe не прошли на новом сервере.
Источники разделяются через `uv pip install --torch-backend cu126`:
PyTorch берётся из CUDA-индекса, общие пакеты — из PyPI. Общий `--index` с URL
PyTorch для всех requirements не используется: он может скрыть новые версии requests.

Ожидаемый объём модели — около 1.2 GB до сжатия, плюс небольшой адаптер,
токенизатор и примерно 34 MB train/val. Точный размер сообщает manifest.
V100 имеет больше VRAM, но скорость относительно 4090 нужно измерять;
старые GPU-секунды нельзя автоматически приписывать новому оборудованию.
Длинный контекст может потребовать больше памяти из-за другого attention kernel.

## Порядок действий с ПК

Из обычного PowerShell, команды выполняются сверху вниз. Для старого сервера
нужна его прежняя доступность; для нового — сеть университета. Приватные ключи
остаются на ПК. Скрипт `Transfer-TeamShpana.ps1` и `model-transfer.py` лежат
в `C:\Users\Tripozik\Desktop\MWS\outputs` рядом с пакетом кода.
Для нового сервера разрешены интерактивные запросы passphrase. Вводите фразу
в своём терминале при запросе; она может понадобиться несколько раз за этап
(для отдельных `scp` и `ssh`). Символы при вводе не отображаются.

**1. Подготовить только модель и данные на старом сервере:**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Tripozik\Desktop\MWS\outputs\Transfer-TeamShpana.ps1" -Stage PrepareOld
```

Фоновый tmux сохранит задачу при разрыве SSH. Готовность можно посмотреть:

```powershell
ssh -i "C:\Users\Tripozik\Desktop\MWS\hackathon-235824-Tripozik.pem" ubuntu@81.94.159.247 "tail -n 20 /home/ubuntu/team-shpana-transfer/pack-001.log"
```

Пакет готов, когда в конце появилась сводка с `archive`, `manifest`, `bytes`.
До этого команда Download ожидаемо не найдёт итоговый manifest.

**2. Доставить код на новый сервер:**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Tripozik\Desktop\MWS\outputs\Transfer-TeamShpana.ps1" -Stage Code
```

Это создаёт `~/team_shpana`, без GPU-задач. При наличии `team_shpana/src`
повторная распаковка останавливается, чтобы сохранить существующие файлы.

**3. Когда старый пакет готов, скачать его на D и проверить SHA256:**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Tripozik\Desktop\MWS\outputs\Transfer-TeamShpana.ps1" -Stage Download
```

**4. Передать готовую модель новому серверу:**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Tripozik\Desktop\MWS\outputs\Transfer-TeamShpana.ps1" -Stage UploadModel
```

Сначала проверяется SHA256 архива, после распаковки — каждого файла.
Проверяется SHA256 именно второй merged-модели:
`ef0bc5ae0646f6a4bda4cee69fe64b292d710e5b54119ac5ae9868720af584f7`.
Создаётся `~/team_shpana/out/v100-assets.json` с новыми абсолютными путями.
Также сохраняется `~/team_shpana/runtime.env` с путями модели, каталога и
настройками XGrammar/дат/истории для ручного использования через `source`.
Существующие целевые файлы не перезаписываются.

**5. Поставить окружение в фоне:**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Tripozik\Desktop\MWS\outputs\Transfer-TeamShpana.ps1" -Stage Install
```

Загрузка пакетов не требует свободной GPU, но в конце выполняется небольшой
CUDA-probe. Он проверяет хотя бы 2 GiB свободной памяти перед операциями.
В новом SSH-сеансе:

```bash
tail -n 30 -F ~/team_shpana/install-001.log
```

Успех: `INSTALL AND TESTS COMPLETE`. При сбое сохраняются журнал, freeze и
диагностика; нельзя считать окружение проверенным только по факту создания venv.
Если первая установка остановилась на разрешении зависимостей, существующую
`.venv-v100` можно продолжить без пересоздания и повторной передачи модели:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Tripozik\Desktop\MWS\outputs\Transfer-TeamShpana.ps1" -Stage ResumeInstall -RunId 002
```

Новый журнал на сервере: `~/team_shpana/install-resume-002.log`.
Сценарий повторно проверяет исходные хеши кода, сохраняет собственную копию,
устанавливает оставшиеся пакеты и выполняет тот же комплект проверок.
При NixOS/нестандартном loader ошибка запуска managed Python требует отдельной
настройки администратора; автоматического изменения системы в сценарии нет.

## Первая проверка на GPU

Сначала нужно убедиться в `nvtop`, что есть согласованное окно для проверки.
GPU общий с другими командами. Сценарий не останавливает чужие процессы.
Начальные настройки: batch 1, 4 CPU-потока, 4 потока грамматики,
ограничение PyTorch allocator 12 GiB. Проверяется минимум 14 GiB свободной
VRAM до запуска. Лимит не резервирует память эксклюзивно и не охватывает
все сторонние CUDA-аллокаторы; параллельные команды всё ещё могут вызвать OOM.

В SSH нового сервера после установки и переноса модели:

```bash
tmux new-session -d -s shpana-smoke-001 \
  '$HOME/team_shpana/.venv-v100/bin/python -u $HOME/team_shpana/docs/migrations/4090-to-v100/validate-v100.py --run-id v100-smoke-001 --mode smoke > $HOME/team_shpana/out/v100-smoke-001.log 2>&1'

tail -n 30 -F ~/team_shpana/out/v100-smoke-001.log
```

Это проверка первых пяти звонков, без обучения. После успеха отдельным запуском
нужно проверить четыре самых длинных (`--mode long --run-id v100-long-001`),
а затем весь val (`--mode full --run-id v100-full-001`). При OOM нельзя молча
урезать транскрипты: сначала изучить лог и согласовать иной бюджет/окно GPU.

Каждый запуск пишет обычный комплект `report.json`, `metrics.json`, `memory.csv`,
сырые ответы, результаты до/после памяти, прогресс и суммарный failed.
Отдельный `*-gpu-policy.json` фиксирует бюджет и свободную VRAM перед запуском.
Грамматика, календарь и история включены; всё выполняется в измеряемом инференсе.

## Что подготовлено и что ещё не сделано

Из локальных инструментов SSH нового сервера закрывает соединение, старый
не ответил вовремя. Удалённые команды выше ещё не выполнены агентом;
фактические модель и окружение на новом сервере не проверены. Программные
тесты подготовки не являются доказательством успешной работы V100.
Локальный комплект: 200 тестов, 195 успешно, 5 пропущены из-за отсутствия
ML-зависимостей; Python-компиляция, PowerShell parser и `bash -n` успешны.
Git-коммиты, push и удаление файлов на серверах не выполнялись.

Официальные источники:
[PyTorch CUDA 12.6 builds](https://pytorch.org/get-started/previous-versions/),
[NVIDIA: Volta, sm_70 и последняя CUDA 12.x](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html),
[bitsandbytes: NF4 и требования](https://huggingface.co/docs/bitsandbytes/installation),
[PyTorch: проверка нативного BF16](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.is_bf16_supported.html).
