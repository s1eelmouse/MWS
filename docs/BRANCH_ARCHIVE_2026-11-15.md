# Архив локальных веток — 15.11.2026

Перед очисткой создан и проверен полный Git bundle `../output/git-backups/MWS-before-release-20261115.bundle`.
Карта ссылок: `../output/git-backups/retired-local-branches-20261115.json`.
Это локальный бэкап истории кода; веса и игнорируемые результаты остаются отдельно и не удалялись.

Удалены только устаревшие локальные ссылки:

- `audit/criteria-1-3-readiness` — `36bdd73997797da1b5c995de23fba0f69743c709`
- `audit/current-model-005` — `b9dbfcbb08ec7159d5ec40493b8557e76255f1a7`
- `audit/dataset-v21-v100` — `f36f7eac22c336a5564adbebdc79eb1cd5ba8d35`
- `codex/compactfact-adaptation` — `ecea4c3946d64c071f68d76c849397884fbfb64f`
- `feat/defense-evidence-suite` — `36bdd73997797da1b5c995de23fba0f69743c709`
- `feat/temporal-binding-v2` — `992b439f2203b047254aa75530fc6646462758ac`
- `feat/temporal-normalization` — `5fc3d98c4cff3833f0319180c81f9276c68e34e6`
- `fix/temporal-evidence-scoping` — `297bc5cacaa676a2ee03018e51ec2a1cb9ddef56`
- `test/multilingual-readiness` — `c11d8f760b51fa052e4e84aa12d34cec1f4ac435`

Оставлены `main`, `fix/release-005-readiness`, `fix/current-tariff-service-recovery` и `fix/state-interval-dates-v21`.
Две последние содержат ещё не проверенные на 005 альтернативы; их нельзя считать ненужными до A/B.
Удалённые ветки восстанавливаются из bundle либо существующих удалённых ссылок.
Удалённые ветки GitHub и рабочая ветка товарища `egor` не удалялись: на них остаются материалы команды.

Из `feat/defense-evidence-suite` перенесён и адаптирован `resource_matrix.py`: теперь он включает историю и проверяет выбранные веса. Из прежних исправлений сохранена проверка эквивалентных путей macOS.
Правила извлечения из экспериментальных веток не подмешивались к измеренной 005 без отдельного сравнения.
