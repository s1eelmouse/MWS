# CompactFactExtraction
- Work on a dedicated branch. Preserve checkpoints and existing experiment outputs.
- The actual case contract is in src/schema.py. Legacy starter schema, demo data and
  teacher-labeling code have been removed; do not reintroduce them.
- Keep src/official_case_metrics.py byte-identical to the supplied organizer version.
  Its SHA256 is recorded next to it. Report known scorer issues instead of changing them.
- Training uses the supplied train split; val is evaluation only. Keep account histories
  separated. Gold facts must never enter model prompts or HTTP requests.
- Never silently truncate transcripts or labels. Over-budget calls must be reported.
- Current organizer data is v2.1 under ignored data/organizer-v2.1. Preserve legacy
  data/organizer and artifacts under ignored out/; never overwrite old experiments.
- Keep src/official_case_metrics_v21.py byte-identical to organizer v2.1 too.
  Current scoring defaults to 2.1; legacy 1 and optional labelled 2.1-datefix are
  separate modes. Do not combine their results in a resource-quality curve.
- Do not call a teacher API or publish datasets as part of local validation.
- Run python -m unittest discover -s tests and compileall after substantive changes.
  tests/test_model_smoke.py uses a tiny random CPU model if ML dependencies are installed.
- A passing CPU test does not establish RTX 4060 memory, speed or extraction quality.
