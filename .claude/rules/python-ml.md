# Python / ML conventions

- Python 3.11+. Pipeline code lives in `code/business_entity_resolution/src/`; notebooks only import from it.
- Every stage is a function callable from one CLI entry point that goes raw TSV → both output TSVs.
- Validation split is by Source 1 entity (group split), never by random pair, or scores leak.
- Report on every experiment: blocking recall (share of true pairs present in candidates), mean candidates per S1
  entity, and macro F0.5 on the held-out split. Log them to `notebooks/experiments.md` in one line per run.
- The organizers require source code "with proper comments describing the functions": every function in `src/` gets a
  short docstring (what it does, inputs, outputs). This overrides the usual no-comments default for this project.
- Prefer vectorized pandas / numpy and rapidfuzz over Python loops on pair-level features.
- Pin versions in `requirements.txt` (`pip freeze`-accurate) before the final package.
