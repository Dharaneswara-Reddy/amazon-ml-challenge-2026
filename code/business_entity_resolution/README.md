# Business Entity Resolution — reproduce end to end

Amazon ML Challenge 2026. For every Source-1 business, find all Source-2/3 records of the same business.
Design and every decision: `docs/DESIGN_V2.md`, `docs/DECISIONS.md` (repo root).

## Setup
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
pip install -e code/business_entity_resolution
```
Place the organizer data at `dataset/train/` and `dataset/test/` (TSV files as distributed). Runs on CPU only; tuned
for 12 cores / ~9 GB free RAM (`configs/base.yaml` → `engine`). No network access or external data is used.

## Run (data → blocking → matching → output)
```bash
ber prepare                              # TSV → Parquet cache, exact row-count checks
ber learn-maps                           # Indic legal-form token map + admin vocabulary (fit group / unlabeled)
ber normalize --split train --force      # names/addresses v2
ber normalize --split test --force
ber block --split train --sample-pct 5   # sample union for the blocking ranker
ber train-ranker                         # linear ranker + candidate cap K from the recall curve
ber block --split train                  # candidates (prints recall per pass)
ber block --split test
ber learn-logodds                        # token log-odds per stacking fold (fit group only)
ber features --split train               # fit / tune / holdout feature files with rival context
ber features --split test
ber train --name v2                      # stage 1 + OOF + stage 2; decision rule chosen on tune; holdout scored
ber predict --run artifacts/runs/<run>   # test: stage 1 → stage 2 → prior shift → decision → output/*.tsv
ber validate-output --check-ids          # organizer validator must print PASS
```
The same sequence is `scripts/run_v2.sh prep | train | predict <run_dir>` at the repo root.

Outputs: `output/matching_results.tsv`, `output/candidate_pairs.tsv`. Every run writes `artifacts/runs/<run>/`
(models as LightGBM text, predictions, `decision_params.json`, `manifest.json` with git commit, config, decisions in
effect, metrics) and appends a row to `docs/EXPERIMENTS.md`.

## Models and licenses
LightGBM (MIT) gradient-boosted trees only — two stages, each far below 8B parameters. No pretrained language model.
Libraries: DuckDB (MIT), PyArrow (Apache-2.0), NumPy (BSD), rapidfuzz (MIT), anyascii (ISC), PyYAML (MIT).

## Tests
```bash
cd code/business_entity_resolution && python -m pytest -q   # unit tests + end-to-end smoke test on synthetic data
```
