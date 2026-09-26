# Business Entity Resolution — reproduce end to end

Amazon ML Challenge 2026. For every Source-1 business, find all Source-2/3 records of the same business.
Design and every decision: `docs/DESIGN_V2.md`, `docs/PLAN_V3.md`, `docs/DECISIONS.md` (D-001 … D-026, repo root).

## Setup
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
pip install -e code/business_entity_resolution
```
Place the organizer data at `dataset/train/` and `dataset/test/` (TSV files as distributed). Runs on CPU only; tuned
for a 12-thread / 15 GB laptop (`configs/base.yaml` → `engine`). No network access or external data is used: every
map, vocabulary and model is hand-written or learned from the provided train/test files.

## Run (raw TSVs → `output/matching_results.tsv` + `output/candidate_pairs.tsv`)
All commands from the repo root; each writes a JSON result to stdout (`--log-file` for a log).
```bash
# S1 data + normalization
ber prepare                              # TSV → Parquet cache, exact row-count checks
ber learn-maps                           # Indic legal-form token map + admin vocabulary (fit group / unlabeled)
ber normalize --split train --force
ber normalize --split test --force

# S2 blocking v2 (token IDF, char trigram, sorted neighbourhood; learned linear ranker, top-K per right)
ber block --split train --sample-pct 5   # sample union for the blocking ranker
ber train-ranker
ber block --split train
ber block --split test

# S2b blocking v3 (D-022): compound-key passes + learned cut on top of the v2 candidates → cand3_<split>.parquet
ber block3 --split train                 # trains the cut GBDT on the fit group; prints recall (~98.2%)
ber block3 --split test

# S3 self-labelled per-country vocabularies (D-025; no ground truth; countries only in test learn from test)
ber learn-selfvocab

# S4 features (147) with rival context; fit / tune / holdout groups + test
ber features --split train
ber features --split test

# S5 stage 1 + out-of-fold stage-1 + stage 2 LightGBM; decision rule chosen on tune; holdout scored
ber train --name v4

# S6 test: stage 1 → stage 2 → per-country EM prior shift (none for France, D-025) → decision → output/*.tsv
ber predict --run artifacts/runs/<run printed by train>
ber validate-output --check-ids          # organizer validator must print PASS
```
`scripts/run_v4.sh all` runs S3 → S6 and freezes the submission (after S1, S2 and S2b); `scripts/run_v3.sh all` runs
S2b. Each stage is resumable (`train --resume <run>`, `predict --resume`). `ber learn-logodds` (label-learned
vocabularies, D-013/D-021) is only needed when `features.self_vocab` is false.

This sequence produced the best leaderboard submission (#6, public 0.982191, local holdout 0.9879); its exact code is
`submissions/06_*/code_*.zip` (git commit in `notes.md`). Other submissions and their commands are listed in
`submissions/log.md`.

Outputs: `output/matching_results.tsv`, `output/candidate_pairs.tsv`. Every run writes `artifacts/runs/<run>/`
(models as LightGBM text, predictions, `decision_params.json`, `manifest.json` with git commit, config, decisions in
effect, metrics) and appends a row to `docs/EXPERIMENTS.md`.

## Models and licenses
LightGBM (MIT) gradient-boosted trees only — the blocking cut model and two matching stages, each far below 8B
parameters. No pretrained language model. Libraries: DuckDB (MIT), PyArrow (Apache-2.0), NumPy (BSD), rapidfuzz (MIT),
anyascii (ISC), PyYAML (MIT).

## Tests
```bash
cd code/business_entity_resolution && python -m pytest -q   # unit tests + end-to-end smoke test on synthetic data
```
The smoke test runs the whole chain (incl. blocking v3, adaptation and pseudo-label paths) on synthetic data and checks
the outputs with the organizer validator.
