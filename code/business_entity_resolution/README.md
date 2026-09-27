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
ber train --name v5

# S6 test: stage 1 → stage 2 → per-country EM prior shift (none for France, D-025) → decision → output/*.tsv
ber predict --run artifacts/runs/<run printed by train>
ber validate-output --check-ids          # organizer validator must print PASS

# S7 cross-encoder stage 3 (D-028; GPU notebooks, packages in ce/requirements-gpu.txt)
python code/business_entity_resolution/ce/export_bundle.py artifacts/runs/<run> <bundle_dir> 1500000 0.001 0.999
#   run ce/ce_colab.ipynb (multilingual-e5-small), ce/ce_kaggle_bge.ipynb (bge-reranker-v2-m3) and
#   ce/ce_kaggle_qwen.ipynb (Qwen2.5-1.5B-Instruct + LoRA, runs ce/qwen_ce.py) on a GPU with the bundle;
#   each writes ce_tune/ce_holdout/ce_test.parquet
python code/business_entity_resolution/ce/stage3.py artifacts/runs/<run> <stage3_dir> e5=<e5_out> bge=<bge_out> qwen=<qwen_out>
ber --override code/business_entity_resolution/configs/predict_bigmem.yaml predict --run artifacts/runs/<run> --stage3 <stage3_dir>
ber validate-output --check-ids

# S8 final submission (#14, D-029): e5 re-scores the pairs just outside its first band with the saved fine-tuned model
python code/business_entity_resolution/ce/export_ext.py artifacts/runs/<run> <e5_ext_bundle> 0.003 0.997 1e-4 0.9999 France India US
#   run ce/ce_colab_e5_ext.ipynb on a GPU (reads the model saved by ce_colab.ipynb) -> <e5_ext_out>/ce_*.parquet
#   India/US: stage 3 with competition context and both e5 bands
python code/business_entity_resolution/ce/stage3.py artifacts/runs/<run> <s3_iu> --ctx e5=<e5_out>+<e5_ext_out> bge=<bge_out> qwen=<qwen_out>
ber --override code/business_entity_resolution/configs/predict_bigmem.yaml predict --run artifacts/runs/<run> --stage3 <s3_iu>
mkdir -p <out_iu> && cp output/*.tsv <out_iu>/
#   France: stage 3 without context on the first-band scores (as #12)
python code/business_entity_resolution/ce/stage3.py artifacts/runs/<run> <s3_fr> e5=<e5_out> bge=<bge_out> qwen=<qwen_out>
ber --override code/business_entity_resolution/configs/predict_bigmem.yaml predict --run artifacts/runs/<run> --stage3 <s3_fr>
mkdir -p <out_fr> && cp output/*.tsv <out_fr>/
#   merge per country (decisions never cross countries) -> output/*.tsv
python code/business_entity_resolution/ce/merge_by_country.py <out_iu> <out_fr> output France
ber validate-output --check-ids
```
The e5 bundle used band 0.003 < p < 0.997 (`export_bundle.py <run> <dir> 600000`), the bge bundle 0.001 < p < 0.999,
the Qwen bundle 0.01 < p < 0.99. The final submission (#14) is v5 (`ber train --name v5`) + S7 + S8; India/US stage 3
holdout 0.98971 (#12: public 0.985004, holdout 0.98937).

`scripts/run_v4.sh all` runs S3 → S6 and freezes the submission (after S1, S2 and S2b); `scripts/run_v3.sh all` runs
S2b. Each stage is resumable (`train --resume <run>`, `predict --resume`). `ber learn-logodds` (label-learned
vocabularies, D-013/D-021) is only needed when `features.self_vocab` is false.

Submission #6 (public 0.982191, local holdout 0.9879) used S1–S6 only; each submission's exact code is
`submissions/06_*/code_*.zip` (git commit in `notes.md`). Other submissions and their commands are listed in
`submissions/log.md`.

Outputs: `output/matching_results.tsv`, `output/candidate_pairs.tsv`. Every run writes `artifacts/runs/<run>/`
(models as LightGBM text, predictions, `decision_params.json`, `manifest.json` with git commit, config, decisions in
effect, metrics) and appends a row to `docs/EXPERIMENTS.md`.

## Models and licenses
Cross-encoders (stage 3, D-028), open weights run inside the notebooks, fine-tuned only on the provided train data:
`intfloat/multilingual-e5-small` (MIT, 117.7M params), `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M),
`Qwen/Qwen2.5-1.5B-Instruct` + LoRA via PEFT (Apache-2.0, 1.54B). All far below the 8B limit; no hosted API is used.
LightGBM (MIT) gradient-boosted trees only — the blocking cut model and two matching stages, each far below 8B
parameters. No pretrained language model. Libraries: DuckDB (MIT), PyArrow (Apache-2.0), NumPy (BSD), rapidfuzz (MIT),
anyascii (ISC), PyYAML (MIT).

## Tests
```bash
cd code/business_entity_resolution && python -m pytest -q   # unit tests + end-to-end smoke test on synthetic data
```
The smoke test runs the whole chain (incl. blocking v3, adaptation and pseudo-label paths) on synthetic data and checks
the outputs with the organizer validator.
