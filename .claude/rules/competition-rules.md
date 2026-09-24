# Competition rules (disqualification-level)

These come from `docs/problem_statement.pdf`. Breaking any of them gets the submission rejected or the team disqualified.

## No external data, ever
The pipeline may use ONLY the provided train/test TSVs. Forbidden inside the pipeline:
- entity-resolution APIs or services
- business registries / government databases
- geocoding or address-normalization APIs
- any web lookup or internet data augmentation (including LLM calls to hosted APIs to "look up" a business)

Abbreviation maps, stopword lists, and normalization rules must be hand-written or derived from the training data.
Pretrained open models (e.g. a sentence-embedding model) are models, not data lookups — allowed if they satisfy the model rule below.

## Fair play
One registration per person. No multiple IDs, no sharing code or outputs with other teams, no plagiarism — instant disqualification.
Only one person logs into the portal per device; never share portal sessions.

## Submission budget and history
- Max 5 leaderboard uploads per day, window ends 27 Sep 2026 23:59 IST. Never upload without passing `validate_submission.py`
  and a local F0.5 improvement worth spending a slot on.
- Every upload gets a folder `submissions/<NN>_<YYYYMMDD-HHMM>/` holding the exact `matching_results.tsv`,
  `candidate_pairs.tsv`, and a `notes.md` (code version, config, local F0.5, public score once known). Append a row to
  `submissions/log.md`. Never overwrite a past submission folder.

## Model rule
Every model used for the final predictions must be MIT or Apache-2.0 licensed and ≤ 8B parameters.
Check the license on the model card before adopting any pretrained model; record it in the methodology doc.

## Output format (validator enforces)
- Two TSVs in `output/`: `matching_results.tsv` (`source1_entity_id`, `matched_entity_ids`) and
  `candidate_pairs.tsv` (`source1_entity_id`, `candidate_entity_ids`).
- Exactly one row per test Source 1 entity, including France and every other country — never filter by country.
- ID lists: comma-separated, no quotes, no spaces, no duplicates, only S2-/S3- IDs that exist in the test set. Empty string for no match.
- Every matched ID must also appear in that entity's candidate list.

## Metric
Macro F0.5 per Source 1 entity. Precision counts 2× recall. A singleton scores 1.0 for an empty prediction and 0.0 for any match,
so when unsure, predict nothing. Tune thresholds for F0.5, not accuracy or F1.

## Generalization
Test adds France, unseen in training. No country one-hot, no country-specific hard-coding that only covers US/India;
normalization must degrade gracefully on French names and addresses.

## Reproducibility
`code/business_entity_resolution/` must regenerate both outputs from the raw TSVs using only its README and pinned
`requirements.txt`. Fix random seeds. No manual steps, no notebook-only logic in the final path.
