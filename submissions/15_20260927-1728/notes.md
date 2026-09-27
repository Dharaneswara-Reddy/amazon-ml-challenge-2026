# Submission 15 (20260927-1728 IST)

- Git commit: `6bf7b2c` (code zip: `code_6bf7b2c.zip`)
- Run dir: `artifacts/runs/20260926T114709_10acc83_99304c0b_v5`
- Local holdout macro F0.5: **0.9882** (tune nan, threshold None)
- Holdout candidate recall: 0.9783
- Change vs previous: France probe: #14 + per-country EM prior shift re-enabled for France (logit -0.53; removes 3,810 French pairs, 4.3% generator-offset vs 0.14% base); India/US identical to #14. Build: predict --stage3 stage3_final_fr with configs/variant_fr_shift.yaml, then merge_by_country
- Output stats: {"rule": "expected_f", "train_prior": 0.7268714919564443, "matches_per_entity_by_country": [["India", 809986, 3.329964962357374], ["US", 663106, 3.3811125219798948], ["France", 259452, 3.420902517614048]], "matching_path": "/home/toji339/Documents/Amazon_ML_Hackathon/output/matching_results.tsv", "candidate_path": "/home/toji339/Documents/Amazon_ML_Hackathon/output/candidate_pairs.tsv", "n_rows": 1732544, "n_candidate_pairs": 18924178, "n_match_pairs": 5826821, "n_empty_candidate_rows": 205, "n_empty_match_rows": 99733}
- Public leaderboard F0.5: _fill in after upload_

Upload: `matching_results.tsv` as the matching file and the code zip as the code file.

## WARNING: src/ differs between the run commit 10acc833678a05756feec6f56577d3af04f9b583 and HEAD

 code/business_entity_resolution/src/ber/cli.py     | 11 +++++-----
 .../business_entity_resolution/src/ber/pipeline.py | 24 ++++++++++++++++------
 2 files changed, 24 insertions(+), 11 deletions(-)
