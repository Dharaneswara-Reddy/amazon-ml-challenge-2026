# Submission 09 (20260926-1804 IST)

- Git commit: `9e7f50a` (code zip: `code_9e7f50a.zip`)
- Run dir: `artifacts/runs/20260926T114709_10acc83_99304c0b_v5`
- Local holdout macro F0.5: **0.9882** (tune nan, threshold None)
- Holdout candidate recall: 0.9783
- Change vs previous: variant of v5 (#8): France EM prior shift re-enabled (decision only)
- Output stats: {"rule": "expected_f", "train_prior": 0.7268714919564443, "matches_per_entity_by_country": [["India", 809986, 3.329964962357374], ["US", 663106, 3.3811125219798948], ["France", 259452, 3.3883685614294743]], "matching_path": "/home/toji339/Documents/Amazon_ML_Hackathon/output/matching_results.tsv", "candidate_path": "/home/toji339/Documents/Amazon_ML_Hackathon/output/candidate_pairs.tsv", "n_rows": 1732544, "n_candidate_pairs": 18924178, "n_match_pairs": 5818380, "n_empty_candidate_rows": 205, "n_empty_match_rows": 100136}
- Public leaderboard F0.5: _fill in after upload_

Upload: `matching_results.tsv` as the matching file and the code zip as the code file.

## WARNING: src/ differs between the run commit 10acc833678a05756feec6f56577d3af04f9b583 and HEAD

 code/business_entity_resolution/src/ber/cli.py      | 8 ++++----
 code/business_entity_resolution/src/ber/pipeline.py | 2 ++
 2 files changed, 6 insertions(+), 4 deletions(-)
