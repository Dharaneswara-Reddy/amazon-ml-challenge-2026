# Amazon ML Challenge 2026 — Business Entity Resolution

Hackathon window: 25 Sep 2026 00:00 → 27 Sep 2026 23:59 IST. Max 5 leaderboard uploads per day.
Given business records from 3 noisy sources, find every Source 2 / Source 3 record that refers to the same
real-world business as each Source 1 record.
Full spec: `docs/problem_statement.pdf`; event rules: `docs/guidelines.pdf`. Distilled: `docs/hackathon_notes.md`.
Non-negotiable constraints live in `.claude/rules/` — read them before writing pipeline code.

## Layout

```
dataset/{train,test}/                 organizer TSVs (never commit, never modify)
code/business_entity_resolution/      the submitted, self-contained pipeline
  src/                                all pipeline source
  README.md                           exact end-to-end reproduce steps
  requirements.txt                    pinned deps
notebooks/                            EDA and experiments (not submitted)
scripts/                              helpers: AWS sync, packaging, validation runs
output/                               matching_results.tsv + candidate_pairs.tsv
submissions/                          frozen copy of every leaderboard upload + log.md (version history)
docs/                                 problem statement, guidelines, notes
```

The final zip is assembled from `output/`, `code/business_entity_resolution/`, and the filled
`Documentation_template.md` (ships with the organizer's `student_resource/`, not here yet).

## Pipeline shape

normalize → block (candidate generation) → pairwise features → classifier → threshold → group per S1 entity.
`candidate_pairs.tsv` is the exact set the classifier scores; every final match must be in it.

## Commands

- Read TSVs with `sep="\t"` always, and `keep_default_na=False` so empty ID lists stay `""`.
- Validate before any leaderboard upload:
  `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`
- Local score = macro F0.5 on a held-out split of train, grouped by Source 1 entity (no leakage across the split).

## AWS

Region `us-east-1`. Profile comes from `AWS_PROFILE` in `.env` (see `.env.example`); pass `--profile` explicitly in CLI calls.
Work on a SageMaker Notebook Instance and train in-notebook; no endpoints. See `.claude/rules/aws-usage.md`.
