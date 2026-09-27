#!/usr/bin/env bash
# Build the final deliverable <team_name>_submission.zip from a frozen submission folder:
#   output/{matching_results,candidate_pairs}.tsv  (from the chosen submissions/NN_*/)
#   code/business_entity_resolution/               (src, README.md, requirements*.txt, configs, ce, tests; from HEAD)
#   Documentation_template.md                      (docs/METHODOLOGY.md, the filled template)
# Usage: scripts/build_final_zip.sh submissions/NN_<stamp> <team_name>
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
SUB=${1:?usage: build_final_zip.sh <submission_dir> <team_name>}
TEAM=${2:?usage: build_final_zip.sh <submission_dir> <team_name>}
git diff --quiet HEAD -- code/ docs/METHODOLOGY.md || { echo "commit code/ and docs/METHODOLOGY.md first" >&2; exit 1; }
STAGE=$(mktemp -d)
mkdir -p "$STAGE/output" "$STAGE/code"
cp "$SUB/matching_results.tsv" "$SUB/candidate_pairs.tsv" "$STAGE/output/"
git archive HEAD code/business_entity_resolution | tar -x -C "$STAGE"
cp docs/METHODOLOGY.md "$STAGE/Documentation_template.md"
.venv/bin/python student_resource/utils/validate_submission.py --matching "$STAGE/output/matching_results.tsv" \
  --candidate "$STAGE/output/candidate_pairs.tsv" --test-dir dataset/test --check-ids | tail -2
OUT="$PWD/${TEAM}_submission.zip"
rm -f "$OUT"
(cd "$STAGE" && zip -qr "$OUT" output code Documentation_template.md)
rm -rf "$STAGE"
ls -la "$OUT"
unzip -l "$OUT" | tail -1
