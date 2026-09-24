#!/usr/bin/env bash
# Freeze output/ as the next leaderboard submission: validated TSVs + code zip of HEAD + notes + log row.
# Usage: scripts/freeze_submission.sh <run_dir> ["what changed vs previous"]
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
RUN_DIR=${1:?usage: freeze_submission.sh <run_dir> [change note]}
CHANGE=${2:-}
PY=${PY:-.venv/bin/python}

if ! git diff --quiet HEAD -- code/; then
  echo "code/ has uncommitted changes; commit first so the zip matches the outputs" >&2
  exit 1
fi
"$PY" student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test --check-ids

N=$(printf "%02d" $(( $(find submissions -maxdepth 1 -type d -name '[0-9][0-9]_*' | wc -l) + 1 )))
STAMP=$(TZ=Asia/Kolkata date +%Y%m%d-%H%M)
DIR="submissions/${N}_${STAMP}"
[ -e "$DIR" ] && { echo "$DIR exists" >&2; exit 1; }
mkdir -p "$DIR"
cp output/matching_results.tsv output/candidate_pairs.tsv "$DIR/"
SHA=$(git rev-parse --short HEAD)
git archive --format=zip -o "$DIR/code_${SHA}.zip" HEAD code/business_entity_resolution

"$PY" - "$RUN_DIR" "$DIR" "$SHA" "$N" "$STAMP" "$CHANGE" <<'EOF'
import json, sys
from pathlib import Path
run_dir, out_dir, sha, n, stamp, change = sys.argv[1:]
m = json.loads((Path(run_dir) / "manifest.json").read_text())
metrics = m.get("metrics", {})
pred = json.loads((Path(run_dir) / "predict.json").read_text()) if (Path(run_dir) / "predict.json").exists() else {}
lines = [
    f"# Submission {n} ({stamp} IST)",
    "",
    f"- Git commit: `{sha}` (code zip: `code_{sha}.zip`)",
    f"- Run dir: `{run_dir}`",
    f"- Local holdout macro F0.5: **{metrics.get('holdout_f05', float('nan')):.4f}** "
    f"(tune {metrics.get('tune_f05', float('nan')):.4f}, threshold {metrics.get('threshold')})",
    f"- Holdout candidate recall: {metrics.get('holdout_candidate_recall', float('nan')):.4f}",
    f"- Change vs previous: {change or 'first submission'}",
    f"- Output stats: {json.dumps({k: v for k, v in pred.items() if not isinstance(v, dict)}, default=str)}",
    "- Public leaderboard F0.5: _fill in after upload_",
    "",
    "Upload: `matching_results.tsv` as the matching file and the code zip as the code file.",
]
(Path(out_dir) / "notes.md").write_text("\n".join(lines) + "\n")
row = (f"| {n} | {stamp} | `{out_dir}` | `{sha}` | {metrics.get('holdout_f05', float('nan')):.4f} | | pending | "
       f"{change or 'first submission'} |\n")
with open("submissions/log.md", "a", encoding="utf-8") as fh:
    fh.write(row)
print(f"frozen {out_dir}")
EOF
ls -la "$DIR"
