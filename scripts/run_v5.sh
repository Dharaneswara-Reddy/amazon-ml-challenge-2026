#!/usr/bin/env bash
# Pipeline v5 runbook (v4 + D-027 stage-2 entity features, initials, no admin features). Reuses v3 candidates (cand3_*) in artifacts/cache.
#   scripts/run_v5.sh all      learn-selfvocab -> features train/test (v4) -> train -> predict -> freeze
#   scripts/run_v5.sh train | predict <run_dir> | predict_resume <run_dir>
# Logs: artifacts/logs/v4_<step>.log (+ .json); live view: watch -n 5 scripts/progress.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
B=.venv/bin/ber
PY=.venv/bin/python
L=artifacts/logs
mkdir -p "$L"
step() { echo "=== $(date +%T) $*"; }
run() { local name=$1; shift; step "$name"; flock artifacts/heavy.lock "$B" "$@" --log-file "$L/v5_$name.log" > "$L/v5_$name.json"; }
rundir() { $PY -c "import json;print(json.load(open('$L/v5_train.json'))['run_dir'])"; }
MSG="v5: v4 + stage-2 entity features (D-027) + initialism features + admin features dropped (French region/departement mismatch)"

case "${1:?usage: run_v5.sh all|train|predict <run_dir>|predict_resume <run_dir>}" in
  all)
    run features_train features --split train
    run features_test features --split test
    step "prep done"
    "$0" train
    "$0" predict "$(rundir)"
    step "all done"
    ;;
  train)
    run train train --name v5
    rundir
    ;;
  predict)
    RUN=${2:?}; run predict predict --run "$RUN"; scripts/freeze_submission.sh "$RUN" "$MSG" ;;
  predict_resume)
    RUN=${2:?}; run predict predict --run "$RUN" --resume; scripts/freeze_submission.sh "$RUN" "$MSG" ;;
  *) echo "unknown phase $1" >&2; exit 2 ;;
esac
