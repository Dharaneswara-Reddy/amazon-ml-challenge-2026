#!/usr/bin/env bash
# Pipeline v4 runbook (D-025 self-labelled per-country vocabularies). Reuses v3 candidates (cand3_*) in artifacts/cache.
#   scripts/run_v4.sh all      learn-selfvocab -> features train/test (v4) -> train -> predict -> freeze
#   scripts/run_v4.sh train | predict <run_dir> | predict_resume <run_dir>
# Logs: artifacts/logs/v4_<step>.log (+ .json); live view: watch -n 5 scripts/progress.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
B=.venv/bin/ber
PY=.venv/bin/python
L=artifacts/logs
mkdir -p "$L"
step() { echo "=== $(date +%T) $*"; }
run() { local name=$1; shift; step "$name"; flock artifacts/heavy.lock "$B" "$@" --log-file "$L/v4_$name.log" > "$L/v4_$name.json"; }
rundir() { $PY -c "import json;print(json.load(open('$L/v4_train.json'))['run_dir'])"; }
MSG="v4: v3 + self-labelled per-country vocabularies (France learned from test, no labels) + no EM shift for France"

case "${1:?usage: run_v4.sh all|train|predict <run_dir>|predict_resume <run_dir>}" in
  all)
    run learn_selfvocab learn-selfvocab
    run features_train features --split train
    run features_test features --split test
    step "prep done"
    "$0" train
    "$0" predict "$(rundir)"
    step "all done"
    ;;
  train)
    run train train --name v4
    rundir
    ;;
  predict)
    RUN=${2:?}; run predict predict --run "$RUN"; scripts/freeze_submission.sh "$RUN" "$MSG" ;;
  predict_resume)
    RUN=${2:?}; run predict predict --run "$RUN" --resume; scripts/freeze_submission.sh "$RUN" "$MSG" ;;
  *) echo "unknown phase $1" >&2; exit 2 ;;
esac
