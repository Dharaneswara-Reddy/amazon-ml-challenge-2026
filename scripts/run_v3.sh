#!/usr/bin/env bash
# Pipeline v3 runbook (docs/PLAN_V3.md). Reuses the v2 normalized data, v2 candidates and v2 ranker in artifacts/cache.
#   scripts/run_v3.sh all        block3 train/test -> log-odds + signature vocab -> features -> train -> predict -> freeze
#   scripts/run_v3.sh from_features | from_features_test | train | predict <run_dir>
# Every step logs to artifacts/logs/v3_<step>.log (+ .json result); live view: watch -n 5 scripts/progress.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
B=.venv/bin/ber
PY=.venv/bin/python
L=artifacts/logs
mkdir -p "$L"
step() { echo "=== $(date +%T) $*"; }
run() { local name=$1; shift; step "$name"; flock artifacts/heavy.lock "$B" "$@" --log-file "$L/v3_$name.log" > "$L/v3_$name.json"; }
rundir() { $PY -c "import json;print(json.load(open('$L/v3_train.json'))['run_dir'])"; }

case "${1:?usage: run_v3.sh all|from_features|train|predict <run_dir>}" in
  all)
    run block3_train block3 --split train
    R=$($PY -c "import json;print(json.load(open('$L/v3_block3_train.json'))['recall']['recall'])")
    echo "blocking v3 recall=$R"
    $PY -c "import sys; sys.exit(0 if $R >= 0.97 else 1)" || { echo "RECALL GATE FAILED (< 0.97)"; exit 3; }
    run block3_test block3 --split test
    "$0" from_features
    ;;
  from_features)
    run learn_logodds learn-logodds
    run features_train features --split train
    run features_test features --split test
    step "prep done"
    "$0" train
    "$0" predict "$(rundir)"
    step "all done"
    ;;
  from_features_test)
    run features_test features --split test
    step "prep done"
    "$0" train
    "$0" predict "$(rundir)"
    step "all done"
    ;;
  train)
    run train train --name v3
    rundir
    ;;
  predict_resume)
    RUN=${2:?usage: run_v3.sh predict_resume <run_dir>}
    run predict predict --run "$RUN" --resume
    scripts/freeze_submission.sh "$RUN" "v3: blocking v3 (compound keys + learned cut) + signature features + per-country prior shift + lr 0.2"
    step "all done"
    ;;
  predict)
    RUN=${2:?usage: run_v3.sh predict <run_dir>}
    run predict predict --run "$RUN"
    scripts/freeze_submission.sh "$RUN" "v3: blocking v3 (compound keys + learned cut) + signature features + per-country prior shift + lr 0.2"
    ;;
  *) echo "unknown phase $1" >&2; exit 2 ;;
esac
