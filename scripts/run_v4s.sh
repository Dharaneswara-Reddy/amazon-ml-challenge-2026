#!/usr/bin/env bash
# v4 + France pseudo-label self-training (D-026), run after scripts/run_v4.sh has produced a predicted v4 run.
#   scripts/run_v4s.sh <v4_run_dir>   pseudo rows from the v4 test predictions -> retrain (stage-1 + pseudo rows) -> predict -> freeze
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
B=.venv/bin/ber
PY=.venv/bin/python
L=artifacts/logs
V4=${1:?usage: run_v4s.sh <v4_run_dir>}
step() { echo "=== $(date +%T) $*"; }
run() { local name=$1; shift; step "$name"; flock artifacts/heavy.lock "$B" "$@" --log-file "$L/v4s_$name.log" > "$L/v4s_$name.json"; }
run pseudo pseudo --run "$V4" --country France
P=$($PY -c "import json;print(json.load(open('$L/v4s_pseudo.json'))['path'])")
run train train --name v4s --extra-fit "$P"
RUN=$($PY -c "import json;print(json.load(open('$L/v4s_train.json'))['run_dir'])")
run predict predict --run "$RUN"
scripts/freeze_submission.sh "$RUN" "v4s: v4 + France pseudo-label self-training (stage-1 trained with rule/model-labelled France test pairs)"
step "all done"
