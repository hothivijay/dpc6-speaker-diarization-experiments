#!/usr/bin/env bash
# Step 10 - gather the headline numbers of all steps into $RESULTS/summary.json and print a readable summary.
set -euo pipefail
source "$(dirname "$0")/config.sh"

$PY "$SRC/collect_results.py" --exp "$EXP" --results "$RESULTS"
