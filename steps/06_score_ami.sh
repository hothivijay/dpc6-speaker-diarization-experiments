#!/usr/bin/env bash
# Step 06 - score every AMI system output on Eval under P1 (no collar, overlap scored), P2 and C25,
# on 12 meetings (TNO excluded) and on all 16 meetings, with 95% CIs and the paired SA-EEND - pyannote test.
# Output: $RESULTS/ami_systems.json
set -euo pipefail
source "$(dirname "$0")/config.sh"

$PY "$SRC/ami_score.py" --ami "$AMI" --sys "$EXP/ami_sys" --out "$RESULTS/ami_systems.json"
