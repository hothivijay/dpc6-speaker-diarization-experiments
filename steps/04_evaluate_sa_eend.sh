#!/usr/bin/env bash
# Step 04 - pick SA-EEND inference settings on AMI Dev (model average, window, threshold, median filter),
# FREEZE them, and write Dev and Eval RTTMs. Eval is never used for any choice.
# Time: about 5-15 min on a GPU.
# Output: $EXP/ami_sys/sa_eend/{rttm_dev, rttm_test, sa_eend_results.json}
set -euo pipefail
source "$(dirname "$0")/config.sh"

# WINDOWS (optional) overrides the window grid, e.g. WINDOWS="0 300 600" for the short synthetic smoke test
$PY "$SRC/eend_eval.py" --exp "$EXP/sa_eend_train" --feats "$FEATS" --ami "$AMI" --out "$EXP/ami_sys/sa_eend" \
    ${WINDOWS:+--windows $WINDOWS}
