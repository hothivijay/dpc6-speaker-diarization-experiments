#!/usr/bin/env bash
# Step 09 (optional, needs DISPLACE) - score every DISPLACE system output per session under P1, P2 and C25, with
# 95% CIs, paired tests (vs Sortformer; oracle VAD vs MarbleNet VAD) and the one-speaker-per-frame floor.
# Output: $RESULTS/displace_rescore.json
set -euo pipefail
source "$(dirname "$0")/config.sh"

if [ -z "$DISPLACE_REF" ] || [ ! -d "$EXP/displace_sys" ]; then
  echo "no DISPLACE references or system outputs: skipping"; exit 3
fi
$PY "$SRC/displace_rescore.py" --ref "$DISPLACE_REF" --sys "$EXP/displace_sys" --out "$RESULTS/displace_rescore.json"
