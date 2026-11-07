#!/usr/bin/env bash
# Step 03 - train SA-EEND on the 136 AMI train meetings (dev loss on the 14 non-TNO dev meetings).
# Time: set by MAXMIN (default 75 min on a T4 GPU). Resumable: re-running continues from last.pt.
# Output: $EXP/sa_eend_train/{avg.pt, avg_best.pt, last.pt, train_log.json}
set -euo pipefail
source "$(dirname "$0")/config.sh"

$PY "$SRC/sa_eend.py" train \
    --feats "$FEATS" \
    --train-list "$AMI/lists/train.txt" \
    --dev-list "$AMI/lists/dev_noTNO.txt" \
    --out "$EXP/sa_eend_train" \
    --epochs "$EPOCHS" --max-minutes "$MAXMIN" \
    --batch "$BATCH" --chunk "$CHUNK" --warmup "$WARMUP" --layers "$LAYERS" --avg "$AVG"
