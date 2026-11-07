#!/usr/bin/env bash
# Step 02 - SA-EEND features for every AMI meeting: 23-dim log-Mel, +-7 frames context, x10 subsampling
# -> one 345-dim vector per 100 ms, plus 4-speaker training targets from the references.
# Time: about 10-20 min on a Colab GPU. Meetings that already have a .npz are skipped.
# Output: $FEATS/<meeting>.npz
set -euo pipefail
source "$(dirname "$0")/config.sh"

# all references in one folder (the feature script looks up each meeting by id)
mkdir -p "$DATA/ami_ref_all"
cp "$AMI"/ref/*/*.rttm "$DATA/ami_ref_all/"
cat "$AMI/lists/train.txt" "$AMI/lists/dev.txt" "$AMI/lists/test.txt" > "$AMI/lists/all.txt"

$PY "$SRC/sa_eend.py" feats --wav "$AMI/wav" --rttm "$DATA/ami_ref_all" --list "$AMI/lists/all.txt" --out "$FEATS"
echo "feature files: $(ls "$FEATS"/*.npz | wc -l)"
