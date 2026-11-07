#!/usr/bin/env bash
# Step 01 - download AMI Mix-Headset audio (170 meetings, about 10 GB) and the BUT references (only_words).
# Time: 30-60 min on Colab. Safe to re-run: finished files are skipped, so just run it again after a disconnect.
# Output: $AMI/wav, $AMI/ref/{train,dev,test}, $AMI/uem/{train,dev,test}, $AMI/lists/*.txt
set -euo pipefail
source "$(dirname "$0")/config.sh"

$PY "$SRC/ami_prep.py" --out "$AMI" --workers "${WORKERS:-8}"

echo "meetings per list:"; wc -l "$AMI"/lists/*.txt
echo "audio files: $(ls "$AMI"/wav/*.wav | wc -l)  ($(du -sh "$AMI/wav" | cut -f1))"
