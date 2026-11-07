#!/usr/bin/env bash
# Step 05 - baseline: pyannote/speaker-diarization-3.1 zero-shot on the 16 AMI test meetings.
# Needs: HF_TOKEN set, model conditions accepted on Hugging Face, and pyannote.audio 3.3.2 installed
#        (requirements-pyannote.txt; on Colab the notebook creates a separate venv and sets PYANNOTE_PY).
# Time: about 10-20 min on a GPU. Meetings already done are skipped.
# Output: $EXP/ami_sys/pyannote/<meeting>.rttm
set -euo pipefail
source "$(dirname "$0")/config.sh"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set: skipping the pyannote baseline (see README, step 05)"; exit 3
fi
$PYANNOTE_PY "$SRC/run_pyannote.py" --wav "$AMI/wav" --list "$AMI/lists/test.txt" --out "$EXP/ami_sys/pyannote"
