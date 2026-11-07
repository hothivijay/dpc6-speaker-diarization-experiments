#!/usr/bin/env bash
# Step 08 (optional, needs the licensed DISPLACE-2024 data) - run the off-the-shelf diarizers on the 32 Track-1
# eval sessions: NeMo clustering (MarbleNet VAD), NeMo clustering with reference speech regions (oracle VAD,
# analysis only), Streaming Sortformer v2, and pyannote 3.1.
# Needs: DISPLACE_AUDIO and DISPLACE_REF set in config.sh or the environment; pip install "nemo_toolkit[asr]";
#        HF_TOKEN for pyannote. Time: several hours on a T4 GPU. Every system skips sessions that are done.
# Output: $EXP/displace_sys/<system>/<session>.rttm
set -euo pipefail
source "$(dirname "$0")/config.sh"

if [ -z "$DISPLACE_AUDIO" ] || [ -z "$DISPLACE_REF" ]; then
  echo "DISPLACE_AUDIO / DISPLACE_REF not set: skipping (the DISPLACE data are licensed and not distributed here)"; exit 3
fi
WAV16=$DATA/displace_wav16
$PY "$SRC/run_systems.py" prep --audio-dir "$DISPLACE_AUDIO" --sd-ref "$DISPLACE_REF" --out "$WAV16"
$PY "$SRC/run_systems.py" sortformer          --wav "$WAV16" --out "$EXP/displace_sys"
$PY "$SRC/run_systems.py" nemo_clus           --wav "$WAV16" --out "$EXP/displace_sys"
$PY "$SRC/run_systems.py" nemo_clus_oraclevad --wav "$WAV16" --sd-ref "$DISPLACE_REF" --out "$EXP/displace_sys"
if [ -n "${HF_TOKEN:-}" ]; then
  $PYANNOTE_PY "$SRC/run_systems.py" pyannote --wav "$WAV16" --out "$EXP/displace_sys"
fi
