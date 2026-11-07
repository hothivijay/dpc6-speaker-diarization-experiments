#!/usr/bin/env bash
# run_all.sh - run every step in order. A finished step leaves $WORK/state/<step>.done and is skipped next time,
# so after a Colab disconnect you simply run this script again.
#
#   bash run_all.sh                 # all steps
#   bash run_all.sh 03 04           # only the steps whose names start with 03 and 04
#   FORCE=1 bash run_all.sh 04      # run step 04 again even if it is marked done
#
# Logs: $WORK/logs/<step>.log ; one line per step in $WORK/logs/progress.txt
# A step that exits with code 3 was skipped because its inputs are missing (no HF_TOKEN, no DISPLACE data).
set -uo pipefail
cd "$(dirname "$0")"
source steps/config.sh
STATE=$WORK/state; LOG=$WORK/logs; mkdir -p "$STATE" "$LOG"

say() { echo "$(date '+%F %T') $*" | tee -a "$LOG/progress.txt"; }

STEPS=$(ls steps/[0-9][0-9]_*.sh | xargs -n1 basename | sed 's/\.sh$//')
if [ $# -gt 0 ]; then   # keep only the requested steps
  SEL=""; for p in "$@"; do SEL="$SEL $(echo "$STEPS" | grep "^$p")"; done; STEPS=$SEL
fi

for st in $STEPS; do
  if [ -f "$STATE/$st.done" ] && [ -z "${FORCE:-}" ]; then say "skip  $st (done)"; continue; fi
  say "start $st"
  bash "steps/$st.sh" 2>&1 | tee "$LOG/$st.log"; rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    touch "$STATE/$st.done"; say "done  $st"
  elif [ "$rc" -eq 3 ]; then
    say "skip  $st (its inputs are not available; not marked done)"
  else
    say "FAILED $st - see $LOG/$st.log; fix the problem and run this script again"; exit 1
  fi
done
say "all requested steps finished"
