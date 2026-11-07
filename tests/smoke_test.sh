#!/usr/bin/env bash
# smoke_test.sh - run the whole AMI pipeline (steps 02-07 and 10) on a tiny synthetic data set, on CPU, in a few
# minutes. Use it to check an installation before downloading 10 GB of AMI audio.
#
#   bash tests/smoke_test.sh            # work files go to /tmp/dpc6_smoke (deleted at the start of every run)
#
# The DER values it prints are meaningless (the "speakers" are tones); it only proves every step runs end to end.
set -euo pipefail
cd "$(dirname "$0")/.."
export WORK=${SMOKE_WORK:-/tmp/dpc6_smoke}
rm -rf "$WORK"
# small settings so that it finishes quickly on a CPU
export EPOCHS=6 MAXMIN=10 BATCH=8 CHUNK=300 WARMUP=200 LAYERS=2 AVG=3 WINDOWS="0 300 600" EDGE_THREADS=2
source steps/config.sh

python3 tests/make_synthetic_ami.py --out "$AMI"
for st in 02_extract_features 03_train_sa_eend 04_evaluate_sa_eend 06_score_ami 07_edge_benchmark 10_collect_results; do
  echo "================ $st"
  bash "steps/$st.sh"
done
echo "SMOKE TEST PASSED"
