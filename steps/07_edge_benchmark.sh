#!/usr/bin/env bash
# Step 07 - edge deployment check: export the frozen SA-EEND model to ONNX (FP32) and INT8 (dynamic quantisation),
# run the whole pipeline (audio -> features -> model -> RTTM) on the CPU with ONNX Runtime only, measure speed and
# memory, and score both variants to get the DER change caused by INT8.
# Time: about 10-20 min on Colab's 2 vCPUs.
# Output: $EXP/edge/sa_eend_{fp32,int8}.onnx, $EXP/edge/{fp32,int8}/bench.json, $EXP/edge/edge_scores.json
set -euo pipefail
source "$(dirname "$0")/config.sh"

# read the setting frozen on Dev in step 04 (whole-recording window -> 1500 frames, to bound memory on small devices)
read MODEL WIN TH MED < <($PY -c "
import json; f = json.load(open('$EXP/ami_sys/sa_eend/sa_eend_results.json'))['frozen']
print(f['model'], f['window'] if f['window'] > 0 else 1500, f['th'], f['med'])")
echo "edge configuration: model $MODEL, window $WIN frames, threshold $TH, median $MED"

$PY "$SRC/sa_eend.py" export --model "$EXP/sa_eend_train/$MODEL" --out "$EXP/edge/sa_eend"
for v in fp32 int8; do
  $PY "$SRC/bench_edge.py" run --onnx "$EXP/edge/sa_eend_$v.onnx" --wav "$AMI/wav" --list "$AMI/lists/test_noTNO.txt" \
      --out "$EXP/edge/$v" --window "$WIN" --th "$TH" --med "$MED" --threads "$EDGE_THREADS"
done
$PY "$SRC/bench_edge.py" score --ami "$AMI" --out "$EXP/edge" --variants fp32 int8
