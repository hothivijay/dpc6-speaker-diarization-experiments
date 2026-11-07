#!/usr/bin/env bash
# Step 00 - check the environment: Python packages, GPU, free disk.
# Nothing is changed. Fix anything marked MISSING with:  pip install -r requirements.txt
set -uo pipefail
source "$(dirname "$0")/config.sh"

echo "repository : $REPO"
echo "work dir   : $WORK"
$PY - <<'PY'
import importlib, sys
print("python     :", sys.version.split()[0])
for m in ["numpy", "scipy", "soundfile", "torch", "onnx", "onnxruntime", "matplotlib"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:11s}: {getattr(mod, '__version__', 'ok')}")
    except Exception as e:  # noqa: BLE001
        print(f"{m:11s}: MISSING ({e.__class__.__name__})")
try:
    import torch
    print("GPU        :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (training on CPU is very slow)")
except Exception:
    pass
PY
echo "free disk  : $(df -h --output=avail "$WORK" | tail -1) (AMI audio needs about 10 GB)"
