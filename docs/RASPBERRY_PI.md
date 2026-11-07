# Running the edge benchmark on a Raspberry Pi 5

`src/bench_edge.py` runs the complete SA-EEND pipeline with ONNX Runtime only: audio, log-Mel, model on
overlapping windows, permutation linking, then RTTM. It does not load PyTorch, so its memory figure is that of
the deployable pipeline. The script is the same one that produced the Colab CPU numbers.

## 1. What to copy to the Pi

From a finished run (`WORK/exp/edge/` and `WORK/data/ami/`):

```
sa_eend_fp32.onnx  sa_eend_int8.onnx          # WORK/exp/edge/
src/eend_core.py  src/bench_edge.py  src/der.py
ami/wav/<12 eval meetings>.Mix-Headset.wav     # the meetings in ami/lists/test_noTNO.txt (about 0.7 GB)
ami/ref/test  ami/uem/test  ami/lists           # references for scoring
```

## 2. Install (Raspberry Pi OS 64-bit, Python 3.11+)

```bash
python3 -m venv ~/sd && source ~/sd/bin/activate
pip install -r requirements-edge.txt      # numpy, scipy, soundfile, onnxruntime
```

## 3. Run (use the window, threshold and median filter printed by step 07 on Colab)

```bash
for v in fp32 int8; do
  python src/bench_edge.py run --onnx sa_eend_$v.onnx --wav ami/wav --list ami/lists/test_noTNO.txt \
      --out pi/$v --window 1500 --th 0.5 --med 11 --threads 4
done
python src/bench_edge.py score --ami ami --out pi --variants fp32 int8
```

Each `pi/<variant>/bench.json` records: real-time factor, ms per 10 s of audio, model-only RTF, peak RSS (MB),
model size, CPU name, threads, ONNX Runtime version, and **SoC temperature and `vcgencmd get_throttled`
before and after** the run. Report a run as valid only if `throttled_after` is `0x0`. Use the active cooler and
the official 27 W supply.

`pi/edge_scores.json` gives the INT8 − FP32 DER difference with its 95% bootstrap CI on the same 12 meetings.
