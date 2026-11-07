#!/usr/bin/env python3
"""
bench_edge.py - full-pipeline CPU benchmark of SA-EEND, FP32 vs INT8, with ONNX Runtime.

Runs unchanged on a laptop, on Colab's CPU, or on a Raspberry Pi 5. Only numpy, scipy, soundfile and onnxruntime are
needed; PyTorch is NOT imported, so the memory figure is that of the deployable pipeline.

  python bench_edge.py run   --onnx MODEL.onnx --wav WAV_DIR --list meetings.txt --out OUT_DIR/fp32 \
                             --window 1500 --th 0.5 --med 11 --threads 4
  python bench_edge.py score --ami AMI_DIR --out OUT_DIR --variants fp32 int8

'run' processes every recording end to end and times each stage:
    read audio -> log-Mel -> splice/subsample -> ONNX model on overlapping windows with permutation linking
    -> threshold + median filter -> RTTM
  and records: real-time factor (processing time / audio time), ms per 10 s of audio, model-only RTF,
  peak resident memory of the process (ru_maxrss), model file size, CPU name, thread count, ONNX Runtime version,
  and on a Raspberry Pi the SoC temperature and throttling flags (vcgencmd) before and after the run.
'score' gives DER per variant under P1/P2 and the paired INT8 - FP32 DER difference with its 95% bootstrap CI.
"""
import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cpu_name():
    """Human-readable CPU name from /proc/cpuinfo (x86: 'model name'; Raspberry Pi: 'Model'/'Hardware')."""
    info = {}
    try:
        for line in open("/proc/cpuinfo"):
            if ":" in line:
                k, v = line.split(":", 1)
                info.setdefault(k.strip().lower(), v.strip())
    except OSError:
        pass
    return info.get("model name") or info.get("model") or info.get("hardware") or platform.processor() or platform.machine()


def vcgencmd(*args):
    """Raspberry Pi firmware query (temperature, throttling). Returns None on any other machine."""
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def cmd_run(a):
    import onnxruntime as ort
    import soundfile as sf
    import eend_core as E
    from der import write_rttm
    th = a.threads or os.cpu_count()
    so = ort.SessionOptions()
    so.intra_op_num_threads = th   # threads used inside one operator (matrix multiplications)
    so.inter_op_num_threads = 1
    t_load = time.time()
    sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
    t_load = time.time() - t_load
    # model runner: (t, 345) features -> (t, 4) sigmoid probabilities
    run = lambda x: 1 / (1 + np.exp(-sess.run(None, {"feats": np.asarray(x, np.float32)[None]})[0][0]))  # noqa: E731
    os.makedirs(os.path.join(a.out, "rttm"), exist_ok=True)
    rows, temp0, thr0 = [], vcgencmd("measure_temp"), vcgencmd("get_throttled")
    ids = [l.split()[0] for l in open(a.list) if l.strip()]
    run(np.zeros((50, 345), np.float32))  # warm-up (first call allocates buffers)
    for m in ids:
        t0 = time.perf_counter()
        x, sr = sf.read(os.path.join(a.wav, m + ".Mix-Headset.wav"), dtype="float32", always_2d=True)
        x = x.mean(1)
        t1 = time.perf_counter()
        f = E.splice_subsample(E.logmel_np(x))
        t2 = time.perf_counter()
        p = E.predict_np(run, f, a.window)
        t3 = time.perf_counter()
        write_rttm(os.path.join(a.out, "rttm", m + ".rttm"), m, E.to_segments(p, a.th, a.med))
        t4 = time.perf_counter()
        dur = len(x) / sr
        rows.append(dict(recording=m, audio_s=dur, read_s=t1 - t0, feat_s=t2 - t1, model_s=t3 - t2, post_s=t4 - t3,
                         total_s=t4 - t0, rtf=(t4 - t0) / dur))
        print(f"{m}: {dur / 60:.1f} min audio, {t4 - t0:.1f} s, RTF {(t4 - t0) / dur:.4f}", flush=True)
    audio = sum(r["audio_s"] for r in rows)
    proc = sum(r["total_s"] for r in rows)
    res = dict(model=os.path.basename(a.onnx), model_mb=os.path.getsize(a.onnx) / 1e6, device=cpu_name(),
               machine=platform.machine(), threads=th, onnxruntime=ort.__version__, window=a.window,
               recordings=len(rows), audio_h=audio / 3600, processing_s=proc, rtf=proc / audio,
               ms_per_10s=1e4 * proc / audio, model_rtf=sum(r["model_s"] for r in rows) / audio,
               python_modules_torch="torch" in sys.modules,  # must be False: proves torch was not loaded
               peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, load_s=t_load,
               temp_before=temp0, temp_after=vcgencmd("measure_temp"), throttled_before=thr0,
               throttled_after=vcgencmd("get_throttled"), per_recording=rows)
    json.dump(res, open(os.path.join(a.out, "bench.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_recording"}, indent=1))


def cmd_score(a):
    import der
    ref = der.read_rttm(os.path.join(a.ami, "ref", "test"))
    uem = der.read_uem(os.path.join(a.ami, "uem", "test"))
    files = [l.split()[0] for l in open(os.path.join(a.ami, "lists", "test_noTNO.txt")) if l.strip()]
    systems = {v: der.read_rttm(os.path.join(a.out, v, "rttm")) for v in a.variants}
    res = der.evaluate(ref, systems, uem, files, ("P1", "P2"), baseline=a.variants[0])
    json.dump(res, open(os.path.join(a.out, "edge_scores.json"), "w"), indent=1)
    for pn, o in res["protocols"].items():
        for v in a.variants:
            s = o[v]
            extra = ""
            if "vs_" + a.variants[0] in s:
                c = s["vs_" + a.variants[0]]
                extra = f"  dDER {c['delta']:+.2f} [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}] p {c['p_wilcoxon']:.3g}"
            print(f"{pn} {v:6s} DER {s['pooled']['der']:.2f} [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]{extra}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("--onnx"); ap.add_argument("--wav"); ap.add_argument("--list"); ap.add_argument("--out")
    ap.add_argument("--ami"); ap.add_argument("--variants", nargs="+", default=["fp32", "int8"])
    ap.add_argument("--window", type=int, default=1500, help="model window in 100 ms frames (0 = whole recording)")
    ap.add_argument("--th", type=float, default=0.5); ap.add_argument("--med", type=int, default=11)
    ap.add_argument("--threads", type=int, default=0, help="ONNX Runtime threads (0 = all cores)")
    a = ap.parse_args()
    {"run": cmd_run, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
