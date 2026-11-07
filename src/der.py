#!/usr/bin/env python3
"""
der.py - Diarization Error Rate (DER) under a declared scoring protocol, per recording, with uncertainty.

WHY THIS FILE EXISTS
  A DER number means little unless you also say (a) whether a "collar" around reference boundaries is forgiven and
  (b) whether overlapped speech is scored. The same system output can score 23% or 6% depending on those two
  switches (see results/RESULTS.md). This scorer makes the protocol explicit and reports every recording separately.

PROTOCOLS
  P1   no collar, overlapped speech scored               (DISPLACE challenge rule; the default we report)
  P2   0.25 s collar each side, overlapped speech ignored (older SpeechBrain AMI recipe, "legacy" numbers)
  C25  0.25 s collar each side, overlapped speech scored  (shown for reference only)

HOW DER IS COMPUTED (per recording)
  1. Reference and hypothesis segments are turned into 10 ms frame-level activity matrices (frames x speakers).
  2. Frames outside the UEM (if one is given), inside the collar, and (P2 only) with >1 reference speaker are removed.
  3. One optimal one-to-one speaker mapping is found with the Hungarian algorithm (maximises jointly active time).
  4. For every frame with N_ref reference speakers and N_hyp hypothesis speakers:
        missed       = max(N_ref - N_hyp, 0)
        false alarm  = max(N_hyp - N_ref, 0)
        confusion    = min(N_ref, N_hyp) - (number of correctly mapped speakers)
  5. DER = (missed + false alarm + confusion) / total reference speaker time.
  This is the NIST md-eval definition; on AMI and DISPLACE it agrees with pyannote.metrics (tests/test_der.py).

UNCERTAINTY
  * Pooled DER over a set of recordings = sum of all errors / sum of all reference time (not the mean of per-file DERs).
  * 95% confidence interval: bootstrap over recordings (resample recordings with replacement, B = 10000).
  * Paired comparison of two systems on the same recordings: bootstrap CI of the pooled-DER difference and a
    Wilcoxon signed-rank test on the per-recording DER differences.

USAGE
  python der.py --ref REF_DIR_or_RTTM --sys NAME=DIR_or_RTTM [NAME2=...] [--uem UEM_DIR] [--protocols P1 P2 C25]
                [--files list.txt] [--baseline NAME] --out results.json [--csv per_recording.csv]

  example:
  python src/der.py --ref data/ami/ref/test --uem data/ami/uem/test --files data/ami/lists/test_noTNO.txt \
      --sys sa_eend=exp/sa_eend/rttm_test pyannote=exp/pyannote --baseline pyannote --out results/ami.json
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

STEP = 0.01  # frame length used for scoring: 10 ms

# protocol name -> keyword arguments of score_file()
PROTOCOLS = {
    "P1": dict(collar=0.0, ignore_overlap=False),
    "P2": dict(collar=0.25, ignore_overlap=True),
    "C25": dict(collar=0.25, ignore_overlap=False),
}


# ----------------------------------------------------------------------------------------------- I/O
def read_rttm(path):
    """Read one RTTM file, or every *.rttm file in a directory.

    RTTM line format (space separated):  SPEAKER <file> <chan> <start> <duration> <NA> <NA> <speaker> <NA> <NA>
    Returns {file_id: [(start_s, end_s, speaker_label), ...]}. Zero-length segments are dropped.
    """
    paths = sorted(glob.glob(os.path.join(path, "*.rttm"))) if os.path.isdir(path) else [path]
    out = {}
    for p in paths:
        for line in open(p):
            q = line.split()
            if len(q) >= 8 and q[0] == "SPEAKER":
                s, d = float(q[3]), float(q[4])
                if d > 0:
                    out.setdefault(q[1], []).append((s, s + d, q[7]))
    return out


def read_uem(path):
    """Read one UEM file, or every *.uem file in a directory.

    UEM line format:  <file> <chan> <start> <end>   (the region of the recording that is scored)
    Returns {file_id: [(start_s, end_s), ...]}; an empty dict when path is None (score everything).
    """
    if path is None:
        return {}
    paths = sorted(glob.glob(os.path.join(path, "*.uem"))) if os.path.isdir(path) else [path]
    out = {}
    for p in paths:
        for line in open(p):
            q = line.split()
            if len(q) >= 4:
                out.setdefault(q[0], []).append((float(q[2]), float(q[3])))
    return out


def write_rttm(path, fid, segs):
    """Write [(start, end, label), ...] of one recording as an RTTM file (sorted, empty segments skipped)."""
    with open(path, "w") as f:
        for s, e, lab in sorted(segs):
            if e - s > 1e-6:
                f.write(f"SPEAKER {fid} 1 {s:.3f} {e - s:.3f} <NA> <NA> {lab} <NA> <NA>\n")


# ----------------------------------------------------------------------------------------------- scoring
def activity(segs, n):
    """Segments -> boolean activity matrix of shape (n frames, n speakers) and the sorted speaker labels.

    Frame i covers [i * 10 ms, (i + 1) * 10 ms). Segment boundaries are rounded to the nearest frame.
    """
    labs = sorted({l for _, _, l in segs})
    idx = {l: i for i, l in enumerate(labs)}
    a = np.zeros((n, max(1, len(labs))), dtype=bool)
    for s, e, l in segs:
        i0, i1 = int(round(max(0.0, s) / STEP)), int(round(e / STEP))
        if i1 > i0:
            a[i0:min(i1, n), idx[l]] = True
    if not labs:
        a = a[:, :0]  # no speakers at all -> zero columns
    return a, labs


def score_file(ref, hyp, uem=None, collar=0.0, ignore_overlap=False):
    """Score one recording.

    ref, hyp : [(start, end, label), ...]          (labels of ref and hyp do not need to match)
    uem      : [(start, end), ...] or None         (None = score the whole recording)
    collar   : seconds forgiven on EACH side of every reference boundary
    ignore_overlap : True removes frames where more than one reference speaker talks (protocol P2)

    Returns seconds of missed speech, false alarm, speaker confusion and total reference speaker time, plus the
    number of reference and hypothesis speakers. Use der() to turn it into a percentage.
    """
    # 1. frame grids long enough to hold reference, hypothesis and UEM
    end = max([e for _, e, _ in ref] + [e for _, e, _ in hyp] + [e for _, e in (uem or [])] + [0.0])
    n = int(np.ceil(end / STEP)) + 1
    R, rl = activity(ref, n)
    H, hl = activity(hyp, n)

    # 2. which frames are scored: UEM region, minus collars, minus overlap (P2 only)
    mask = np.ones(n, dtype=bool)
    if uem:
        mask[:] = False
        for s, e in uem:
            mask[int(round(s / STEP)):int(round(e / STEP))] = True
    if collar > 0:
        c = int(round(collar / STEP))
        for s, e, _ in ref:
            for t in (s, e):
                i = int(round(t / STEP))
                mask[max(0, i - c):i + c] = False
    nr = R.sum(1)  # number of active reference speakers per frame
    if ignore_overlap:
        mask &= nr <= 1
    R, H, nr = R[mask], H[mask], nr[mask]
    nh = H.sum(1)  # number of active hypothesis speakers per frame

    # 3. optimal one-to-one speaker mapping (Hungarian algorithm on jointly active time)
    correct = np.zeros(len(nr), dtype=np.int64)
    if R.shape[1] and H.shape[1]:
        co = R.T.astype(np.float32) @ H.astype(np.float32)  # co[i, j] = frames where ref i and hyp j are both active
        ri, hi = linear_sum_assignment(-co)  # maximise total overlap
        for i, j in zip(ri, hi):
            correct += (R[:, i] & H[:, j])

    # 4. NIST error components, counted per frame
    miss = np.maximum(nr - nh, 0).sum()
    fa = np.maximum(nh - nr, 0).sum()
    conf = (np.minimum(nr, nh) - correct).sum()
    return dict(miss=miss * STEP, fa=fa * STEP, conf=conf * STEP, total=nr.sum() * STEP,
                n_ref=len(rl), n_hyp=len(hl))


def der(d):
    """DER in percent from a score_file() result (NaN if the recording has no reference speech)."""
    return 100.0 * (d["miss"] + d["fa"] + d["conf"]) / d["total"] if d["total"] > 0 else float("nan")


# ----------------------------------------------------------------------------------------------- statistics
def pooled(rows):
    """Pooled DER over several recordings = total error time / total reference time (long recordings weigh more)."""
    t = {k: sum(r[k] for r in rows) for k in ("miss", "fa", "conf", "total")}
    return dict(der=der(t), miss=100 * t["miss"] / t["total"], fa=100 * t["fa"] / t["total"],
                conf=100 * t["conf"] / t["total"], total_h=t["total"] / 3600)


def bootstrap_ci(rows, B=10000, seed=0):
    """95% CI of the pooled DER: resample the RECORDINGS with replacement B times (fixed seed: reproducible)."""
    rng = np.random.default_rng(seed)
    err = np.array([r["miss"] + r["fa"] + r["conf"] for r in rows])
    tot = np.array([r["total"] for r in rows])
    idx = rng.integers(0, len(rows), size=(B, len(rows)))
    d = 100 * err[idx].sum(1) / tot[idx].sum(1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def paired(rows_a, rows_b, B=10000, seed=0):
    """Compare system b with system a on the SAME recordings (same order).

    delta = pooled DER(b) - pooled DER(a)  (negative = b is better), with a paired recording-bootstrap 95% CI,
    a Wilcoxon signed-rank p-value on per-recording differences, and counts of recordings where b is
    better / tied / worse.
    """
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(seed)
    ea = np.array([r["miss"] + r["fa"] + r["conf"] for r in rows_a])
    eb = np.array([r["miss"] + r["fa"] + r["conf"] for r in rows_b])
    ta = np.array([r["total"] for r in rows_a])  # same reference, so the same total for a and b
    idx = rng.integers(0, len(ta), size=(B, len(ta)))
    d = 100 * (eb[idx].sum(1) - ea[idx].sum(1)) / ta[idx].sum(1)
    point = 100 * (eb.sum() - ea.sum()) / ta.sum()
    per = np.array([der(b) - der(a) for a, b in zip(rows_a, rows_b)])
    try:
        p = float(wilcoxon(per).pvalue) if np.any(per != 0) else 1.0
    except ValueError:
        p = 1.0
    return dict(delta=float(point), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], p_wilcoxon=p,
                better=int((per < 0).sum()), tied=int((per == 0).sum()), worse=int((per > 0).sum()))


# ----------------------------------------------------------------------------------------------- main
def evaluate(ref, systems, uem=None, files=None, protocols=("P1", "P2"), baseline=None):
    """Score several systems on the same recordings under several protocols.

    ref      : {file: segments}            systems: {name: {file: segments}}
    files    : recordings to score (default: every recording in ref). A recording missing from a system's output
               counts as all-missed speech for that system (and is listed under "missing").
    baseline : if given, every other system gets a "vs_<baseline>" paired comparison.
    """
    files = sorted(files or ref)
    res = {"files": files, "protocols": {}}
    for pn in protocols:
        pr = PROTOCOLS[pn]
        out = {}
        for name, hyp in systems.items():
            rows = []
            for f in files:
                r = score_file(ref[f], hyp.get(f, []), (uem or {}).get(f), **pr)
                r["file"] = f
                r["der"] = der(r)
                rows.append(r)
            dv = np.array([r["der"] for r in rows])
            out[name] = dict(pooled=pooled(rows), ci95=bootstrap_ci(rows), per_recording=rows,
                             median=float(np.median(dv)), worst=rows[int(np.argmax(dv))]["file"],
                             worst_der=float(dv.max()), best_der=float(dv.min()),
                             missing=[f for f in files if f not in hyp])
        if baseline and baseline in out:
            for name in out:
                if name != baseline:
                    out[name]["vs_" + baseline] = paired(out[baseline]["per_recording"], out[name]["per_recording"])
        res["protocols"][pn] = out
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="reference RTTM file or directory")
    ap.add_argument("--sys", nargs="+", required=True, help="one or more NAME=PATH (RTTM file or directory)")
    ap.add_argument("--uem", help="UEM file or directory (optional)")
    ap.add_argument("--files", help="text file with one recording id per line (optional)")
    ap.add_argument("--protocols", nargs="+", default=["P1", "P2"], choices=list(PROTOCOLS))
    ap.add_argument("--baseline", help="system name to compare every other system against")
    ap.add_argument("--out", required=True, help="output JSON with all numbers")
    ap.add_argument("--csv", help="optional per-recording CSV")
    a = ap.parse_args()
    ref = read_rttm(a.ref)
    files = [l.split()[0] for l in open(a.files) if l.strip()] if a.files else None
    systems = {s.split("=", 1)[0]: read_rttm(s.split("=", 1)[1]) for s in a.sys}
    res = evaluate(ref, systems, read_uem(a.uem), files, a.protocols, a.baseline)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    # human-readable summary
    for pn, out in res["protocols"].items():
        print(f"== {pn} ({len(res['files'])} recordings)")
        for name, o in out.items():
            p = o["pooled"]
            line = (f"{name:22s} DER {p['der']:6.2f} [{o['ci95'][0]:.2f}, {o['ci95'][1]:.2f}]  miss {p['miss']:5.2f} "
                    f"FA {p['fa']:5.2f} conf {p['conf']:5.2f}  median {o['median']:6.2f}  worst {o['worst']} {o['worst_der']:.2f}")
            if a.baseline and "vs_" + a.baseline in o:
                c = o["vs_" + a.baseline]
                line += f"  d {c['delta']:+.2f} [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}] p {c['p_wilcoxon']:.3g}"
            print(line)
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["protocol", "system", "recording", "der", "miss_s", "fa_s", "conf_s", "ref_speaker_s", "n_ref", "n_hyp"])
            for pn, out in res["protocols"].items():
                for name, o in out.items():
                    for r in o["per_recording"]:
                        w.writerow([pn, name, r["file"], f"{r['der']:.3f}", f"{r['miss']:.2f}", f"{r['fa']:.2f}",
                                    f"{r['conf']:.2f}", f"{r['total']:.2f}", r["n_ref"], r["n_hyp"]])


if __name__ == "__main__":
    sys.exit(main())
