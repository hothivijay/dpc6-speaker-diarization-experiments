#!/usr/bin/env python3
"""
eend_eval.py - choose SA-EEND inference settings on AMI Dev, FREEZE them, then score Eval.

Selection is done on Dev only (P1 DER = no collar, overlap scored), so the Eval numbers are not tuned on Eval.

  Dev grid (Dev = 14 meetings, TNO excluded):
    model     {avg.pt (paper: last 10 epochs), avg_best.pt (10 epochs around the lowest dev loss)}
    window    {whole recording, 500, 1500, 3000 frames}   (hop = window/2, permutation linking)
    threshold {0.3, 0.4, 0.5, 0.6, 0.7}
    median    {1, 5, 11} frames

  The lowest-DER setting is frozen and used for:
    eval12 = 12 test meetings without TNO (thesis split, as in the SpeechBrain recipe)
    eval16 = all 16 test meetings (the split behind the published pyannote AMI numbers)

  python eend_eval.py --exp EXP_DIR --feats FEAT_DIR --ami AMI_DIR --out OUT_DIR

Writes OUT_DIR/rttm_dev, OUT_DIR/rttm_test and OUT_DIR/sa_eend_results.json (frozen setting, full grid, scores).
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import der  # noqa: E402
import sa_eend as S  # noqa: E402


def ids(p):
    """Meeting ids from a list file (one per line)."""
    return [l.split()[0] for l in open(p) if l.strip()]


def probs_for(model, feats, meetings, L, dev):
    """Speaker probabilities for every meeting with window L (computed once, reused for all thresholds)."""
    out = {}
    for m in meetings:
        x = np.load(os.path.join(feats, m + ".npz"))["x"]
        out[m] = S.predict(model, x, L, dev)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, help="training directory with avg.pt / avg_best.pt")
    ap.add_argument("--feats", required=True, help="feature directory (*.npz from sa_eend.py feats)")
    ap.add_argument("--ami", required=True, help="AMI directory from ami_prep.py (ref/, uem/, lists/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", nargs="+", type=int, default=[0, 500, 1500, 3000])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref_dev, ref_test = der.read_rttm(os.path.join(a.ami, "ref", "dev")), der.read_rttm(os.path.join(a.ami, "ref", "test"))
    uem_dev, uem_test = der.read_uem(os.path.join(a.ami, "uem", "dev")), der.read_uem(os.path.join(a.ami, "uem", "test"))
    dev_ids = ids(os.path.join(a.ami, "lists", "dev_noTNO.txt"))

    # ---- 1. grid search on Dev
    grid = []
    cache = {}
    for mname in ("avg.pt", "avg_best.pt"):
        mp = os.path.join(a.exp, mname)
        if not os.path.exists(mp):
            continue
        model = S.load_model(mp, dev)
        for L in a.windows:
            try:
                P = probs_for(model, a.feats, dev_ids, L, dev)
            except torch.cuda.OutOfMemoryError:  # whole-recording attention can exceed GPU memory: skip that window
                torch.cuda.empty_cache()
                print(f"{mname} window {L}: out of GPU memory, skipped", flush=True)
                continue
            cache[(mname, L)] = P
            for th, med in itertools.product([0.3, 0.4, 0.5, 0.6, 0.7], [1, 5, 11]):
                rows = [der.score_file(ref_dev[m], S.to_segments(P[m], th, med), uem_dev.get(m)) for m in dev_ids]
                d = der.pooled(rows)["der"]
                grid.append(dict(model=mname, window=L, th=th, med=med, dev_P1=d))
            best_here = min((g for g in grid if g["model"] == mname and g["window"] == L), key=lambda g: g["dev_P1"])
            print(f"{mname} window {L}: best dev P1 {best_here['dev_P1']:.2f} (th {best_here['th']}, med {best_here['med']})",
                  flush=True)

    # ---- 2. freeze the best Dev setting
    best = min(grid, key=lambda g: g["dev_P1"])
    print("FROZEN on Dev:", best, flush=True)

    # ---- 3. Dev and Eval RTTMs with the frozen setting
    model = S.load_model(os.path.join(a.exp, best["model"]), dev)
    test_ids = ids(os.path.join(a.ami, "lists", "test.txt"))
    for split, meetings in (("dev", dev_ids), ("test", test_ids)):
        od = os.path.join(a.out, "rttm_" + split)
        os.makedirs(od, exist_ok=True)
        P = cache[(best["model"], best["window"])] if split == "dev" else probs_for(model, a.feats, meetings, best["window"], dev)
        for m in meetings:
            der.write_rttm(os.path.join(od, m + ".rttm"), m, S.to_segments(P[m], best["th"], best["med"]))

    # ---- 4. score Dev, Eval-12 and Eval-16 under P1, P2 and C25
    hyp_dev, hyp_test = der.read_rttm(os.path.join(a.out, "rttm_dev")), der.read_rttm(os.path.join(a.out, "rttm_test"))
    res = dict(frozen=best, grid=grid,
               dev=der.evaluate(ref_dev, {"sa_eend": hyp_dev}, uem_dev, dev_ids, ("P1", "P2", "C25")),
               eval12=der.evaluate(ref_test, {"sa_eend": hyp_test}, uem_test, ids(os.path.join(a.ami, "lists", "test_noTNO.txt")),
                                   ("P1", "P2", "C25")),
               eval16=der.evaluate(ref_test, {"sa_eend": hyp_test}, uem_test, test_ids, ("P1", "P2", "C25")))
    json.dump(res, open(os.path.join(a.out, "sa_eend_results.json"), "w"), indent=1)
    for k in ("dev", "eval12", "eval16"):
        for pn, o in res[k]["protocols"].items():
            s = o["sa_eend"]
            print(f"{k:6s} {pn:3s} DER {s['pooled']['der']:6.2f} [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}] miss {s['pooled']['miss']:.2f} "
                  f"FA {s['pooled']['fa']:.2f} conf {s['pooled']['conf']:.2f}", flush=True)


if __name__ == "__main__":
    main()
