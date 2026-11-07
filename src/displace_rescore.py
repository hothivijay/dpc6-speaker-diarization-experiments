#!/usr/bin/env python3
"""
displace_rescore.py - DISPLACE-2024 Track-1 eval (32 sessions): every system output re-scored per recording under
P1 (challenge rule), P2 and C25, with recording-bootstrap 95% CIs, paired tests and reference statistics.

The DISPLACE data are licensed and are NOT part of this repository. Point --ref at your own copy of the
Track-1 speaker-diarization eval RTTMs and --sys at the folder written by run_systems.py.

  python displace_rescore.py --ref DISPLACE_EVAL_RTTM_DIR --sys exp/displace_sys --out results/displace_rescore.json

Systems looked for in SYS_DIR (one sub-folder each, one RTTM per session):
  sortformer, pyannote, nemo_clus, nemo_clus_oraclevad, and optionally sel_oracle, dover
Paired comparisons (pooled-DER difference with bootstrap CI; Wilcoxon on per-recording DER):
  every system vs Sortformer (all sessions);
  NeMo-C with reference (oracle) speech regions vs NeMo-C with MarbleNet VAD (isolates speech-detection error).
Reference statistics per session: speakers, speech minutes, overlap %, total speaker minutes. From them,
  "one-speaker-per-frame floor" = (speaker time - speech time) / speaker time: the smallest P1 DER that any system
  emitting one speaker per frame can reach (it must miss every overlapped speaker beyond the first).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import der  # noqa: E402

SYSTEMS = ["sortformer", "pyannote", "nemo_clus", "nemo_clus_oraclevad", "sel_oracle", "dover"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True); ap.add_argument("--sys", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = der.read_rttm(a.ref)
    files = sorted(ref)
    hyp = {s: der.read_rttm(os.path.join(a.sys, s)) for s in SYSTEMS if os.path.isdir(os.path.join(a.sys, s))}
    full = {s: h for s, h in hyp.items() if all(f in h for f in files)}  # systems with an output for every session
    res = dict(n_files=len(files), files=files, coverage={s: sum(f in h for f in files) for s, h in hyp.items()})
    res["all"] = der.evaluate(ref, full, None, files, ("P1", "P2", "C25"), baseline="sortformer")
    if "nemo_clus_oraclevad" in hyp:
        sub = [f for f in files if f in hyp["nemo_clus_oraclevad"]]
        pair = {k: hyp[k] for k in ("nemo_clus", "nemo_clus_oraclevad", "sortformer", "pyannote") if k in hyp}
        res["oraclevad_subset"] = der.evaluate(ref, pair, None, sub, ("P1", "P2"), baseline="nemo_clus")
    # reference statistics per recording (speaker count, speech, overlap)
    import numpy as np
    stats = {}
    for f in files:
        n = int(np.ceil(max(e for _, e, _ in ref[f]) / der.STEP)) + 1
        R, labs = der.activity(ref[f], n)
        nr = R.sum(1)
        stats[f] = dict(speakers=len(labs), speech_min=float((nr > 0).sum() * der.STEP / 60),
                        overlap_pct=float(100 * (nr > 1).sum() / max(1, (nr > 0).sum())),
                        ref_speaker_min=float(nr.sum() * der.STEP / 60))
    res["ref_stats"] = stats
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("coverage", res["coverage"])
    for key in ("all", "oraclevad_subset"):
        if key not in res:
            continue
        for pn, o in res[key]["protocols"].items():
            print(f"== {key} {pn} ({len(res[key]['files'])} recordings)")
            for s, v in o.items():
                c = next((v[k] for k in v if k.startswith("vs_")), None)
                cs = f"  d {c['delta']:+.2f} [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}] p {c['p_wilcoxon']:.2g}" if c else ""
                print(f"  {s:20s} DER {v['pooled']['der']:6.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] miss {v['pooled']['miss']:5.2f} "
                      f"FA {v['pooled']['fa']:5.2f} conf {v['pooled']['conf']:5.2f} median {v['median']:.2f}{cs}")
    tot_sp = sum(v["ref_speaker_min"] for v in stats.values()); tot_s = sum(v["speech_min"] for v in stats.values())
    print(f"one-speaker-per-frame floor (P1, pooled): {100 * (tot_sp - tot_s) / tot_sp:.2f}%  "
          f"speech {tot_s / 60:.2f} h, speaker time {tot_sp / 60:.2f} h")


if __name__ == "__main__":
    main()
