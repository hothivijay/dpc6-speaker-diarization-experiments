#!/usr/bin/env python3
"""
ami_score.py - AMI Eval scoring of every available system output under P1, P2 and C25.

Systems looked for in SYS_DIR:   pyannote/*.rttm   and   sa_eend/rttm_test/*.rttm
Test sets:
  eval12 = 12 meetings, TNO excluded (thesis split)
  eval16 = all 16 test meetings (split of the published pyannote numbers)
Reports pooled DER with recording-bootstrap 95% CIs and, when pyannote is present, the paired SA-EEND - pyannote
difference (bootstrap CI + Wilcoxon).

  python ami_score.py --ami data/ami --sys exp --out results/ami_systems.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import der  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ami", required=True, help="AMI directory from ami_prep.py")
    ap.add_argument("--sys", required=True, help="directory holding pyannote/ and sa_eend/rttm_test/")
    ap.add_argument("--out", required=True, help="output JSON")
    a = ap.parse_args()
    ref = der.read_rttm(os.path.join(a.ami, "ref", "test"))
    uem = der.read_uem(os.path.join(a.ami, "uem", "test"))
    ids = lambda p: [l.split()[0] for l in open(os.path.join(a.ami, "lists", p)) if l.strip()]  # noqa: E731
    systems = {}
    for name, d in (("pyannote", "pyannote"), ("sa_eend", "sa_eend/rttm_test")):
        if os.path.isdir(os.path.join(a.sys, d)):
            systems[name] = der.read_rttm(os.path.join(a.sys, d))
    base = "pyannote" if "pyannote" in systems else None
    out = {k: der.evaluate(ref, systems, uem, ids(f), ("P1", "P2", "C25"), baseline=base)
           for k, f in (("eval12", "test_noTNO.txt"), ("eval16", "test.txt"))}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    for k, o in out.items():
        for pn, v in o["protocols"].items():
            for s, r in v.items():
                print(f"{k} {pn} {s:9s} DER {r['pooled']['der']:6.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}] "
                      f"miss {r['pooled']['miss']:.2f} FA {r['pooled']['fa']:.2f} conf {r['pooled']['conf']:.2f}")


if __name__ == "__main__":
    main()
