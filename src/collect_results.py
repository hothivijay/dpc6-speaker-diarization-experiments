#!/usr/bin/env python3
"""
collect_results.py - gather every headline number into one compact JSON (and print a readable summary), so that each
value quoted in a paper or slide can be traced back to a result file.

  python collect_results.py --exp WORK/exp --results WORK/results

Reads (whatever exists):
  RESULTS/displace_rescore.json                DISPLACE-2024 rescoring (displace_rescore.py)
  EXP/ami_sys/sa_eend/sa_eend_results.json     SA-EEND Dev grid, frozen setting, Dev/Eval scores (eend_eval.py)
  EXP/sa_eend_train/train_log.json             training history (sa_eend.py train)
  RESULTS/ami_systems.json                     AMI Eval: SA-EEND and pyannote (ami_score.py)
  EXP/edge/{fp32,int8}/bench.json              CPU benchmark (bench_edge.py run)
  EXP/edge/edge_scores.json                    INT8 vs FP32 DER (bench_edge.py score)
Writes RESULTS/summary.json and prints a readable summary.
"""
import argparse
import json
import os


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def r2(x):
    return None if x is None else round(float(x), 2)


def sysrow(o):
    """Compact view of one system: pooled DER and components, CI, median, worst/best recording, paired tests."""
    p = o["pooled"]
    row = dict(der=r2(p["der"]), miss=r2(p["miss"]), fa=r2(p["fa"]), conf=r2(p["conf"]), ci=[r2(c) for c in o["ci95"]],
               median=r2(o["median"]), worst=o["worst"], worst_der=r2(o["worst_der"]), best_der=r2(o["best_der"]),
               n=len(o["per_recording"]), ref_h=r2(p["total_h"]))
    for k, v in o.items():
        if k.startswith("vs_"):
            row[k] = dict(delta=r2(v["delta"]), ci=[r2(c) for c in v["ci"]], p=float(f"{v['p_wilcoxon']:.3g}"),
                          better=v["better"], tied=v["tied"], worse=v["worse"])
    return row


def block(ev):
    return {pn: {s: sysrow(o) for s, o in d.items()} for pn, d in ev["protocols"].items()}


def per_rec(ev, pn, systems):
    out = {}
    for s in systems:
        if s in ev["protocols"][pn]:
            out[s] = {r["file"]: r2(r["der"]) for r in ev["protocols"][pn][s]["per_recording"]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, help="WORK/exp"); ap.add_argument("--results", required=True, help="WORK/results")
    a = ap.parse_args()
    R = a.results
    res = {}
    d = load(os.path.join(R, "displace_rescore.json"))
    if d:
        st = d["ref_stats"]
        floor = {f: r2(100 * (v["ref_speaker_min"] - v["speech_min"]) / v["ref_speaker_min"]) for f, v in st.items()}
        tot_sp = sum(v["ref_speaker_min"] for v in st.values()); tot_s = sum(v["speech_min"] for v in st.values())
        res["displace"] = dict(coverage=d["coverage"], all=block(d["all"]),
                               oraclevad_subset=block(d["oraclevad_subset"]) if "oraclevad_subset" in d else None,
                               per_recording_P1=per_rec(d["all"], "P1", ["sortformer", "pyannote", "nemo_clus", "nemo_clus_oraclevad"]),
                               floor_P1=floor, floor_P1_pooled=r2(100 * (tot_sp - tot_s) / tot_sp),
                               speech_h=r2(tot_s / 60), ref_speaker_h=r2(tot_sp / 60))
    e = load(os.path.join(a.exp, "ami_sys", "sa_eend", "sa_eend_results.json"))
    if e:
        best = {}
        for g in e["grid"]:  # best threshold / median for every (model, window) pair on Dev
            k = f"{g['model']}|{g['window']}"
            if k not in best or g["dev_P1"] < best[k]["dev_P1"]:
                best[k] = dict(th=g["th"], med=g["med"], dev_P1=r2(g["dev_P1"]))
        res["sa_eend"] = dict(frozen=e["frozen"], grid_best=best, dev=block(e["dev"]), eval12=block(e["eval12"]),
                              eval16=block(e["eval16"]))
    t = load(os.path.join(a.exp, "sa_eend_train", "train_log.json"))
    if t:
        h = t["hist"]
        bi = min(range(len(h)), key=lambda i: h[i]["dev_loss"])
        res["train"] = dict(params=t["params"], epochs=len(h), minutes=r2(sum(x["sec"] for x in h) / 60),
                            best_epoch=h[bi]["epoch"], best_dev_loss=round(h[bi]["dev_loss"], 4),
                            final_train_loss=round(h[-1]["train_loss"], 4), final_dev_loss=round(h[-1]["dev_loss"], 4),
                            averaged=t["averaged"], batch=t["batch"], chunk=t["chunk"], warmup=t["warmup"])
    s = load(os.path.join(R, "ami_systems.json"))
    if s:
        res["ami_systems"] = {k: block(v) for k, v in s.items()}
    edge = {}
    for v in ("fp32", "int8"):
        b = load(os.path.join(a.exp, "edge", v, "bench.json"))
        if b:
            edge[v] = {k: (round(x, 5) if isinstance(x, float) else x) for k, x in b.items() if k != "per_recording"}
    es = load(os.path.join(a.exp, "edge", "edge_scores.json"))
    if es:
        edge["scores"] = block(es)
    if edge:
        res["edge"] = edge
    os.makedirs(R, exist_ok=True)
    json.dump(res, open(os.path.join(R, "summary.json"), "w"), indent=1)

    # ---- readable summary
    def line(name, r):
        return f"  {name:22s} DER {r['der']:6.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]  miss {r['miss']:5.2f}  FA {r['fa']:5.2f}  conf {r['conf']:5.2f}"
    if "displace" in res:
        print("DISPLACE-2024 eval, P1 (no collar, overlap scored)")
        for k, r in res["displace"]["all"]["P1"].items():
            print(line(k, r))
        print(f"  one-speaker-per-frame floor {res['displace']['floor_P1_pooled']}%")
    if "ami_systems" in res:
        for split in ("eval12", "eval16"):
            print(f"AMI {split}, P1")
            for k, r in res["ami_systems"][split]["P1"].items():
                print(line(k, r))
    if "edge" in res and "fp32" in res["edge"]:
        for v in ("fp32", "int8"):
            b = res["edge"].get(v)
            if b:
                print(f"edge {v}: {b['model_mb']:.1f} MB, RTF {b['rtf']:.4f}, {b['ms_per_10s']:.0f} ms per 10 s, "
                      f"peak RSS {b['peak_rss_mb']:.0f} MB on {b['device']} ({b['threads']} threads)")
    print("written", os.path.join(R, "summary.json"))


if __name__ == "__main__":
    main()
