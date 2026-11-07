#!/usr/bin/env python3
"""
make_synthetic_ami.py - a tiny fake "AMI" folder (same layout as ami_prep.py writes) for the smoke test.

Each 90 s "meeting" has 4 speakers, each a modulated tone at its own pitch, talking in random turns with overlaps,
plus a little noise. It is NOT speech; it only exercises every step of the pipeline in a few minutes on a CPU.

  python tests/make_synthetic_ami.py --out /tmp/smoke/data/ami
"""
import argparse
import os

import numpy as np
import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=24)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for d in ["wav", "ref/train", "ref/dev", "ref/test", "uem/train", "uem/dev", "uem/test", "lists"]:
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    # same structure as AMI: TS* meetings play the role of the TNO meetings left out of the *_noTNO lists
    sets = {"train": [f"TR{i:03d}" for i in range(a.train)], "dev": ["DV001", "DV002", "TS0001"],
            "test": ["TE001", "TE002", "TS0002"]}
    for s, ms in sets.items():
        open(os.path.join(a.out, "lists", f"{s}.txt"), "w").write("\n".join(ms) + "\n")
        if s != "train":
            open(os.path.join(a.out, "lists", f"{s}_noTNO.txt"), "w").write("\n".join(m for m in ms if not m.startswith("TS")) + "\n")
        for m in ms:
            dur = 90.0
            n = int(dur * 16000)
            x = np.zeros(n, np.float32)
            t = np.arange(n) / 16000
            f0 = rng.uniform(100, 300, size=4)
            lines = []
            for j in range(4):
                s0 = 0.0
                while s0 < dur:
                    d = rng.uniform(0.5, 4)
                    if rng.random() < 0.35:
                        i0, i1 = int(s0 * 16000), int(min(dur, s0 + d) * 16000)
                        x[i0:i1] += 0.1 * np.sin(2 * np.pi * f0[j] * t[i0:i1]) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t[i0:i1]))
                        lines.append(f"SPEAKER {m} 1 {s0:.2f} {min(d, dur - s0):.2f} <NA> <NA> S{j} <NA> <NA>")
                    s0 += d
            x += 0.01 * rng.normal(size=n).astype(np.float32)
            sf.write(os.path.join(a.out, "wav", f"{m}.Mix-Headset.wav"), x, 16000)
            open(os.path.join(a.out, "ref", s, f"{m}.rttm"), "w").write("\n".join(lines) + "\n")
            open(os.path.join(a.out, "uem", s, f"{m}.uem"), "w").write(f"{m} 1 0.000 {dur:.3f}\n")
    print("synthetic AMI written to", a.out)


if __name__ == "__main__":
    main()
