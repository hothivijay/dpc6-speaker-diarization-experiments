#!/usr/bin/env python3
"""
run_pyannote.py - pyannote/speaker-diarization-3.1 (pyannote.audio 3.3.2), zero-shot, on AMI Mix-Headset.

The pipeline uses its own speech detection and estimates the number of speakers itself (nothing oracle).
One RTTM per meeting is written; meetings already done are skipped.

Published reference value for exactly this setting (pyannote model card: AMI headset mix, only_words references,
no collar, overlap scored): 18.8% DER. Reproducing it checks that our data preparation and scorer are right.

BEFORE RUNNING
  1. Create a Hugging Face access token (huggingface.co/settings/tokens).
  2. Accept the user conditions of pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0 on Hugging Face.
  3. Install the pinned environment: pip install -r requirements-pyannote.txt (a separate venv is recommended,
     because pyannote.audio 3.3.2 needs torch 2.5.1 and numpy < 2).

  HF_TOKEN=hf_xxx python run_pyannote.py --wav data/ami/wav --list data/ami/lists/test.txt --out exp/pyannote
"""
import argparse
import os
import time

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", required=True); ap.add_argument("--list", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="pyannote/speaker-diarization-3.1")
    a = ap.parse_args()
    tok = os.environ["HF_TOKEN"]
    try:  # the keyword changed between pyannote versions
        pipe = Pipeline.from_pretrained(a.model, use_auth_token=tok)
    except TypeError:
        pipe = Pipeline.from_pretrained(a.model, token=tok)
    pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(a.out, exist_ok=True)
    for m in [l.split()[0] for l in open(a.list) if l.strip()]:
        dst = os.path.join(a.out, m + ".rttm")
        if os.path.exists(dst):
            continue
        t0 = time.time()
        x, sr = sf.read(os.path.join(a.wav, m + ".Mix-Headset.wav"), dtype="float32", always_2d=True)
        # pass the waveform in memory (avoids torchaudio backend issues with file paths)
        res = pipe({"waveform": torch.from_numpy(np.ascontiguousarray(x.mean(1)))[None], "sample_rate": sr})
        ann = getattr(res, "speaker_diarization", res)
        with open(dst, "w") as f:
            for seg, _, lab in ann.itertracks(yield_label=True):
                f.write(f"SPEAKER {m} 1 {seg.start:.3f} {seg.duration:.3f} <NA> <NA> {lab} <NA> <NA>\n")
        print(f"pyannote {m}: {len(ann.labels())} speakers, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
