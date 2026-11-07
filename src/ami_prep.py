#!/usr/bin/env python3
"""
ami_prep.py - download AMI Mix-Headset audio and the BUT diarization references.

  python ami_prep.py --out data/ami [--sets train dev test] [--workers 6]

DATA
  * Audio: AMI Meeting Corpus, headset mix (one 16 kHz mono file per meeting), from the official Edinburgh mirror.
    The AMI corpus is released under CC BY 4.0. About 10 GB for all 170 meetings.
  * References: BUT "AMI-diarization-setup" (github.com/BUTSpeechFIT/AMI-diarization-setup), "only_words" RTTMs
    (speech = word-aligned regions) and UEMs for the Full-corpus-ASR partition. These are the references used
    for the published pyannote AMI results.

OUTPUT
  OUT/wav/<meeting>.Mix-Headset.wav
  OUT/ref/{train,dev,test}/<meeting>.rttm      OUT/uem/{train,dev,test}/<meeting>.uem
  OUT/lists/train.txt (136)  dev.txt (18)  test.txt (16)  dev_noTNO.txt (14)  test_noTNO.txt (12)

  The TNO meetings (ids starting with "TS") are left out of dev_noTNO / test_noTNO, as in the SpeechBrain AMI recipe
  that our earlier published results used. Both versions of the test set are scored (eval12 and eval16).

Downloads are resumable: a file whose size equals the server's Content-Length is skipped; partial files are
written as *.part and renamed only when complete.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/{m}/audio/{m}.Mix-Headset.wav"
BUT = "https://github.com/BUTSpeechFIT/AMI-diarization-setup"


def fetch(m, wavdir, tries=4):
    """Download one meeting (with retries). Returns (meeting, size in bytes or -1, status)."""
    dst = os.path.join(wavdir, f"{m}.Mix-Headset.wav")
    url = URL.format(m=m)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, method="HEAD")
            size = int(urllib.request.urlopen(req, timeout=60).headers.get("Content-Length", -1))
            if os.path.exists(dst) and os.path.getsize(dst) == size:
                return m, size, "cached"
            tmp = dst + ".part"
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 22)
            if size > 0 and os.path.getsize(tmp) != size:
                raise IOError(f"size {os.path.getsize(tmp)} != {size}")
            os.replace(tmp, dst)
            return m, size, "ok"
        except Exception as e:  # noqa: BLE001  (network errors of any kind: wait and retry)
            err = e
            time.sleep(5 * (k + 1))
    return m, -1, f"failed: {err}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory (about 10 GB for all sets)")
    ap.add_argument("--sets", nargs="+", default=["train", "dev", "test"], help="which audio to download")
    ap.add_argument("--workers", type=int, default=6, help="parallel downloads")
    ap.add_argument("--but", default=None, help="existing clone of AMI-diarization-setup (default: OUT/AMI-diarization-setup)")
    a = ap.parse_args()
    but = a.but or os.path.join(a.out, "AMI-diarization-setup")
    # 1. references and lists (small: a git clone of the BUT setup)
    if not os.path.isdir(but):
        subprocess.run(["git", "clone", "-q", "--depth", "1", BUT, but], check=True)
    for d in ("wav", "ref", "uem", "lists"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    meetings = []
    for s in ("train", "dev", "test"):
        ms = [l.strip() for l in open(os.path.join(but, "lists", f"{s}.meetings.txt")) if l.strip()]
        open(os.path.join(a.out, "lists", f"{s}.txt"), "w").write("\n".join(ms) + "\n")
        if s != "train":
            open(os.path.join(a.out, "lists", f"{s}_noTNO.txt"), "w").write(
                "\n".join(m for m in ms if not m.startswith("TS")) + "\n")
        shutil.copytree(os.path.join(but, "only_words", "rttms", s), os.path.join(a.out, "ref", s), dirs_exist_ok=True)
        shutil.copytree(os.path.join(but, "uems", s), os.path.join(a.out, "uem", s), dirs_exist_ok=True)
        if s in a.sets:
            meetings += ms
    # 2. audio (large: parallel, resumable)
    wavdir = os.path.join(a.out, "wav")
    t0, done, total = time.time(), 0, 0
    with ThreadPoolExecutor(a.workers) as ex:
        for m, size, st in ex.map(lambda m: fetch(m, wavdir), meetings):
            done += 1
            total += max(size, 0)
            print(f"[{done}/{len(meetings)}] {m} {size / 1e6:.0f} MB {st}  ({total / 1e9:.1f} GB, {(time.time() - t0) / 60:.1f} min)",
                  flush=True)
    bad = [m for m in meetings if not os.path.exists(os.path.join(wavdir, f"{m}.Mix-Headset.wav"))]
    print("missing:", bad if bad else "none", flush=True)
    sys.exit(1 if bad else 0)  # non-zero exit = run it again to fetch the missing meetings


if __name__ == "__main__":
    main()
