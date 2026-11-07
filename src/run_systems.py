#!/usr/bin/env python3
"""
run_systems.py - run off-the-shelf diarizers zero-shot on DISPLACE-2024 (or any folder of recordings) and write one
RTTM per recording and system. Each system uses its own speech detection unless "oraclevad" is in its name.

  python run_systems.py prep                --audio-dir RAW_AUDIO --sd-ref REF_RTTM_DIR --out WAV16K
  python run_systems.py nemo_clus           --wav WAV16K --out SYS_DIR
  python run_systems.py nemo_clus_oraclevad --wav WAV16K --sd-ref REF_RTTM_DIR --out SYS_DIR
  python run_systems.py sortformer          --wav WAV16K --out SYS_DIR
  python run_systems.py pyannote            --wav WAV16K --out SYS_DIR        (needs HF_TOKEN in the environment)
  python run_systems.py nemo_msdd           --wav WAV16K --out SYS_DIR        (extra option, not used in the results)

SYS_DIR/<system>/<recording>.rttm is written; recordings that are already done are skipped, so every command can be
restarted after a disconnect.

SYSTEMS
  nemo_clus            NVIDIA NeMo clustering diarizer, "meeting" inference config: MarbleNet multilingual VAD,
                       TitaNet-Large speaker embeddings, NME-SC spectral clustering (number of speakers estimated).
  nemo_clus_oraclevad  the same diarizer, but the REFERENCE speech regions replace MarbleNet. Comparing it with
                       nemo_clus separates speech-detection errors from embedding/clustering errors. It uses
                       reference information, so it is an analysis tool, not a deployable system.
  sortformer           NVIDIA Streaming Sortformer v2 (end-to-end, at most 4 speakers), highest-latency preset.
  pyannote             pyannote/speaker-diarization-3.1 with pyannote.audio 3.3.2.

REQUIREMENTS
  NeMo systems: pip install "nemo_toolkit[asr]"  (GPU strongly recommended)
  pyannote: see requirements-pyannote.txt and run_pyannote.py for the Hugging Face token steps.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
import urllib.request

import numpy as np

# NeMo keeps its inference configs in the repository, not in the pip package: download the one we need
NEMO_CONF = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{}"


def write_rttm(path, fid, segs):
    with open(path, "w") as f:
        for s, e, lab in sorted(segs):
            if e > s:
                f.write(f"SPEAKER {fid} 1 {s:.3f} {e - s:.3f} <NA> <NA> {lab} <NA> <NA>\n")


def wav_list(d):
    return sorted(glob.glob(os.path.join(d, "*.wav")))


def todo(wavs, outdir):
    """Recordings that do not have an RTTM in outdir yet."""
    os.makedirs(outdir, exist_ok=True)
    return [w for w in wavs if not os.path.exists(os.path.join(outdir, os.path.basename(w)[:-4] + ".rttm"))]


# ------------------------------------------------------------------ prep
def cmd_prep(a):
    """16 kHz mono wav copies of every recording that has a speaker reference (searches --audio-dir recursively
    for <id>.wav or <id>.flac)."""
    import soundfile as sf
    ids = sorted({line.split()[1] for p in glob.glob(os.path.join(a.sd_ref, "*.rttm")) for line in open(p)
                  if line.startswith("SPEAKER")})
    os.makedirs(a.out, exist_ok=True)
    for i, fid in enumerate(ids):
        dst = os.path.join(a.out, fid + ".wav")
        if os.path.exists(dst):
            continue
        src = (glob.glob(os.path.join(a.audio_dir, "**", fid + ".wav"), recursive=True) +
               glob.glob(os.path.join(a.audio_dir, "**", fid + ".flac"), recursive=True))
        if not src:
            print("no audio for", fid)
            continue
        x, sr = sf.read(src[0], dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != 16000:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        sf.write(dst, x, 16000, subtype="PCM_16")
        print(f"[{i + 1}/{len(ids)}] {fid}: {len(x) / 16000 / 60:.1f} min", flush=True)


# ------------------------------------------------------------------ NeMo
def nemo_cfg(conf_name, manifest, out_dir):
    """Load NeMo's diar_infer_meeting.yaml and set the models used here (MarbleNet VAD, TitaNet-L)."""
    from omegaconf import OmegaConf
    local = os.path.join(out_dir, conf_name)
    if not os.path.exists(local):
        urllib.request.urlretrieve(NEMO_CONF.format(conf_name), local)
    cfg = OmegaConf.load(local)
    cfg.diarizer.manifest_filepath = manifest
    cfg.diarizer.out_dir = out_dir
    cfg.diarizer.oracle_vad = False
    cfg.diarizer.clustering.parameters.oracle_num_speakers = False  # the number of speakers is estimated
    cfg.diarizer.vad.model_path = "vad_multilingual_marblenet"
    cfg.diarizer.speaker_embeddings.model_path = "titanet_large"
    cfg.num_workers = 0
    return cfg


def nemo_manifest(wavs, path, rttm=None):
    with open(path, "w") as f:
        for w in wavs:
            f.write(json.dumps({"audio_filepath": w, "offset": 0, "duration": None, "label": "infer", "text": "-",
                                "num_speakers": None, "rttm_filepath": rttm, "uem_filepath": None}) + "\n")


def ref_speech_rttm(sd_ref, fid, path):
    """Reference speech regions of one recording as an RTTM for NeMo's oracle VAD (speaker labels are not used)."""
    segs = []
    for p in glob.glob(os.path.join(sd_ref, "*.rttm")):
        for line in open(p):
            q = line.split()
            if len(q) >= 8 and q[0] == "SPEAKER" and q[1] == fid and float(q[4]) > 0:
                segs.append((float(q[3]), float(q[3]) + float(q[4]), q[7]))
    write_rttm(path, fid, segs)
    return path


def cmd_nemo(a, msdd, oracle_vad=False):
    """NeMo clustering diarizer (or MSDD), one recording at a time so that a crash loses at most one recording."""
    name = "nemo_msdd" if msdd else ("nemo_clus_oraclevad" if oracle_vad else "nemo_clus")
    outdir = os.path.join(a.out, name)
    wavs = todo(wav_list(a.wav), outdir)
    work = os.path.join(a.out, f"_{name}_work")
    os.makedirs(work, exist_ok=True)
    for w in wavs:
        t0 = time.time()
        fid = os.path.basename(w)[:-4]
        for d in os.listdir(work):  # fresh intermediate folders for every recording (the config file is kept)
            if os.path.isdir(os.path.join(work, d)):
                shutil.rmtree(os.path.join(work, d), ignore_errors=True)
        man = os.path.join(work, "manifest.json")
        rttm = None
        if oracle_vad:  # NeMo requires the RTTM file to have the same base name as the audio file
            os.makedirs(work + "_ref", exist_ok=True)
            rttm = ref_speech_rttm(a.sd_ref, fid, os.path.join(work + "_ref", fid + ".rttm"))
        nemo_manifest([w], man, rttm)
        cfg = nemo_cfg("diar_infer_meeting.yaml", man, work)
        cfg.diarizer.oracle_vad = bool(oracle_vad)
        if msdd:
            from nemo.collections.asr.models.msdd_models import NeuralDiarizer
            cfg.diarizer.msdd_model.model_path = "diar_msdd_telephonic"
            NeuralDiarizer(cfg=cfg).diarize()
        else:
            from nemo.collections.asr.models import ClusteringDiarizer
            ClusteringDiarizer(cfg=cfg).diarize()
        shutil.copy(os.path.join(work, "pred_rttms", fid + ".rttm"), os.path.join(outdir, fid + ".rttm"))
        print(f"{name} {fid}: {time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------ Sortformer
def cmd_sortformer(a):
    """Streaming Sortformer v2, "very high latency" preset of the model card (best accuracy), at most 4 speakers."""
    from nemo.collections.asr.models import SortformerEncLabelModel
    outdir = os.path.join(a.out, "sortformer")
    wavs = todo(wav_list(a.wav), outdir)
    model = SortformerEncLabelModel.from_pretrained(a.sortformer_model).eval()
    m = model.sortformer_modules
    m.chunk_len, m.chunk_right_context, m.fifo_len, m.spkcache_update_period, m.spkcache_len = 340, 40, 40, 300, 188
    if hasattr(m, "_check_streaming_parameters"):
        m._check_streaming_parameters()
    for w in wavs:
        t0 = time.time()
        fid = os.path.basename(w)[:-4]
        out = model.diarize(audio=[w], batch_size=1)
        segs = []
        for seg in out[0]:  # NeMo returns "start end label" strings or tuples, depending on the version
            if isinstance(seg, str):
                p = seg.replace(",", " ").split()
                s, e, lab = float(p[0]), float(p[1]), p[2]
            else:
                s, e, lab = float(seg[0]), float(seg[1]), str(seg[2])
            segs.append((s, e, lab))
        write_rttm(os.path.join(outdir, fid + ".rttm"), fid, segs)
        print(f"sortformer {fid}: {len(segs)} segments, {len({l for _, _, l in segs})} speakers, {time.time() - t0:.0f} s",
              flush=True)


# ------------------------------------------------------------------ pyannote
def cmd_pyannote(a):
    """pyannote 3.1 pipeline (pyannote.audio 3.3.2); needs HF_TOKEN and accepted model conditions."""
    import soundfile as sf
    import torch
    from pyannote.audio import Pipeline
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        raise SystemExit("HF_TOKEN is not set")
    try:
        pipe = Pipeline.from_pretrained(a.pyannote_model, token=tok)
    except TypeError:
        pipe = Pipeline.from_pretrained(a.pyannote_model, use_auth_token=tok)
    if pipe is None:
        raise SystemExit("pyannote pipeline could not be loaded: accept the model conditions on Hugging Face")
    pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    outdir = os.path.join(a.out, "pyannote")
    for w in todo(wav_list(a.wav), outdir):
        t0 = time.time()
        fid = os.path.basename(w)[:-4]
        x, sr = sf.read(w, dtype="float32")
        res = pipe({"waveform": torch.from_numpy(np.asarray(x))[None], "sample_rate": sr})
        ann = getattr(res, "speaker_diarization", res)
        segs = [(t.start, t.end, str(l)) for t, _, l in ann.itertracks(yield_label=True)]
        write_rttm(os.path.join(outdir, fid + ".rttm"), fid, segs)
        print(f"pyannote {fid}: {len(segs)} segments, {len({l for _, _, l in segs})} speakers, {time.time() - t0:.0f} s",
              flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prep", "nemo_clus", "nemo_clus_oraclevad", "nemo_msdd", "sortformer", "pyannote"])
    ap.add_argument("--audio-dir", help="prep: folder with the original recordings (searched recursively)")
    ap.add_argument("--sd-ref", help="folder with the reference speaker RTTMs")
    ap.add_argument("--wav", help="folder with 16 kHz mono wav files (output of prep)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sortformer-model", default="nvidia/diar_streaming_sortformer_4spk-v2")
    ap.add_argument("--pyannote-model", default="pyannote/speaker-diarization-3.1")
    a = ap.parse_args()
    t0 = time.time()
    {"prep": cmd_prep, "nemo_clus": lambda a: cmd_nemo(a, False), "nemo_msdd": lambda a: cmd_nemo(a, True),
     "nemo_clus_oraclevad": lambda a: cmd_nemo(a, False, oracle_vad=True),
     "sortformer": cmd_sortformer, "pyannote": cmd_pyannote}[a.cmd](a)
    print(f"{a.cmd} finished in {(time.time() - t0) / 60:.1f} min", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
