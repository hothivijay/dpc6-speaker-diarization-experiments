#!/usr/bin/env python3
"""
sa_eend.py - SA-EEND re-implemented in PyTorch and trained on AMI.

Reference: Y. Fujita, N. Kanda, S. Horiguchi, Y. Xue, K. Nagamatsu, S. Watanabe,
           "End-to-end neural speaker diarization with self-attention", IEEE ASRU 2019.

WHAT IS KEPT FROM THE PAPER
  * features: 23-dim log-Mel (25 ms / 10 ms), mean-normalised per recording, +-7 frames context (345 dims),
    subsampled x10 -> one frame per 100 ms                                            (see eend_core.py)
  * model: linear 345 -> 256, 4 Transformer encoder blocks (256 units, 4 heads, feed-forward 1024),
    NO positional encoding, one sigmoid output per speaker
  * loss: permutation-invariant binary cross-entropy (PIT-BCE): the speaker order that gives the lowest loss is used
  * optimiser: Adam with the Noam learning-rate schedule (warm-up 4000 steps)
  * inference: threshold 0.5 and 11-frame median filter; model = average of the last 10 epoch checkpoints

WHAT IS CHANGED FOR AMI (and why)
  * 16 kHz audio (AMI is 16 kHz)
  * 4 fixed output tracks = the 4 speakers with the most talk time (AMI meetings have 3-5 speakers, mostly 4)
  * trained directly on the 136 AMI train meetings, no simulated mixtures (none were available in the time budget)
  * 50 s training chunks (500 frames), mixed precision on GPU
  * chunked inference with permutation linking (eend_core.link_chunks); window length, threshold and median
    filter are chosen on AMI Dev and FROZEN before Eval is scored (eend_eval.py)

USAGE
  python sa_eend.py feats  --wav WAV_DIR --rttm RTTM_DIR --list meetings.txt --out FEAT_DIR
  python sa_eend.py train  --feats FEAT_DIR --train-list train.txt --dev-list dev.txt --out EXP_DIR \
                           [--epochs 150 --max-minutes 75]
  python sa_eend.py infer  --model EXP_DIR/avg.pt --feats FEAT_DIR --list test.txt --out RTTM_DIR \
                           [--chunk 1500 --th 0.5 --med 11]
  python sa_eend.py export --model EXP_DIR/avg.pt --out EXP_DIR/sa_eend      (writes _fp32.onnx and _int8.onnx)
"""
import argparse
import glob
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eend_core import (CTX, FRAME, HOP, NFFT, NMEL, NSPK, SR, WIN, link_chunks, mel_fb,  # noqa: E402,F401
                       splice_subsample, to_segments)


# --------------------------------------------------------------------------------------------- features
def logmel(x, device="cpu"):
    """Waveform at 16 kHz -> (T, 23) log-Mel, mean-normalised (EEND 'logmel23_mn'). Uses the GPU when available."""
    t = torch.as_tensor(x, dtype=torch.float32, device=device)
    spec = torch.stft(t, NFFT, HOP, WIN, window=torch.hamming_window(WIN, device=device), center=True,
                      return_complex=True).abs() ** 2
    m = torch.as_tensor(mel_fb(), device=device) @ spec
    y = torch.log10(torch.clamp(m, min=1e-10)).T
    return (y - y.mean(0, keepdim=True)).cpu().numpy()


def read_rttm(path):
    """dir of *.rttm or one RTTM -> {file: [(start, end, label), ...]} (same as der.read_rttm)."""
    out = {}
    for p in (sorted(glob.glob(os.path.join(path, "*.rttm"))) if os.path.isdir(path) else [path]):
        for line in open(p):
            q = line.split()
            if len(q) >= 8 and q[0] == "SPEAKER" and float(q[4]) > 0:
                out.setdefault(q[1], []).append((float(q[3]), float(q[3]) + float(q[4]), q[7]))
    return out


def labels(segs, n, nspk=NSPK):
    """Training targets: speaker activity at 100 ms frames (frame t covers [0.1 t, 0.1 t + 0.1)).

    Only the `nspk` speakers with the most talk time get a track; returns (n, nspk) uint8 and their labels.
    """
    tot = {}
    for s, e, l in segs:
        tot[l] = tot.get(l, 0) + e - s
    keep = sorted(tot, key=lambda l: -tot[l])[:nspk]
    y = np.zeros((n, nspk), dtype=np.uint8)
    for s, e, l in segs:
        if l in keep:
            y[int(round(s / FRAME)):int(round(e / FRAME)), keep.index(l)] = 1
    return y, keep


def cmd_feats(a):
    """Compute features (+ targets when --rttm is given) for every meeting in --list; one .npz per meeting.

    The .npz holds x (frames, 345) float16, y (frames, 4) uint8, spk (speaker labels), dur (seconds).
    Meetings whose .npz already exists are skipped, so the command can be restarted after a disconnect.
    """
    import soundfile as sf
    os.makedirs(a.out, exist_ok=True)
    ref = read_rttm(a.rttm) if a.rttm else {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ids = [l.split()[0] for l in open(a.list) if l.strip()]
    for i, m in enumerate(ids):
        fo = os.path.join(a.out, m + ".npz")
        if os.path.exists(fo):
            continue
        w = os.path.join(a.wav, m + ".Mix-Headset.wav")  # AMI naming; falls back to <meeting>.wav
        if not os.path.exists(w):
            w = os.path.join(a.wav, m + ".wav")
        x, sr = sf.read(w, dtype="float32", always_2d=True)
        x = x.mean(1)  # mono
        assert sr == SR, (w, sr)
        f = splice_subsample(logmel(x, dev)).astype(np.float16)  # float16 halves disk and RAM use
        y, keep = labels(ref.get(m, []), len(f))
        np.savez(fo, x=f, y=y, spk=np.array(keep), dur=len(x) / SR)
        print(f"[{i + 1}/{len(ids)}] {m}: {len(f)} frames, speakers {keep}", flush=True)


# --------------------------------------------------------------------------------------------- model
class Block(nn.Module):
    """One pre-LayerNorm Transformer encoder block: multi-head self-attention + feed-forward, both residual."""

    def __init__(self, d=256, h=4, ff=1024, p=0.1):
        super().__init__()
        self.h, self.p = h, p
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.f1, self.f2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        B, T, D = x.shape
        y = self.ln1(x)
        sh = lambda z: z.view(B, T, self.h, D // self.h).transpose(1, 2)  # noqa: E731  (B, heads, T, D/heads)
        a = F.scaled_dot_product_attention(sh(self.q(y)), sh(self.k(y)), sh(self.v(y)),
                                           dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.f2(self.drop(F.relu(self.f1(self.ln2(x))))))


class SAEEND(nn.Module):
    """345-dim frames -> per-frame logits for NSPK speakers. No positional encoding (as in the paper):
    self-attention compares every frame with every other frame of the chunk, which is what clusters speakers."""

    def __init__(self, idim=(2 * CTX + 1) * NMEL, d=256, h=4, ff=1024, layers=4, nspk=NSPK, p=0.1):
        super().__init__()
        self.inp = nn.Linear(idim, d)
        self.ln_in = nn.LayerNorm(d)
        self.blocks = nn.ModuleList(Block(d, h, ff, p) for _ in range(layers))
        self.ln_out = nn.LayerNorm(d)
        self.out = nn.Linear(d, nspk)

    def forward(self, x):  # (B, T, 345) -> logits (B, T, S)
        x = self.ln_in(self.inp(x))
        for b in self.blocks:
            x = b(x)
        return self.out(self.ln_out(x))


PERMS = None  # cached tensor of all speaker permutations (4! = 24)


def pit_bce(logits, y):
    """Permutation-invariant BCE. logits, y: (B, T, S).

    For every item the BCE is computed for all S! orders of the reference speakers and the smallest is kept.
    Implemented with one pairwise (pred, ref) cost matrix, so the 24 permutations cost almost nothing.
    Returns the mean loss per frame and speaker, and the best permutation per item.
    """
    global PERMS
    S = y.shape[-1]
    if PERMS is None or PERMS.shape[1] != S or PERMS.device != logits.device:
        PERMS = torch.tensor(list(itertools.permutations(range(S))), device=logits.device)
    lp = logits.float().unsqueeze(3).expand(-1, -1, -1, S)
    yr = y.float().unsqueeze(2).expand(-1, -1, S, -1)
    pair = F.binary_cross_entropy_with_logits(lp, yr, reduction="none").sum(1)  # (B, S_pred, S_ref)
    cost = pair[:, torch.arange(S, device=logits.device)[None, :], PERMS].sum(-1)  # (B, P)
    best, idx = cost.min(1)
    return best.sum() / (y.shape[0] * y.shape[1] * S), PERMS[idx]


# --------------------------------------------------------------------------------------------- training
def load_set(featdir, ids):
    """Load the .npz features of the given meetings into memory: {meeting: (x, y)}."""
    data = {}
    for m in ids:
        z = np.load(os.path.join(featdir, m + ".npz"))
        data[m] = (z["x"], z["y"])
    return data


def chunks(data, L, rng=None):
    """All non-overlapping chunks of L frames as (meeting, start). With rng, each epoch uses a random offset."""
    out = []
    for m, (x, _) in data.items():
        off = int(rng.integers(0, L)) if rng is not None else 0
        for s in range(off, len(x) - L + 1, L):
            out.append((m, s))
    return out


def batch(data, items, L, device):
    xs = np.stack([data[m][0][s:s + L] for m, s in items]).astype(np.float32)
    ys = np.stack([data[m][1][s:s + L] for m, s in items])
    return torch.from_numpy(xs).to(device), torch.from_numpy(ys).to(device)


def cmd_train(a):
    """Train with PIT-BCE; save one checkpoint per epoch, then the two averaged models avg.pt and avg_best.pt.

    Resumable: last.pt holds model, optimiser, scheduler and history, and is loaded automatically if present.
    --max-minutes stops training after the epoch in which the time budget was reached (useful on free Colab).
    """
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    tr = load_set(a.feats, [l.split()[0] for l in open(a.train_list) if l.strip()])
    dv = load_set(a.feats, [l.split()[0] for l in open(a.dev_list) if l.strip()])
    model = SAEEND(layers=a.layers).to(dev)
    # Noam schedule: lr = d^-0.5 * min(step^-0.5, step * warmup^-1.5)  (base lr 1.0 is multiplied by this)
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    noam = lambda st: (256 ** -0.5) * min((st + 1) ** -0.5, (st + 1) * a.warmup ** -1.5)  # noqa: E731
    sched = torch.optim.lr_scheduler.LambdaLR(opt, noam)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")  # mixed precision on GPU only
    ck = os.path.join(a.out, "last.pt")
    ep0, hist = 0, []
    if os.path.exists(ck):  # resume after a disconnect
        st = torch.load(ck, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        ep0, hist = st["epoch"], st["hist"]
        print(f"resumed at epoch {ep0}", flush=True)
    t_start = time.time()
    nparams = sum(p.numel() for p in model.parameters())
    print(f"SA-EEND {nparams / 1e6:.2f} M parameters; {len(tr)} train / {len(dv)} dev recordings; device {dev}", flush=True)
    for ep in range(ep0, a.epochs):
        # ---- one training epoch over shuffled 50 s chunks
        model.train()
        items = chunks(tr, a.chunk, rng)
        rng.shuffle(items)
        tl, nb, t0 = 0.0, 0, time.time()
        for i in range(0, len(items) - a.batch + 1, a.batch):
            x, y = batch(tr, items[i:i + a.batch], a.chunk, dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                lo = model(x)
            loss, _ = pit_bce(lo, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item(); nb += 1
        # ---- dev loss (used to pick the avg_best window; DER-based choices happen in eend_eval.py)
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            vi = chunks(dv, a.chunk)
            for i in range(0, len(vi), a.batch):
                x, y = batch(dv, vi[i:i + a.batch], a.chunk, dev)
                loss, _ = pit_bce(model(x), y)
                vl += loss.item() * len(x); vn += len(x)
        hist.append(dict(epoch=ep + 1, train_loss=tl / max(nb, 1), dev_loss=vl / max(vn, 1),
                         lr=sched.get_last_lr()[0], sec=time.time() - t0))
        torch.save(model.state_dict(), os.path.join(a.out, f"ep{ep + 1:03d}.pt"))
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep + 1,
                        hist=hist), ck)
        print(f"epoch {ep + 1}: train {hist[-1]['train_loss']:.4f} dev {hist[-1]['dev_loss']:.4f} "
              f"lr {hist[-1]['lr']:.2e} {hist[-1]['sec']:.0f}s", flush=True)
        if (time.time() - t_start) / 60 > a.max_minutes:
            print(f"time budget of {a.max_minutes} min reached after epoch {ep + 1}", flush=True)
            break
    # Two averaged models:
    #   avg.pt      = average of the last a.avg epoch models (the paper's recipe)
    #   avg_best.pt = average of a.avg epochs centred on the lowest dev loss (helps when training is cut short)
    # Which one is used is decided on Dev DER only (eend_eval.py), never on Eval.
    allp = sorted(glob.glob(os.path.join(a.out, "ep*.pt")))
    n = len(allp)
    best = int(np.argmin([h["dev_loss"] for h in hist[-n:]]))
    lo = max(0, min(best - a.avg // 2, n - a.avg))
    sets = {"avg.pt": allp[-a.avg:], "avg_best.pt": allp[lo:lo + a.avg]}
    for name, eps in sets.items():
        avg = None
        for p in eps:
            sd = torch.load(p, map_location="cpu")
            avg = {k: v.clone().float() for k, v in sd.items()} if avg is None else {k: avg[k] + sd[k].float() for k in avg}
        avg = {k: v / len(eps) for k, v in avg.items()}
        torch.save(dict(model=avg, layers=a.layers, epochs_averaged=[os.path.basename(p) for p in eps]),
                   os.path.join(a.out, name))
        print(name, "averages", [os.path.basename(p) for p in eps], flush=True)
    keep = set(sets["avg.pt"]) | set(sets["avg_best.pt"])
    for p in allp:  # delete epoch checkpoints that are not part of either average (saves Drive space)
        if p not in keep:
            os.remove(p)
    json.dump(dict(params=nparams, hist=hist, averaged={k: [os.path.basename(p) for p in v] for k, v in sets.items()},
                   chunk=a.chunk, batch=a.batch, warmup=a.warmup, seed=a.seed, layers=a.layers),
              open(os.path.join(a.out, "train_log.json"), "w"), indent=1)


# --------------------------------------------------------------------------------------------- inference
def load_model(path, device="cpu"):
    """Load avg.pt / avg_best.pt (dict with 'model' and 'layers') in eval mode."""
    st = torch.load(path, map_location="cpu")
    m = SAEEND(layers=st.get("layers", 4))
    m.load_state_dict(st["model"])
    return m.to(device).eval()


def predict(model, x, L, device="cpu"):
    """x: (T, 345) -> per-frame speaker probabilities (T, 4); windows of L frames linked by permutation."""
    def fn(s, e):
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(x[s:e], dtype=np.float32))[None].to(device)
            return torch.sigmoid(model(t))[0].float().cpu().numpy()
    return link_chunks(fn, len(x), L)


def cmd_infer(a):
    """Write one RTTM per meeting with the given window (--chunk), threshold (--th) and median filter (--med)."""
    from der import write_rttm
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    model = load_model(a.model, dev)
    os.makedirs(a.out, exist_ok=True)
    for m in [l.split()[0] for l in open(a.list) if l.strip()]:
        x = np.load(os.path.join(a.feats, m + ".npz"))["x"]
        p = predict(model, x, a.chunk, dev)
        write_rttm(os.path.join(a.out, m + ".rttm"), m, to_segments(p, a.th, a.med))
        print("inferred", m, flush=True)


# --------------------------------------------------------------------------------------------- export
def cmd_export(a):
    """Export the model to ONNX (FP32) and make a dynamically quantised INT8 copy (weights of every Linear/MatMul
    in int8, activations quantised on the fly). Both files are what bench_edge.py runs on a CPU / Raspberry Pi."""
    import onnx  # noqa: F401
    from onnxruntime.quantization import QuantType, quantize_dynamic
    m = load_model(a.model, "cpu")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fp = a.out + "_fp32.onnx"
    torch.onnx.export(m, torch.randn(1, 500, (2 * CTX + 1) * NMEL), fp, input_names=["feats"], output_names=["logits"],
                      opset_version=17, dynamic_axes={"feats": {1: "T"}, "logits": {1: "T"}}, dynamo=False)
    quantize_dynamic(fp, a.out + "_int8.onnx", weight_type=QuantType.QInt8)
    for p in (fp, a.out + "_int8.onnx"):
        print(p, f"{os.path.getsize(p) / 1e6:.2f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["feats", "train", "infer", "export"])
    ap.add_argument("--wav"); ap.add_argument("--rttm"); ap.add_argument("--list"); ap.add_argument("--out")
    ap.add_argument("--feats"); ap.add_argument("--train-list"); ap.add_argument("--dev-list")
    ap.add_argument("--model")
    ap.add_argument("--epochs", type=int, default=150); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=500, help="train: chunk length; infer: window (0 = whole)")
    ap.add_argument("--warmup", type=int, default=4000)
    ap.add_argument("--layers", type=int, default=4); ap.add_argument("--avg", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=1e9, help="stop training after this many minutes")
    ap.add_argument("--th", type=float, default=0.5); ap.add_argument("--med", type=int, default=11)
    ap.add_argument("--cpu", action="store_true", help="infer on CPU even if a GPU is present")
    a = ap.parse_args()
    {"feats": cmd_feats, "train": cmd_train, "infer": cmd_infer, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
