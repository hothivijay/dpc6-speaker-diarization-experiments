"""
eend_core.py - the NumPy-only parts of SA-EEND: audio features, chunk linking and post-processing.

They live in their own file so that edge inference (bench_edge.py with ONNX Runtime, e.g. on a Raspberry Pi 5)
runs WITHOUT PyTorch. sa_eend.py imports the same functions for training, so training and deployment share one
feature definition. The NumPy log-Mel matches the torch.stft path of sa_eend.py to < 1e-3 (tests/test_eend.py).

Feature pipeline (as in Fujita et al., ASRU 2019):
  16 kHz audio -> 25 ms Hamming window, 10 ms hop -> 512-point FFT power -> 23 Mel bands -> log10
  -> subtract the per-recording mean ("logmel23_mn")
  -> stack +-7 neighbouring frames (15 x 23 = 345 dims) and keep every 10th frame  =>  one vector per 100 ms.
"""
import numpy as np

# sample rate, window, hop, FFT size, Mel bands, context frames each side, subsampling factor
SR, WIN, HOP, NFFT, NMEL, CTX, SUB = 16000, 400, 160, 512, 23, 7, 10
NSPK = 4                # the model always outputs 4 speaker tracks
FRAME = HOP * SUB / SR  # model frame = 0.1 s


def mel_fb(sr=SR, n_fft=NFFT, n_mels=NMEL, fmin=0.0, fmax=None):
    """Slaney-style Mel filterbank, identical to librosa.filters.mel defaults (no librosa dependency needed)."""
    fmax = fmax or sr / 2
    hz2mel = lambda f: np.where(f >= 1000, 15 + 27 * np.log(np.maximum(f, 1e-9) / 1000) / np.log(6.4), 3 * f / 200)  # noqa
    mel2hz = lambda m: np.where(m >= 15, 1000 * np.exp(np.log(6.4) / 27 * (m - 15)), 200 * m / 3)  # noqa
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    hz = mel2hz(np.linspace(hz2mel(np.array(fmin)), hz2mel(np.array(fmax)), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, ce, hi = hz[i], hz[i + 1], hz[i + 2]
        # triangle between lo and hi with its peak at ce, area-normalised ("slaney" norm)
        fb[i] = np.maximum(0, np.minimum((freqs - lo) / (ce - lo), (hi - freqs) / (hi - ce))) * (2.0 / (hi - lo))
    return fb


_FB = mel_fb()
# periodic Hamming window of 400 samples, centred in a 512-sample frame (what torch.stft does with win_length=400)
_W = np.zeros(NFFT, np.float32)
_W[(NFFT - WIN) // 2:(NFFT - WIN) // 2 + WIN] = (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(WIN) / WIN)).astype(np.float32)


def logmel_np(x, block=6000):
    """Waveform (16 kHz float) -> (T, 23) mean-normalised log-Mel, T = 1 + len(x) // 160.

    Same as torch.stft(center=True, reflect padding, periodic Hamming 400 in a 512 frame) -> power -> Mel -> log10.
    Frames are processed in blocks of `block` so that memory stays small even for hour-long recordings on a Pi.
    """
    x = np.asarray(x, np.float32)
    p = np.pad(x, (NFFT // 2, NFFT // 2), mode="reflect")  # centre the first frame on sample 0
    n = 1 + (len(p) - NFFT) // HOP
    out = np.empty((n, NMEL), np.float32)
    for b0 in range(0, n, block):
        b1 = min(n, b0 + block)
        idx = (np.arange(b0, b1) * HOP)[:, None] + np.arange(NFFT)[None, :]  # sample indices of each frame
        spec = np.abs(np.fft.rfft(p[idx] * _W, axis=1)) ** 2                  # power spectrum
        out[b0:b1] = np.log10(np.maximum(spec.astype(np.float32) @ _FB.T, 1e-10))
    return out - out.mean(0, keepdims=True)                                    # per-recording mean normalisation


def splice_subsample(y):
    """(T, 23) -> (ceil(T / 10), 345): +-7 frames of context around every 10th frame (edge frames replicated)."""
    p = np.pad(y, ((CTX, CTX), (0, 0)), mode="edge")
    idx = np.arange(0, len(y), SUB)[:, None] + np.arange(2 * CTX + 1)[None, :]
    return p[idx].reshape(len(idx), -1)


def link_chunks(probs_fn, T, L):
    """Run the model on overlapping windows and stitch the outputs into one consistent speaker order.

    probs_fn(s, e) must return speaker probabilities (e - s, 4) for model frames s..e-1.
    Windows have L frames and hop L/2. EEND outputs speakers in an arbitrary order per window, so each new window
    is permuted to agree with the running average on the half it shares with earlier windows (Hungarian
    assignment on the dot product of probabilities). Overlapping predictions are then averaged.
    L <= 0 (or L >= T) processes the whole recording in one pass.
    """
    from scipy.optimize import linear_sum_assignment
    if L <= 0 or L >= T:
        return probs_fn(0, T)
    hop = L // 2
    acc = np.zeros((T, NSPK), dtype=np.float64)  # sum of aligned probabilities
    cnt = np.zeros((T, 1), dtype=np.float64)     # how many windows covered each frame
    starts = list(range(0, max(T - L, 0) + 1, hop))
    if starts[-1] + L < T:
        starts.append(T - L)  # last window ends exactly at the end of the recording
    for s in starts:
        p = probs_fn(s, s + L)
        if cnt[s:s + L].sum() > 0:
            ov = cnt[s:s + L, 0] > 0                       # frames already seen
            prev = acc[s:s + L][ov] / cnt[s:s + L][ov]     # current estimate on those frames
            _, perm = linear_sum_assignment(-(prev.T @ p[ov]))
            p = p[:, perm]                                 # reorder this window's speakers
        acc[s:s + L] += p
        cnt[s:s + L] += 1
    return acc / np.maximum(cnt, 1)


def median_filter(b, k):
    """Median filter of k frames along time for every speaker track (k <= 1: no filtering)."""
    if k <= 1:
        return b
    from scipy.ndimage import median_filter as mf
    return mf(b.astype(np.uint8), size=(k, 1), mode="nearest")


def to_segments(prob, th, med):
    """Probabilities (T, 4) -> [(start_s, end_s, 'spkJ'), ...] using threshold `th` and median filter `med`."""
    b = median_filter(prob > th, med)
    segs = []
    for j in range(b.shape[1]):
        d = np.diff(np.concatenate([[0], b[:, j].astype(np.int8), [0]]))  # +1 = segment start, -1 = segment end
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            segs.append((s * FRAME, e * FRAME, f"spk{j}"))
    return segs


def predict_np(runner, x, L):
    """runner: callable(np array (t, 345)) -> probabilities (t, 4). Returns linked probabilities for all of x."""
    return link_chunks(lambda s, e: runner(x[s:e]), len(x), L)
