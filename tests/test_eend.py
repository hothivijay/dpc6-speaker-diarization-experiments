"""
Unit tests of the SA-EEND building blocks.   Run:  python -m pytest tests -q

  * PIT-BCE equals a brute-force minimum over all speaker permutations
  * chunk linking recovers the speaker order when every window is randomly permuted
  * NumPy features (used on the edge device) match the PyTorch features (used in training)
  * ONNX export: FP32 matches PyTorch; INT8 stays close
"""
import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import eend_core as E  # noqa: E402

torch = pytest.importorskip("torch")
import sa_eend as S  # noqa: E402


def test_pit_bce_matches_brute_force():
    torch.manual_seed(0)
    lo = torch.randn(3, 50, 4)
    y = (torch.rand(3, 50, 4) > 0.6).float()
    loss, _ = S.pit_bce(lo, y)
    brute = sum(min(torch.nn.functional.binary_cross_entropy_with_logits(lo[b][:, list(p)], y[b], reduction="sum").item()
                    for p in itertools.permutations(range(4))) for b in range(3))
    assert loss.item() == pytest.approx(brute / (3 * 50 * 4), rel=1e-5)


def test_link_chunks_recovers_random_permutations():
    rng = np.random.default_rng(0)
    T = 3000
    base = np.zeros((T, 4))
    base[np.arange(T), (np.arange(T) // 37) % 4] = 1  # each speaker talks in turn
    perms = [rng.permutation(4) for _ in range(40)]
    calls = [0]

    def window(s, e):  # what an EEND model does: right activity, arbitrary speaker order per window
        p = base[s:e][:, perms[calls[0] % 40]]
        calls[0] += 1
        return p

    out = E.link_chunks(window, T, 300)
    co = out.T @ base  # after linking, every output track matches exactly one reference speaker
    assert np.allclose(np.sort(co.max(0)), np.sort(base.sum(0)))


def test_numpy_features_match_torch_features():
    x = np.random.default_rng(1).normal(size=16000 * 7).astype(np.float32) * 0.1
    a, b = E.logmel_np(x), S.logmel(x)
    assert a.shape == b.shape == (1 + len(x) // E.HOP, E.NMEL)
    assert np.abs(a - b).max() < 1e-3


def test_splice_subsample_shape():
    f = E.splice_subsample(np.zeros((701, 23), np.float32))
    assert f.shape == (71, 345)  # ceil(701 / 10) frames of 15 x 23


def test_to_segments():
    p = np.zeros((30, 4))
    p[5:15, 0] = 0.9
    p[10:20, 2] = 0.8
    segs = E.to_segments(p, 0.5, 1)
    assert segs == [(0.5, 1.5, "spk0"), (1.0, 2.0, "spk2")]


def test_onnx_export_fp32_and_int8(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    m = S.SAEEND(layers=2).eval()
    torch.save(dict(model=m.state_dict(), layers=2), tmp_path / "avg.pt")

    class A:
        model = str(tmp_path / "avg.pt")
        out = str(tmp_path / "sa")
    S.cmd_export(A)
    x = np.random.default_rng(0).normal(size=(1, 400, 345)).astype(np.float32)
    with torch.no_grad():
        ref = torch.sigmoid(m(torch.from_numpy(x))).numpy()
    for v, tol in (("fp32", 1e-4), ("int8", 0.1)):
        sess = ort.InferenceSession(str(tmp_path / f"sa_{v}.onnx"), providers=["CPUExecutionProvider"])
        out = 1 / (1 + np.exp(-sess.run(None, {"feats": x})[0]))
        assert np.abs(out - ref).max() < tol, v
    assert os.path.getsize(tmp_path / "sa_int8.onnx") < 0.5 * os.path.getsize(tmp_path / "sa_fp32.onnx")
