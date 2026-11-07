"""
Unit tests of the DER scorer on small hand-computed cases.   Run:  python -m pytest tests -q

Each case is simple enough to check with pen and paper; the comment says how the expected value is obtained.
"""
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import der  # noqa: E402

P1, P2, C25 = (der.PROTOCOLS[k] for k in ("P1", "P2", "C25"))


def D(ref, hyp, uem=None, **kw):
    return der.der(der.score_file(ref, hyp, uem, **kw))


def test_perfect_hypothesis_is_zero_whatever_the_labels():
    ref = [(0, 10, "A"), (10, 20, "B")]
    hyp = [(0, 10, "x"), (10, 20, "y")]  # different label names: the Hungarian mapping takes care of them
    assert D(ref, hyp) == pytest.approx(0.0)


def test_empty_hypothesis_is_all_missed():
    r = der.score_file([(0, 10, "A")], [])
    assert r["miss"] == pytest.approx(10.0) and der.der(r) == pytest.approx(100.0)


def test_false_alarm():
    # 5 s of hypothesis speech where the reference is silent; 10 s reference -> 50% DER, all false alarm
    r = der.score_file([(0, 10, "A")], [(0, 10, "A"), (10, 15, "B")])
    assert r["fa"] == pytest.approx(5.0) and der.der(r) == pytest.approx(50.0)


def test_confusion():
    # one hypothesis speaker covers two reference speakers: it maps to one of them, the other 10 s are confusion
    r = der.score_file([(0, 10, "A"), (10, 20, "B")], [(0, 20, "X")])
    assert r["conf"] == pytest.approx(10.0) and der.der(r) == pytest.approx(50.0)


def test_overlap_scored_vs_ignored():
    ref = [(0, 10, "A"), (5, 10, "B")]  # 5 s overlap; total reference speaker time 15 s
    hyp = [(0, 10, "A")]                # single-speaker output misses B in the overlap
    assert D(ref, hyp, **P1) == pytest.approx(100 * 5 / 15)
    assert D(ref, hyp, **P2) == pytest.approx(0.0)  # P2 drops overlap frames (and a 0.25 s collar)


def test_collar_forgives_boundary_errors():
    ref, hyp = [(0, 10, "A")], [(0.2, 10, "A")]  # 0.2 s late start
    assert D(ref, hyp, **P1) == pytest.approx(2.0)   # 0.2 / 10
    assert D(ref, hyp, **C25) == pytest.approx(0.0)  # inside the 0.25 s collar


def test_uem_limits_the_scored_region():
    ref, hyp = [(0, 10, "A")], [(0, 5, "A")]
    assert D(ref, hyp) == pytest.approx(50.0)
    assert D(ref, hyp, uem=[(0, 5)]) == pytest.approx(0.0)


def test_pooled_weighs_by_reference_time():
    a = der.score_file([(0, 10, "A")], [])           # 10 s, 100% missed
    b = der.score_file([(0, 90, "A")], [(0, 90, "A")])  # 90 s, perfect
    assert der.pooled([a, b])["der"] == pytest.approx(10.0)  # 10 / 100, not the mean of 100% and 0%


def test_bootstrap_and_paired_are_consistent():
    rng = random.Random(0)
    rows_a, rows_b = [], []
    for i in range(20):
        ref = [(0, 60, "A"), (60, 120, "B")]
        cut = rng.uniform(55, 65)
        rows_a.append(der.score_file(ref, [(0, cut, "x"), (cut, 120, "y")]))
        rows_b.append(der.score_file(ref, [(0, 120, "x")]))  # much worse: one speaker for everything
    lo, hi = der.bootstrap_ci(rows_a)
    assert lo <= der.pooled(rows_a)["der"] <= hi
    same = der.paired(rows_a, rows_a)
    assert same["delta"] == 0 and same["tied"] == 20
    worse = der.paired(rows_a, rows_b)
    assert worse["delta"] > 0 and worse["ci"][0] > 0 and worse["worse"] == 20 and worse["p_wilcoxon"] < 0.01


def test_evaluate_counts_missing_recordings_as_missed():
    ref = {"r1": [(0, 10, "A")], "r2": [(0, 10, "A")]}
    res = der.evaluate(ref, {"sys": {"r1": [(0, 10, "A")]}}, protocols=("P1",))
    o = res["protocols"]["P1"]["sys"]
    assert o["missing"] == ["r2"] and o["pooled"]["der"] == pytest.approx(50.0)


def test_rttm_round_trip(tmp_path):
    segs = [(0.0, 1.5, "A"), (2.0, 3.25, "B")]
    der.write_rttm(tmp_path / "f.rttm", "f", segs)
    assert der.read_rttm(str(tmp_path / "f.rttm")) == {"f": segs}


def test_agrees_with_pyannote_metrics():
    """Cross-check with pyannote.metrics on random multi-speaker data (skipped if pyannote.metrics is missing)."""
    pm = pytest.importorskip("pyannote.metrics.diarization")
    from pyannote.core import Annotation, Segment
    rng = np.random.default_rng(1)

    def rand_segs(n_spk, prefix):
        segs = []
        for j in range(n_spk):
            t = 0.0
            while t < 300:
                d = float(rng.uniform(0.5, 8))
                if rng.random() < 0.4:
                    segs.append((round(t, 2), round(min(t + d, 300), 2), f"{prefix}{j}"))
                t += d + float(rng.uniform(0, 3))
        return segs

    def ann(segs):
        a = Annotation()
        for s, e, l in segs:
            a[Segment(s, e)] = l
        return a.support()  # merge touching segments of the same speaker, as RTTM scoring does

    for pname, (c, so) in {"P1": (0.0, False), "C25": (0.5, False), "P2": (0.5, True)}.items():
        # pyannote's collar is the TOTAL width (0.5 = 0.25 s on each side)
        metric = pm.DiarizationErrorRate(collar=c, skip_overlap=so)
        ref, hyp = rand_segs(4, "r"), rand_segs(4, "h")
        ours = D(ref, hyp, **der.PROTOCOLS[pname])
        theirs = 100 * metric(ann(ref), ann(hyp))
        assert ours == pytest.approx(theirs, abs=0.3), pname
