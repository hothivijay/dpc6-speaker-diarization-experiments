# Scoring protocols

Every DER in this repository names its protocol. The same output can look very different under different
rules, so a number without a protocol cannot be compared with anything.

| Name | Collar (each side) | Overlapped speech | Used by |
|---|---|---|---|
| **P1** | none | scored | DISPLACE challenge rule. **Default for every headline number here** |
| **P2** | 0.25 s | ignored | older SpeechBrain AMI recipe ("legacy" numbers in earlier papers) |
| **C25** | 0.25 s | scored | shown for reference |

## How DER is computed (`src/der.py`)

1. Reference and hypothesis segments become 10 ms frame-activity matrices.
2. Frames outside the UEM, inside a collar, or (P2) with more than one reference speaker are removed.
3. One optimal one-to-one speaker mapping per recording (Hungarian algorithm on jointly active time).
4. For every frame with `Nref` reference and `Nhyp` hypothesis speakers:
   * missed = `max(Nref − Nhyp, 0)`
   * false alarm = `max(Nhyp − Nref, 0)`
   * confusion = `min(Nref, Nhyp) − correctly mapped`
5. DER = (missed + false alarm + confusion) / total reference speaker time.

This is the NIST md-eval definition. `tests/test_der.py` checks it against hand-computed cases and against
`pyannote.metrics`. Note that pyannote's `collar` argument is the **total** width (0.5 = 0.25 s on each side).

## Pooled DER, confidence intervals, paired tests

* **Pooled DER** over a test set = total error time / total reference time. This is not the mean of per-recording DERs.
* **95% CI**: bootstrap over recordings (B = 10 000, seed 0). A test set with a few very hard recordings has a
  wide interval, and the interval shows it.
* **Paired comparison** of two systems on the same recordings: a bootstrap CI of the pooled-DER difference, a Wilcoxon
  signed-rank test on per-recording differences, and counts of recordings where the system is better, tied or worse.

## The one-speaker-per-frame floor

When overlap is scored (P1, C25), a system that outputs at most one speaker per frame **must** miss every
overlapping speaker beyond the first. Its lowest reachable DER is therefore

    floor = (total reference speaker time − total speech time) / total reference speaker time

For DISPLACE-2024 eval this is **16.04%** (19.45 h of speaker time, 16.33 h of speech). The NeMo clustering
diarizer with reference speech regions misses exactly this much (16.04% miss, 0% false alarm). In other words,
all of its remaining missed speech is overlap that a clustering system cannot assign.

## Why P2 numbers look so much better

P2 removes overlap frames and 0.5 s around every boundary. On DISPLACE, the same NeMo-C oracle-VAD output scores
23.29% under P1, 14.23% under C25, and 5.59% under P2. Single-digit DISPLACE numbers reported elsewhere need
to be read with this in mind.
