# Measured results

All DERs are pooled over recordings with a recording-bootstrap 95% CI (B = 10 000). **P1** = no collar,
overlap scored. P2 = 0.25 s collar, overlap ignored. C25 = 0.25 s collar, overlap scored
([protocols](../docs/SCORING_PROTOCOLS.md)).

## DISPLACE-2024 Track-1 eval (32 sessions, 17.92 h audio, 16.33 h speech)

Produced by `src/run_systems.py` and `src/displace_rescore.py`. Numbers: [`displace_summary.json`](displace_summary.json).

| System (zero-shot) | P1 DER % [95% CI] | Miss | FA | Conf | Median | P2 DER % |
|---|---|---|---|---|---|---|
| Streaming Sortformer v2 | **26.87** [21.78, 33.09] | 11.95 | 6.75 | 8.17 | 21.05 | 15.32 |
| pyannote 3.1 | 34.75 [29.24, 40.76] | 10.70 | 8.51 | 15.53 | 32.04 | 20.46 |
| NeMo clustering (MarbleNet VAD) | 68.34 [59.93, 76.08] | 61.46 | 1.91 | 4.97 | 61.95 | 59.17 |
| NeMo clustering, reference speech regions* | 23.29 [18.87, 28.09] | 16.04 | 0.00 | 7.25 | 20.34 | 5.59 |

\* Uses the reference speech regions in place of the VAD, so it is an analysis condition and not a deployable system.

Main findings:

* **Speech detection is the bottleneck of the clustering pipeline.** Swapping MarbleNet for the reference speech
  regions lowers P1 DER by 45.05 points [35.79, 53.54] and improves all 32 of 32 sessions (Wilcoxon p = 4.7e-10).
* **Overlap sets a hard floor.** A system that outputs one speaker per frame cannot go below **16.04%** P1 DER on
  this set. The oracle-VAD clustering run misses exactly 16.04%.
* **The protocol changes the picture.** The same oracle-VAD output scores 23.29% (P1), 14.23% (C25) and 5.59% (P2).
* **A few sessions dominate.** The 5 worst sessions hold 32% of all Sortformer errors on 14% of the speech.
  M035 is the worst session for Sortformer (84.32%), pyannote (79.33%) and oracle-VAD clustering (64.78%).
* pyannote − Sortformer: +7.87 points [4.16, 11.32], with pyannote worse on 25 of 32 sessions (p = 4.0e-4).
  Choosing the best system per session afterwards (oracle selection) gives 25.49%, and DOVER-Lap fusion gives 32.41%.

## AMI Mix-Headset, BUT only_words references

SA-EEND (3.25 M parameters) was trained on the 136 train meetings for 41 epochs in 23 min on a Colab T4 (stopped there to fit the review schedule). The lowest dev loss came at epoch 40. Settings chosen on Dev (14 meetings) and frozen: `avg.pt`, window 3000 frames (300 s), threshold 0.4, median 11 frames (Dev P1 53.82%).

| System | Eval-12 P1 % [95% CI] | Miss | FA | Conf | Eval-12 P2 % | Eval-16 P1 % [95% CI] |
|---|---|---|---|---|---|---|
| SA-EEND (ours, re-implemented) | 51.11 [48.79, 53.49] | 19.07 | 8.54 | 23.50 | 45.50 | 51.98 [49.79, 54.22] |
| pyannote 3.1 (zero-shot) | 19.64 [15.99, 23.01] | 9.77 | 3.32 | 6.55 | 5.97 | 18.78 [15.68, 21.68] |

Reproduction check: the published pyannote 3.1 value for this setting (all 16 meetings, no collar, overlap scored) is 18.8% (miss 9.5, FA 3.6, confusion 5.7). Ours: 18.78% (miss 9.52, FA 3.58, confusion 5.68).
SA-EEND − pyannote on Eval-12 (P1): +31.46 points [27.80, 35.70], SA-EEND better on 0 of 12 meetings, Wilcoxon p = 0.00049.
