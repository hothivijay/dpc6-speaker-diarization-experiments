# Speaker diarization experiments: DISPLACE-2024 and AMI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hothivijay/dpc6-speaker-diarization-experiments/blob/main/notebooks/run_on_colab.ipynb)

This repository holds the code behind the measured results of a PhD progress review (DPC-6) on speaker
diarization ("who spoke when"). Every number can be regenerated with one script per step, and every script is
commented line by line.

| Experiment | What it answers | Code |
|---|---|---|
| **DER scorer with uncertainty** | What exactly does a DER number mean? (collar, overlap, CI, per recording) | `src/der.py` |
| **DISPLACE-2024 rescoring** | How do Sortformer, pyannote and NeMo clustering do on 32 multilingual far-field sessions, and where do the errors come from? | `src/run_systems.py`, `src/displace_rescore.py` |
| **SA-EEND on AMI** | Re-implementation of self-attentive end-to-end diarization (Fujita et al., ASRU 2019), trained and evaluated on AMI | `src/sa_eend.py`, `src/eend_eval.py` |
| **pyannote 3.1 on AMI** | Reproduction check against the published 18.8% | `src/run_pyannote.py` |
| **Edge deployment** | Speed, memory and DER change of an INT8 model on a CPU / Raspberry Pi 5 | `src/bench_edge.py` |

Results so far: [results/RESULTS.md](results/RESULTS.md).

---

## Quick start

### Option A: Google Colab (no installation)
1. Click **Open in Colab** above.
2. *Runtime → Change runtime type → T4 GPU*.
3. Run the cells from top to bottom. Each cell is one step and explains what it does. The full AMI pipeline takes
   about 2–3 hours on a free T4.

### Option B: your own Linux machine with an NVIDIA GPU
```bash
git clone https://github.com/hothivijay/dpc6-speaker-diarization-experiments.git
cd dpc6-speaker-diarization-experiments
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash tests/smoke_test.sh        # 1 minute, synthetic data: checks that every step runs
python -m pytest tests -q       # unit tests of the scorer and of SA-EEND

bash run_all.sh                 # all steps in order; run it again after any interruption
```
Data, models and results go to `./work` by default. Change this and every other setting in
[`steps/config.sh`](steps/config.sh), or override them with environment variables (`WORK=/big/disk bash run_all.sh`).

---

## The steps

Each step is a short, commented shell script in [`steps/`](steps). You can run steps one by one
(`bash steps/03_train_sa_eend.sh`) or all together (`bash run_all.sh`, or `bash run_all.sh 04 06` for a
selection). Finished steps are skipped, so after a disconnect you just start again.

| Step | Script | What it does | Output (under `WORK/`) | Time (T4) |
|---|---|---|---|---|
| 00 | `00_check_env.sh` | prints package versions, GPU and free disk | – | seconds |
| 01 | `01_download_ami.sh` | AMI headset-mix audio (170 meetings, ~10 GB) + BUT `only_words` references | `data/ami/` | 30–60 min |
| 02 | `02_extract_features.sh` | 23 log-Mel, ±7 context, ×10 subsampling → 345-dim per 100 ms; 4-speaker targets | `data/ami_feats/` | 10–20 min |
| 03 | `03_train_sa_eend.sh` | trains SA-EEND (4 blocks, 256 units, PIT-BCE, Noam) within a time budget (`MAXMIN`, default 40) | `exp/sa_eend_train/` | 40 min |
| 04 | `04_evaluate_sa_eend.sh` | chooses model average, window, threshold, median **on Dev**, freezes them, decodes Eval | `exp/ami_sys/sa_eend/` | 5–15 min |
| 05 | `05_pyannote_ami.sh` | pyannote 3.1 on AMI test (needs `HF_TOKEN`) | `exp/ami_sys/pyannote/` | 10–20 min |
| 06 | `06_score_ami.sh` | DER under P1/P2/C25 on 12 and 16 test meetings, CIs, paired test | `results/ami_systems.json` | 1 min |
| 07 | `07_edge_benchmark.sh` | ONNX FP32 + INT8 export; CPU-only pipeline benchmark; INT8 − FP32 DER | `exp/edge/` | 10–20 min |
| 08 | `08_displace_systems.sh` | *(licensed data)* Sortformer, NeMo clustering, NeMo + reference speech regions, pyannote on DISPLACE | `exp/displace_sys/` | hours |
| 09 | `09_displace_rescore.sh` | *(licensed data)* per-session scoring, CIs, paired tests, overlap floor | `results/displace_rescore.json` | 1 min |
| 10 | `10_collect_results.sh` | all headline numbers in one file, printed as a summary | `results/summary.json` | seconds |

### Step 05 needs a Hugging Face token
1. Create a token at <https://huggingface.co/settings/tokens>.
2. Accept the conditions of `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`.
3. pyannote.audio 3.3.2 needs torch 2.5.1 and numpy < 2, so give it its own environment:
   ```bash
   python3 -m venv pyenv && pyenv/bin/pip install -r requirements-pyannote.txt
   export PYANNOTE_PY=$PWD/pyenv/bin/python HF_TOKEN=hf_...
   bash steps/05_pyannote_ami.sh
   ```
Without a token, step 05 is skipped and everything else still runs.

### Steps 08–09 need the DISPLACE-2024 data
DISPLACE is distributed by the challenge organisers under a licence, so it is not included here. Set
`DISPLACE_AUDIO` and `DISPLACE_REF` (in `steps/config.sh` or the environment) to your copy and install NeMo
(`pip install "nemo_toolkit[asr]"`). See [docs/DATA.md](docs/DATA.md).

---

## How the evaluation is kept honest

* **The protocol is always stated.** P1 = no collar, overlap scored (DISPLACE rule, our default). P2 and C25 are
  reported beside it. See [docs/SCORING_PROTOCOLS.md](docs/SCORING_PROTOCOLS.md).
* **Uncertainty is always reported.** Pooled DER comes with a recording-bootstrap 95% CI. Comparisons use a
  paired bootstrap of the difference and a Wilcoxon signed-rank test.
* **Eval is never used for a choice.** SA-EEND's model average, window, threshold and median filter are picked on
  Dev and frozen before Eval is scored (`src/eend_eval.py`).
* **The scorer is tested.** It is checked against hand-computed cases and against `pyannote.metrics` (`tests/test_der.py`).
* **Edge numbers come from the deployable pipeline.** `bench_edge.py` never imports PyTorch and records whether it did.

## Repository layout
```
run_all.sh                 run every step in order (resumable)
steps/config.sh            all paths and settings
steps/00_ … 10_*.sh        one commented script per step
src/der.py                 DER, protocols P1/P2/C25, bootstrap CI, paired tests
src/eend_core.py           NumPy features, chunk linking, post-processing (shared by training and edge)
src/sa_eend.py             SA-EEND model, PIT loss, training, inference, ONNX export
src/eend_eval.py           Dev grid search, freeze, Eval decoding and scoring
src/run_pyannote.py        pyannote 3.1 on AMI
src/ami_prep.py            AMI download + BUT references
src/ami_score.py           AMI Eval scoring of all systems
src/bench_edge.py          CPU / Raspberry Pi benchmark, FP32 vs INT8
src/run_systems.py         DISPLACE systems: Sortformer, NeMo clustering (+ reference speech regions), pyannote
src/displace_rescore.py    DISPLACE per-session scoring and overlap floor
src/collect_results.py     summary of all numbers
notebooks/run_on_colab.ipynb  the same steps as a Colab notebook
tests/                     unit tests, synthetic smoke test
docs/                      protocols, data, Raspberry Pi guide
results/                   measured numbers of the reference run (scores only, no data)
```

## Running on a Raspberry Pi 5
See [docs/RASPBERRY_PI.md](docs/RASPBERRY_PI.md). Only `numpy`, `scipy`, `soundfile` and `onnxruntime` are needed
(`requirements-edge.txt`).

## References
* Y. Fujita, N. Kanda, S. Horiguchi, Y. Xue, K. Nagamatsu, S. Watanabe. *End-to-end neural speaker diarization with
  self-attention.* IEEE ASRU 2019.
* H. Bredin. *pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe.* Interspeech 2023;
  model `pyannote/speaker-diarization-3.1`.
* NVIDIA NeMo: clustering diarizer (MarbleNet, TitaNet-L, NME-SC) and Streaming Sortformer v2.
* J. Carletta et al. *The AMI Meeting Corpus.* MLMI 2005. References from
  [BUTSpeechFIT/AMI-diarization-setup](https://github.com/BUTSpeechFIT/AMI-diarization-setup).
* DISPLACE-2024 challenge (speaker and language diarization of code-mixed conversational speech).

## License
Code: MIT (see `LICENSE`). Datasets keep their own licences (AMI: CC BY 4.0; DISPLACE: challenge licence).
