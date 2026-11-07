# config.sh - every path and setting used by the step scripts, in one place.
# Each step script does `source steps/config.sh`. Override any value with an environment variable, e.g.
#   WORK=/content/work MAXMIN=60 bash steps/03_train_sa_eend.sh

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # repository root
SRC=$REPO/src                                           # Python code

# ---- where data, models and results go (about 15 GB in total with all AMI audio)
WORK=${WORK:-$REPO/work}
DATA=${DATA:-$WORK/data}
AMI=${AMI:-$DATA/ami}               # AMI audio, references, UEMs, lists      (step 01)
FEATS=${FEATS:-$DATA/ami_feats}     # SA-EEND features, one .npz per meeting  (step 02)
EXP=${EXP:-$WORK/exp}               # checkpoints and system outputs          (steps 03-08)
RESULTS=${RESULTS:-$WORK/results}   # scored JSON files and summary           (steps 06, 09, 10)

# ---- Python interpreters
PY=${PY:-python3}                   # main environment (requirements.txt)
PYANNOTE_PY=${PYANNOTE_PY:-$PY}     # environment with pyannote.audio 3.3.2 (requirements-pyannote.txt)

# ---- SA-EEND training (values used for the reported run)
EPOCHS=${EPOCHS:-150}               # upper limit on epochs
MAXMIN=${MAXMIN:-75}                # time budget in minutes (training stops after the epoch that passes it)
BATCH=${BATCH:-32}
CHUNK=${CHUNK:-500}                 # 500 frames x 100 ms = 50 s training chunks
WARMUP=${WARMUP:-4000}              # Noam warm-up steps
LAYERS=${LAYERS:-4}                 # Transformer blocks
AVG=${AVG:-10}                      # checkpoints averaged

# ---- edge benchmark
EDGE_THREADS=${EDGE_THREADS:-2}     # 2 on Colab (2 vCPUs); use 4 on a Raspberry Pi 5

# ---- DISPLACE-2024 (licensed; NOT included in this repository). Leave empty to skip steps 08-09.
DISPLACE_AUDIO=${DISPLACE_AUDIO:-}  # folder with the Track-1 eval recordings (searched recursively)
DISPLACE_REF=${DISPLACE_REF:-}      # folder with the Track-1 speaker-diarization eval RTTMs

mkdir -p "$DATA" "$EXP" "$RESULTS"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false
