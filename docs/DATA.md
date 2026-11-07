# Data

This repository has code and scores only. No audio or reference annotation is committed.

## AMI Meeting Corpus (downloaded by step 01)

* **Audio**: headset mix (`<meeting>.Mix-Headset.wav`, 16 kHz mono), 170 meetings, about 10 GB, from the
  official Edinburgh mirror (`groups.inf.ed.ac.uk/ami`). Licence: CC BY 4.0. Please cite the AMI corpus
  (Carletta et al., 2005).
* **References**: [BUTSpeechFIT/AMI-diarization-setup](https://github.com/BUTSpeechFIT/AMI-diarization-setup),
  `only_words` RTTMs and UEMs, Full-corpus-ASR partition: train 136, dev 18, test 16 meetings.
* **Splits used**
  * `dev_noTNO` (14) and `test_noTNO` (12): TNO meetings (`TS30xx`) removed, as in the SpeechBrain AMI recipe.
    This is the thesis split, called **eval12**.
  * `test` (16): the full test set used for the published pyannote numbers, called **eval16**.

## DISPLACE-2024 (not downloadable here)

DISPLACE-2024 Track-1 (speaker diarization in multilingual, conversational far-field speech) is distributed by
the challenge organisers under a licence. Steps 08 and 09 run only when you set `DISPLACE_AUDIO` and
`DISPLACE_REF` to your own copy. `results/displace_summary.json` holds our scores only.
