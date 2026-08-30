---
pretty_name: AHA-WAM SO101 Plug HIL Training Assets
tags:
  - robotics
  - imitation-learning
  - human-in-the-loop
  - so101
  - aha-wam
license: mit
---

# AHA-WAM SO101 Plug HIL Training Assets

Reproducibility bundle for the SO101 power-adapter insertion experiments.

It contains a compact, ZIP-based representation of the paths expected by the
AHA-WAM training configuration:

- the released `AHA-WAM-pretrained.pt` initialization checkpoint;
- `so101_ahawam_plug.zip` (the downloader extracts only task 02 and 04);
- `ahawam_hil_raw.zip` (raw rich-v1/v2/v3 HG-DAgger recordings);
- the fixed 02+04 action/state normalization statistics;
- cached T5 text embeddings required by training.

The downloader verifies fixed archive sizes and SHA-256 digests, rejects unsafe
ZIP members, and restores only the audited prefixes to their default `data/`
paths. `so101_ahawam_plug.zip` also carries task 01, but task 01 is intentionally
not extracted for these experiments.

The raw HIL recordings include POLICY and SYSTEM_HOLD rows for auditability.
The training loader must use the code repository's HUMAN-only logical view. Do
not train directly on every raw HIL row.

After cloning the code repository, either 5-epoch launcher downloads missing
assets from this public dataset to their default paths before preflight. No
Hugging Face login is required for downloading:

```bash
scripts/so101_pipeline/hil_human_only_training/run_natural_5ep.sh
scripts/so101_pipeline/hil_human_only_training/run_balanced_2plus2_5ep.sh
```

The publisher is responsible for reviewing robot camera recordings for
privacy-sensitive content and confirming checkpoint/data license compatibility.
