# Offline RL experiment conventions

Read `src/ahawam/offline_rl/AGENTS.md` before changing preparation, caching,
splits or training. The current experiment's explicit episode recipe overrides
directory-based train/validation assignment. Never silently filter apparent
timing anomalies: highlight them to the user and maintain
`docs/offline_rl/19_DATA_ANOMALIES_ZH.md`.

`rebuild_pipeline.py` is the current four-GPU cache → validation/smoke → IQL and
TD3+BC queue. Check `runs/offline_rl/cylinder103_rebuild_20261026/pipeline.json`
before launching another supervisor or modifying code used by active workers.
