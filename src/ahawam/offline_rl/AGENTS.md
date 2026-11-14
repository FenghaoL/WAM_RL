# Offline RL data decisions

These are user-confirmed project conventions, updated 2026-10-26. Follow newer
explicit user instructions if the experiment changes.

- Treat each cylinder HIL episode as one continuous sequence of saved frames.
  Collector tick jumps include omitted takeover pauses; do not reset history or
  filter transitions solely for tick jumps or `dropped_tick` on a saved row.
- The current recipe is
  `runs/offline_rl/cylinder103_rebuild_20261026/data/episode_recipe.json`.
  Respect its explicit episode splits and retained-frame limits. Directory
  suffix `_val` is not authoritative: one episode has moved to training.
- Nine recordings with unobserved success enter training. Four use only their
  first half. Their observed rewards are zero, but recording cutoffs bootstrap;
  never fabricate a success reward or failure terminal at these cutoffs.
- Keep POLICY and HUMAN actions; omit only SYSTEM_HOLD actions. History may use
  saved HOLD observations. Never cross episode or retained-prefix boundaries.
- Preserve raw recordings and annotation identities. Cache lineage must match
  the selected timeline and frame limits; old cache features cannot be repaired
  by changing metadata.
- User explicitly requests that unexpected data/annotation/timing behavior be
  highlighted in the conversation, with evidence, affected artifacts and the
  proposed treatment. Do not silently reinterpret or discard such cases.
  Record findings in `docs/offline_rl/19_DATA_ANOMALIES_ZH.md`.

Details: `docs/offline_rl/18_REBUILD_AND_EPISODE_RECIPE_ZH.md`.
