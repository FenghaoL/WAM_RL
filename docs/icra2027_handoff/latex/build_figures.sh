#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
"${PYTHON_BIN:-python3}" figures/build_rollout_figures.py
cd figures
for source in rollout_comparison.tex rollout_comparison_all_frames.tex method_overview_standalone.tex; do
  if [[ -n "${TECTONIC_BIN:-}" ]]; then
    "$TECTONIC_BIN" --keep-logs "$source"
  elif command -v tectonic >/dev/null 2>&1; then
    tectonic --keep-logs "$source"
  else
    latexmk -pdf -interaction=nonstopmode -halt-on-error "$source"
  fi
done
cp method_overview_standalone.pdf method_overview.pdf
