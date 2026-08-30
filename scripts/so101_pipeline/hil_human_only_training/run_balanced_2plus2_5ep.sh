#!/usr/bin/env bash
set -euo pipefail
export HIL_SAMPLING_STRATEGY=balanced_2_2
export HIL_RUN_VARIANT=balanced-2demo-2human
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_common.sh"
