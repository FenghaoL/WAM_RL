#!/usr/bin/env bash
set -euo pipefail
export HIL_SAMPLING_STRATEGY=natural
export HIL_RUN_VARIANT=natural-concat
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_common.sh"
