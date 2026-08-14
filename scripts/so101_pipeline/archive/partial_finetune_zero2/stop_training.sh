#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
PID="$(cat runs/so101_g05_square_v1/latest_pid.txt)"
kill -- "-$PID"
echo "Stopped training process group $PID"
