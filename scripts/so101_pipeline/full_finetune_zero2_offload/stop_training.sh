#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PID="$(cat runs/so101_ahawam_square_v1/latest_pid.txt)"
kill -- "-$PID"
echo "Stopped training process group $PID"

