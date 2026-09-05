#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PID="$(cat runs/so101_ahawam_padapter_v2/latest_flash_pid.txt)"
kill -- "-$PID"
echo "Stopped Flash-distillation process group $PID"

