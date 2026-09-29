#!/usr/bin/env bash
# Progress of the running experiment: stage log, recent training output, GPU, wallet.
#   bash icl_compare/scripts/status.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/state_local/pod.env"
export PYTHONIOENCODING=utf-8 PRIME_DISABLE_VERSION_CHECK=1
SSHO=(-i "$HOME/.ssh/id_rsa" -p "$SSH_PORT" -o ConnectTimeout=20 -o BatchMode=yes)
if ! ssh "${SSHO[@]}" "$SSH_HOST" true 2>/dev/null; then
  echo "pod $POD_ID unreachable (terminated?)"; prime --plain pods status "$POD_ID" 2>&1 | head -5
else
  ssh "${SSHO[@]}" "$SSH_HOST" '
    cd ~/icl/icl_compare
    echo "== stages"; grep -E "stage|chose|CUT|->|ALL DONE|auto-terminate|failed" state/progress.log | tail -25
    echo "== latest output"; grep -E "step=|== |s ->|Error|Traceback" ~/run_all.log | tail -4
    echo "== gpu"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    df -h ~ | tail -1'
fi
prime --plain wallet 2>&1 | sed -n 2p
