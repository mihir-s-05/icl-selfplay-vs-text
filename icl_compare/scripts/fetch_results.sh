#!/usr/bin/env bash
# Copy results, logs and DCLM checkpoints from the pod into icl_compare/.
# (Everything is also uploaded to W&B at the end of the run, so this is only
# needed for an early look or if the upload failed.)
#   bash icl_compare/scripts/fetch_results.sh            # results + logs only
#   bash icl_compare/scripts/fetch_results.sh --ckpts    # also DCLM checkpoints
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/state_local/pod.env"
WHAT="results state ckpts/dclm/*/seed-0/log.jsonl ckpts/sp2dclm/*/seed-0/log.jsonl ckpts_sweep/*/*/seed-0/log.jsonl"
[ "${1:-}" = "--ckpts" ] && WHAT="$WHAT ckpts/dclm ckpts/sp2dclm"
ssh -i "$HOME/.ssh/id_rsa" -p "$SSH_PORT" -o BatchMode=yes "$SSH_HOST" \
  "cd ~/icl/icl_compare && tar --exclude='state/pod.env' --exclude='resume.pt' -czf - $WHAT 2>/dev/null" |
  tar -xzf - -C "$HERE"
echo "fetched into $HERE: $WHAT"
