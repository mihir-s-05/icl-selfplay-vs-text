#!/usr/bin/env bash
# Replace the code of a running run_all.py pipeline without losing work.
#   bash icl_compare/scripts/hotswap.sh
#
# 1. Uploads icl_compare's code (not state/, ckpts*/, results/, data/).
# 2. Waits for a safe moment: if a training job is running, until it has just
#    written resume.pt (so at most ~1 min of progress is lost); an ICL eval is
#    stopped right away (finished checkpoints are kept, the current one redone).
# 3. Stops run_all.py and its child, deletes any unreadable result JSON, and
#    restarts run_all.py with the same arguments. Finished stages are skipped.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
PROJ=$(cd "$HERE/.." && pwd)
source "$HERE/state_local/pod.env"
SSHO=(-i "$HOME/.ssh/id_rsa" -p "$SSH_PORT" -o ConnectTimeout=20 -o BatchMode=yes -o ServerAliveInterval=30)

TGZ=$(mktemp -u).tgz
(cd "$PROJ" && tar --exclude='__pycache__' --exclude='icl_compare/ckpts*' --exclude='icl_compare/results' \
   --exclude='icl_compare/data' --exclude='icl_compare/state*' --exclude='icl_compare/wandb' \
   -czf "$TGZ" icl_compare)
scp -q -i "$HOME/.ssh/id_rsa" -P "$SSH_PORT" -o BatchMode=yes "$TGZ" "$SSH_HOST:code_update.tgz"
rm -f "$TGZ"
echo "uploaded code"

ssh "${SSHO[@]}" "$SSH_HOST" 'bash -s' <<'REMOTE'
set -uo pipefail
cd ~/icl/icl_compare
ARGS=$(pgrep -af "^python3 run_all.py" | head -1 | sed 's/.*run_all.py//')
[ -n "$ARGS" ] || { echo "run_all.py is not running; restart it by hand"; exit 1; }
echo "running with:$ARGS"
child=$(pgrep -af "train.py|icl.evaluate_arm" | grep -v pgrep | head -1)
if echo "$child" | grep -q train.py; then
  run_dir=$(echo "$child" | python3 -c "
import sys, re
a = sys.stdin.read().split()
g = lambda k, d=None: a[a.index(k) + 1] if k in a else d
print(f\"{g('--out', 'ckpts')}/{g('--arm', 'dclm')}/{g('--size')}/seed-{g('--seed', '0')}\")")
  echo "training job running in $run_dir; waiting for its next resume.pt save (<= 15 min)"
  before=$(stat -c %Y "$run_dir/resume.pt" 2>/dev/null || echo 0)
  for i in $(seq 1 200); do
    now=$(stat -c %Y "$run_dir/resume.pt" 2>/dev/null || echo 0)
    [ "$now" != "$before" ] && break
    pgrep -f train.py >/dev/null || break
    sleep 5
  done
  sleep 3
fi
pkill -f "python3 run_all.py" || true
pkill -f "train.py" || true
pkill -f "icl.evaluate_arm" || true
for i in $(seq 1 30); do pgrep -f "run_all.py|train.py|icl.evaluate_arm" >/dev/null || break; sleep 2; done
tar -xzf ~/code_update.tgz -C ~/icl && echo "code updated"
python3 - <<'PY'
import json, pathlib
bad = []
for p in pathlib.Path("results").rglob("*.json"):
    try:
        json.loads(p.read_text())
    except ValueError:
        bad.append(p); p.unlink()
print(f"checked result files; removed {len(bad)} unreadable: {bad}")
PY
setsid nohup python3 run_all.py $ARGS >> ~/run_all.log 2>&1 < /dev/null &
sleep 20
grep -v '\$ ' state/progress.log | tail -4
REMOTE
