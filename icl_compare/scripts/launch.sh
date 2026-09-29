#!/usr/bin/env bash
# Launch the full experiment (run_all.py) on a fresh Prime Intellect GPU pod.
# Run from Git Bash on the laptop, anywhere inside the project:
#
#   bash icl_compare/scripts/launch.sh                   # A100 80GB, auto-terminate
#   bash icl_compare/scripts/launch.sh --gpu A6000_48GB  # if no A100 is in stock
#   bash icl_compare/scripts/launch.sh --no-auto-terminate
#
# Steps: pick the best offer for --gpu, create the pod, upload the code and
# .env, install dependencies, then start run_all.py under nohup. The pod id and
# SSH target are saved to icl_compare/state_local/pod.env for status.sh and
# fetch_results.sh.
#
# Your Prime API key is written to the pod (state/pod.env, never uploaded) so
# run_all.py can watch the wallet (stop + upload below --min-balance) and, with
# auto-terminate (the default), delete the pod once everything is on W&B.
# Without auto-terminate the pod keeps billing until you terminate it.
set -euo pipefail
GPU=A100_80GB AUTO=1 MIN_BALANCE=2 EST_HOURS=21.5
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU=$2; shift 2 ;;
    --min-balance) MIN_BALANCE=$2; shift 2 ;;
    --no-auto-terminate) AUTO=0; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done

HERE=$(cd "$(dirname "$0")/.." && pwd)          # icl_compare
PROJ=$(cd "$HERE/.." && pwd)
LOCAL="$HERE/state_local"; mkdir -p "$LOCAL"
export PYTHONIOENCODING=utf-8 PRIME_DISABLE_VERSION_CHECK=1
KEY="$HOME/.ssh/id_rsa"
[ -f "$PROJ/.env" ] || { echo "missing $PROJ/.env (WANDB_API_KEY)"; exit 1; }

echo "== offers for $GPU"
read -r OFFER PRICE <<<"$(prime --plain availability list --gpu-type "$GPU" --gpu-count 1 \
    --no-group-similar --output json 2>/dev/null | python -c "
import json, sys
rs = [r for r in json.load(sys.stdin)['gpu_resources'] if r['stock_status'] != 'Unavailable']
rs.sort(key=lambda r: r['price_value'] * (0.9 if r['socket'] in ('SXM4', 'SXM5') else 1.0))  # SXM runs ~10-15% faster
print(rs[0]['id'], rs[0]['price_value']) if rs else print('none 0')")"
if [ "$OFFER" = none ]; then
  echo "no $GPU available right now. Try again later, or pass --gpu A6000_48GB (slower, similar cost)."
  exit 1
fi
BALANCE=$(prime --plain wallet 2>/dev/null | sed -n 's/.*Balance: *[$]\([0-9.]*\).*/\1/p')
EST=$(python -c "print(round($EST_HOURS * $PRICE, 2))")
echo "offer $OFFER at \$$PRICE/h; expected ~$EST_HOURS h = \$$EST; wallet \$${BALANCE:-?}"
if ! python -c "import sys; sys.exit(0 if float('${BALANCE:-0}') >= $EST * 1.1 + $MIN_BALANCE else 1)"; then
  echo "WARNING: wallet is below the expected cost + 10% + \$$MIN_BALANCE reserve."
  echo "The run stops and uploads (salvage) if the balance falls below \$$MIN_BALANCE."
  read -r -p "Continue anyway? [y/N] " ans; [ "$ans" = y ] || exit 1
fi

echo "== creating pod"
POD=""
for img in ubuntu_22_cuda_12 cuda_12_6_pytorch_2_7 cuda_12_4_pytorch_2_6; do
  out=$(prime --plain pods create --id "$OFFER" --image "$img" --name icl-full -y 2>&1 || true)
  POD=$(echo "$out" | sed -n 's/.*Successfully created pod \([0-9a-f]*\).*/\1/p')
  [ -n "$POD" ] && { echo "pod $POD (image $img)"; break; }
  echo "$out" | tail -2
done
[ -n "$POD" ] || { echo "pod creation failed"; exit 1; }
printf 'POD_ID=%s\nPRICE=%s\n' "$POD" "$PRICE" > "$LOCAL/pod.env"

echo "== waiting for the pod"
for i in $(seq 1 60); do
  st=$(prime --plain pods status "$POD" 2>&1)
  if echo "$st" | grep -q "ACTIVE" && echo "$st" | grep -q "Installation Status *FINISHED"; then break; fi
  sleep 15
done
SSH_STR=$(prime --plain pods status "$POD" 2>&1 | sed -n 's/^SSH *//p' | sed 's/ *$//')
SSH_HOST=$(echo "$SSH_STR" | awk '{print $1}')
SSH_PORT=$(echo "$SSH_STR" | sed -n 's/.*-p *\([0-9]*\).*/\1/p'); SSH_PORT=${SSH_PORT:-22}
[ -n "$SSH_HOST" ] || { echo "no SSH target yet; check: prime pods status $POD"; exit 1; }
printf 'SSH_HOST=%s\nSSH_PORT=%s\n' "$SSH_HOST" "$SSH_PORT" >> "$LOCAL/pod.env"
echo "ssh $SSH_HOST -p $SSH_PORT"
# Prime reuses IPs across pods: drop any old host key for this address first.
ssh-keygen -R "${SSH_HOST#*@}" >/dev/null 2>&1 || true
SSHO=(-i "$KEY" -p "$SSH_PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20
      -o BatchMode=yes -o ServerAliveInterval=30)
for i in $(seq 1 20); do ssh "${SSHO[@]}" "$SSH_HOST" true 2>/dev/null && break; sleep 10; done

echo "== uploading code"
TGZ=$(mktemp -u).tgz
(cd "$PROJ" && tar --exclude='.git' --exclude='*.pdf' --exclude='*.png' --exclude='__pycache__' \
   --exclude='icl_compare/ckpts*' --exclude='icl_compare/results' --exclude='icl_compare/data' \
   --exclude='icl_compare/state*' --exclude='icl_compare/wandb' \
   --exclude='self_play_pretraining/scoring/data' --exclude='self_play_pretraining/figures' \
   -czf "$TGZ" icl_compare self_play_pretraining .env)
scp -q -i "$KEY" -P "$SSH_PORT" -o BatchMode=yes "$TGZ" "$SSH_HOST:code.tgz"
rm -f "$TGZ"

echo "== installing dependencies (~3 min)"
ssh "${SSHO[@]}" "$SSH_HOST" 'bash -s' < "$HERE/scripts/setup_pod.sh" 2>&1 | tail -3

PRIME_KEY=$(python -c "import json, pathlib; print(json.loads((pathlib.Path.home()/'.prime/config.json').read_text())['api_key'])")
printf 'POD_ID=%s\nPRIME_API_KEY=%s\n' "$POD" "$PRIME_KEY" |
  ssh "${SSHO[@]}" "$SSH_HOST" 'mkdir -p ~/icl/icl_compare/state && cat > ~/icl/icl_compare/state/pod.env && chmod 600 ~/icl/icl_compare/state/pod.env'
AUTO_FLAG=""; [ "$AUTO" = 1 ] && AUTO_FLAG="--auto-terminate"

echo "== starting run_all.py"
ssh "${SSHO[@]}" "$SSH_HOST" "cd ~/icl/icl_compare; setsid nohup python3 run_all.py --price $PRICE --min-balance $MIN_BALANCE $AUTO_FLAG > ~/run_all.log 2>&1 < /dev/null & echo started"
cat <<EOF

Launched. Pod $POD at \$$PRICE/h (~$EST_HOURS h expected), auto-terminate=$AUTO.
  progress:  bash icl_compare/scripts/status.sh
  results:   bash icl_compare/scripts/fetch_results.sh
  W&B:       your WANDB_PROJECT (default icl-selfplay) at https://wandb.ai
  stop:      prime pods terminate $POD
EOF
