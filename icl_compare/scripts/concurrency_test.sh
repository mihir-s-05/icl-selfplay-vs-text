#!/usr/bin/env bash
# Does packing several trainings onto one GPU raise total throughput?
# Solo baselines first, then concurrent groups, all timed over the same 60 s
# window after compile warmup. Results: results/bench_concurrency.jsonl.
#   cd ~/icl/icl_compare && bash scripts/concurrency_test.sh "1M=64 3M=64 6M=64 24M=32"
# The argument sets each size's micro-batch; pick values so all four fit in
# memory together (80 GB A100: as above; 48 GB: "1M=64 3M=32 6M=32 24M=16").
set -euo pipefail
OUT=results/bench_concurrency.jsonl
DUR=60
declare -A MB
for kv in ${1:-1M=64 3M=64 6M=64 24M=32}; do MB[${kv%%=*}]=${kv#*=}; done

one() {  # size tag start_at
  python3 bench.py --sizes "$1" --micro-bs "${MB[$1]}" --compile --warmup 3 \
    --start-at "$3" --duration "$DUR" --tag "$2" --out "$OUT" 2>&1 | grep -E "tok/s|OOM|Error"
}

group() {  # tag sizes...
  local tag=$1; shift
  local start=$(( $(date +%s) + 90 ))    # compile finishes well within 90 s
  echo "== $tag: $*"
  for s in "$@"; do one "$s" "$tag" "$start" & done
  wait
}

for s in 1M 3M 6M 24M; do group "solo" "$s"; done
group "small3" 1M 3M 6M
group "all4" 1M 3M 6M 24M
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
