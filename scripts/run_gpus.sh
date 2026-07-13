#!/usr/bin/env bash
# Run the benchmark suite across all GPUs, one shard per GPU.
#
#   scripts/run_gpus.sh [N_GPUS] [extra run.py args]
#   scripts/run_gpus.sh 8
#   scripts/run_gpus.sh 8 --time-budget 10 --samples 40
set -euo pipefail
cd "$(dirname "$0")/.."

N=${1:-8}; shift || true
[ "${1:-}" = "--" ] && shift || true   # allow an optional `--` separator

# FPTaylor needs its opam env; load it if available so the bound step works in each shard.
if command -v opam >/dev/null 2>&1; then eval "$(opam env --switch=fptaylor 2>/dev/null || opam env)"; fi

mkdir -p log
echo "launching $N shards over $N GPUs; logs in log/gpu*.log"
pids=()
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=$i uv run src/run.py --shard "$i/$N" "$@" \
    > "log/gpu$i.log" 2>&1 &
  pids+=($!)
done

# Wait for all shards; report any that failed but don't abort the rest.
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "shard $i done"; else echo "shard $i FAILED (see log/gpu$i.log)"; fail=1; fi
done

echo; echo "===== combined summary ====="
# forward "$@" so --run/--decoding pick the same subdir the shards wrote to
uv run src/run.py --summary-only "$@"
exit $fail
