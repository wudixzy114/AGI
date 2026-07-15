#!/bin/bash
# Parallel driver — runs the REMAINING 9 configs two-at-a-time on the single B200
# (single-config util ~50%, mem ~39/183GB, so 2 concurrent fills the gaps ~2x throughput).
# causal_none already ran separately; this script does the rest.
set -u
cd /media/cfs/xiezongyu.1/AGI
PY=/opt/conda/bin/python
MODEL=/media/cfs/9n-das-admin/llm_models/Qwen2.5-3B-Instruct
OUTBASE=agi_demo/outputs/3b_causal
COMMON="--local-model-dir $MODEL --dtype bfloat16 --batch-size 256 \
  --train-hops 1,2,3,4,5,6 --eval-hops 1,2,3,4,5,6,7,8 --steps 3000 \
  --curriculum --curriculum-patience 300 --process-warmup 400"
FILTER='Loading checkpoint|it/s\]|ConvergenceWarning|warnings.warn|STOP:|n_iter_i|building the font'
SEEDS=0,1,2

# one training run -> its own out-dir + streaming run.log
run () {
  local out="$OUTBASE/$1"; shift
  mkdir -p "$out"
  echo "===== $(date +%H:%M:%S)  START $out  flags: $* ====="
  HF_HUB_OFFLINE=1 stdbuf -oL -eL $PY -u -m agi_demo.run $COMMON "$@" --out-dir "$out" 2>&1 \
    | grep --line-buffered -vE "$FILTER" | tee "$out/run.log" >/dev/null
  echo "===== $(date +%H:%M:%S)  DONE $out ====="
}

# Each batch launches two configs concurrently (&) then waits for both (barrier) before the next.
echo "########## PARALLEL BATCH 1 ##########"
run causal_first_half  --arm C --process-steps first_half  &
run causal_second_half --arm C --process-steps second_half &
wait

echo "########## PARALLEL BATCH 2 ##########"
run causal_all --arm C --process-steps all &
run seeds_A    --arm A --seeds $SEEDS      &
wait

echo "########## PARALLEL BATCH 3 ##########"
run seeds_B --arm B --seeds $SEEDS            &
run seeds_C --arm C --process-steps all --seeds $SEEDS &
wait

echo "########## PARALLEL BATCH 4 ##########"
run perm_A --arm A --task perm &
run perm_B --arm B --task perm &
wait

echo "########## PARALLEL BATCH 5 ##########"
run perm_C --arm C --task perm --process-steps all &
wait

echo "ALL_CAUSAL_DONE $(date +%H:%M:%S)"
