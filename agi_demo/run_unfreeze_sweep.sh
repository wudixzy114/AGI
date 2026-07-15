#!/bin/bash
# Attribution sweep, PARALLELIZED: does unfreezing a few base layers let the model USE latent memory?
# M2 (address-sup + decode-re-embed), unfreeze_last_n in {0,1,2,4}. Each config ~46GB, so run
# 2-at-a-time (batches of 2 ≈ 92GB of 183GB, safe headroom). nohup-safe, streaming logs.
set -u
cd /media/cfs/xiezongyu.1/AGI
PY=/opt/conda/bin/python
MODEL=/media/cfs/9n-das-admin/llm_models/Qwen2.5-3B-Instruct
OUTBASE=agi_demo/outputs/3b_unfreeze
COMMON="--session --arm M2 --local-model-dir $MODEL --dtype bfloat16 --batch-size 128 --steps 600"
FILTER='Loading checkpoint|it/s\]|FutureWarning|warnings.warn|ConvergenceWarning|n_iter_i'

run () {  # $1 = N
  local N=$1; local out="$OUTBASE/unfreeze_$N"; mkdir -p "$out"
  echo "===== $(date +%H:%M:%S) START unfreeze_last_n=$N ====="
  HF_HUB_OFFLINE=1 stdbuf -oL -eL $PY -u -m agi_demo.run $COMMON --unfreeze-last-n $N --out-dir "$out" 2>&1 \
    | grep --line-buffered -vE "$FILTER" | tee "$out/run.log" >/dev/null
  echo "===== $(date +%H:%M:%S) DONE unfreeze_last_n=$N ====="
}

# unfreeze_0 (frozen baseline) already known to stay at chance across many runs; focus compute on
# whether plasticity HELPS. Run 0 (as clean control) + 2 in batch1, then 4 + 6 in batch2.
echo "########## BATCH 1: unfreeze 0 + 2 ##########"
run 0 & run 2 & wait
echo "########## BATCH 2: unfreeze 4 + 6 ##########"
run 4 & run 6 & wait
echo "ALL_UNFREEZE_DONE $(date +%H:%M:%S)"
