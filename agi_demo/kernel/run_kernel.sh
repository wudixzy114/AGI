#!/bin/bash
# Project 4 — from-scratch fast-weight kernel, full experiment matrix on the B200.
# NO pretrained weights needed (this kernel trains from random init), so no model-hub path.
# Everything is small; the B200 runs configs in parallel to finish fast.
set -u
cd /media/cfs/xiezongyu.1/AGI
PY=/opt/conda/bin/python
OUTBASE=agi_demo/outputs/kernel
# Scale up from the Mac-smoke dims: wider kernel, more state, 4 timescales, more steps + seeds.
COMMON="--d-model 128 --d-state 64 --n-layers 4 --decays 0,0.7,0.95,0.99 \
  --batch-size 128 --device cuda"
FILTER='it/s\]|ConvergenceWarning|warnings.warn|n_iter_i|building the font|UserWarning'
SEEDS=0,1,2,3,4

run () {
  local out="$OUTBASE/$1"; shift
  mkdir -p "$out"
  echo "===== $(date +%H:%M:%S)  START $out  flags: $* ====="
  stdbuf -oL -eL $PY -u -m agi_demo.kernel.run $COMMON "$@" --out-dir "$out" 2>&1 \
    | grep --line-buffered -vE "$FILTER" | tee "$out/run.log" >/dev/null
  echo "===== $(date +%H:%M:%S)  DONE $out ====="
}

# --- BATCH 1: the headline memory-USE study at 3 difficulty levels (K0/K1/K2 each) ---
# hops=0 pure recall (cleanest USE test) | hops=1 recall+op | hops=2 recall+chain
echo "########## BATCH 1: session hops-sweep ##########"
run session_hops0 --session --session-hops 0 --session-len 6 --n-slots 8 --steps 1500 --lr 3e-3 &
run session_hops1 --session --session-hops 1 --session-len 6 --n-slots 8 --steps 2500 --lr 3e-3 &
run session_hops2 --session --session-hops 2 --session-len 6 --n-slots 8 --steps 4000 --lr 3e-3 &
wait

# --- BATCH 2: multi-seed headline (pure recall) + delta-rule ablation ---
echo "########## BATCH 2: multi-seed + delta ablation ##########"
run session_seeds --session --session-hops 0 --session-len 6 --n-slots 8 --steps 1500 --lr 3e-3 --seeds $SEEDS &
run session_hebbian --session --session-hops 1 --session-len 6 --n-slots 8 --steps 2500 --lr 3e-3 --no-delta &
wait

# --- BATCH 3: longer sessions (recall across bigger delay) — stresses the slow timescale ---
echo "########## BATCH 3: long-session recall ##########"
run session_long --session --session-hops 0 --session-len 10 --n-slots 12 --steps 2500 --lr 3e-3 &
wait

# --- BATCH 4: arith secondary (length generalization + per-step probe), with curriculum ---
echo "########## BATCH 4: arith length-gen ##########"
run arith_curric --arith --arm K1 --curriculum --train-hops 1,2,3,4 --eval-hops 1,2,3,4,5,6,7,8 \
  --steps 8000 --lr 2e-3 --curriculum-patience 800 &
run arith_uniform --arith --arm K1 --train-hops 1,2,3,4 --eval-hops 1,2,3,4,5,6,7,8 \
  --steps 8000 --lr 2e-3 &
wait

echo "ALL_KERNEL_DONE $(date +%H:%M:%S)"
