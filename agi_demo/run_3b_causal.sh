#!/bin/bash
# Causal-ablation (①) + robustness (②) matrix on the B200. Sequential, one GPU.
# Writes each config to its own dir under outputs/3b_causal/. Streams live (python -u + line-buffered).
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

run () {  # $1 = out subdir ; $2... = extra flags
  local out="$OUTBASE/$1"; shift
  mkdir -p "$out"
  echo "===== $(date +%H:%M:%S)  START $out  flags: $* ====="
  HF_HUB_OFFLINE=1 stdbuf -oL -eL $PY -u -m agi_demo.run $COMMON "$@" --out-dir "$out" 2>&1 \
    | grep --line-buffered -vE "$FILTER" | tee "$out/run.log"
  echo "===== $(date +%H:%M:%S)  DONE $out ====="
}

# ---- ① CAUSAL ABLATION: arm C, vary WHICH latent steps get process supervision (arith) ----
# process_steps: none(=arm B) / first_half / second_half / all.  Does decodable-subset -> OOD?
run causal_none        --arm B                              # baseline: no process CE
run causal_first_half  --arm C --process-steps first_half   # only shallow steps supervised
run causal_second_half --arm C --process-steps second_half  # only deep steps supervised
run causal_all         --arm C --process-steps all          # full chain supervised

# ---- ② ROBUSTNESS: 3-seed error bars on the core comparison (arith) ----
run seeds_A  --arm A --seeds $SEEDS
run seeds_B  --arm B --seeds $SEEDS
run seeds_C  --arm C --process-steps all --seeds $SEEDS

# ---- ② SECOND TASK FAMILY: multi-hop lookup (perm), core arms ----
run perm_A --arm A --task perm
run perm_B --arm B --task perm
run perm_C --arm C --task perm --process-steps all

echo "ALL_CAUSAL_DONE $(date +%H:%M:%S)"
