#!/bin/bash
# Full 3B experiment matrix on the B200. Runs sequentially (one GPU), each config to its
# own output dir. Logs per-config so we can pull + inspect. Deeper task: train hops 1-6, eval to 8.
set -u
cd /media/cfs/xiezongyu.1/AGI
PY=/opt/conda/bin/python
MODEL=/media/cfs/9n-das-admin/llm_models/Qwen2.5-3B-Instruct
COMMON="--local-model-dir $MODEL --dtype bfloat16 --batch-size 256 \
  --train-hops 1,2,3,4,5,6 --eval-hops 1,2,3,4,5,6,7,8 --steps 3000 \
  --curriculum-patience 300 --process-warmup 400"
FILTER='Loading checkpoint|it/s\]|ConvergenceWarning|warnings.warn|STOP:|n_iter_i|building the font'

run () {  # $1 = out subdir, $2... = extra flags
  local out="agi_demo/outputs/3b/$1"; shift
  mkdir -p "$out"
  echo "===== $(date +%H:%M:%S)  START $out  flags: $* ====="
  # python -u + line-buffered grep so run.log streams live (watchable via tail -f)
  HF_HUB_OFFLINE=1 stdbuf -oL -eL $PY -u -m agi_demo.run $COMMON "$@" --out-dir "$out" 2>&1 \
    | grep --line-buffered -vE "$FILTER" | tee "$out/run.log"
  echo "===== $(date +%H:%M:%S)  DONE $out ====="
}

# 1) Curriculum A/B/C — the headline comparison at 3B scale (deeper task)
run curriculum --curriculum

# 2) Q1: residual thought loop, UNIFORM training (does residual alone avoid collapse?)
run q1_residual_uniform_B --residual --arm B
run q1_residual_uniform_C --residual --arm C

# 3) Q2: matched-SFT control — LoRA r=32 (~matches B's trainable), no latent, curriculum
run q2_matched_sft --curriculum --lora-r 32 --arm A

echo "ALL_3B_DONE $(date +%H:%M:%S)"
