#!/usr/bin/env bash
# Live terminal dashboard for the remote kernel run (Project 4). Run locally:
#     bash agi_demo/kernel/watch_kernel.sh
# Auto-refreshes over SSH. Ctrl-C stops the watcher only (NOT the remote run).
#
# Shows: batch stage, finished-config arm results (ref_acc/lit_acc or arith OOD),
# the active config's latest training step, and live GPU utilization.

HOST="${1:-ea-main2}"
BASE="/media/cfs/xiezongyu.1/AGI/agi_demo/outputs/kernel"
INTERVAL="${2:-5}"

while true; do
  clear
  echo "=============================================================="
  echo " KERNEL run (Project 4) — live  (host: $HOST, refresh ${INTERVAL}s)"
  echo " $(date '+%H:%M:%S')   Ctrl-C = stop watcher (remote keeps running)"
  echo "=============================================================="
  ssh -o ConnectTimeout=15 "$HOST" bash -s <<REMOTE 2>/dev/null
cd /media/cfs/xiezongyu.1/AGI
B="$BASE"
echo ""
echo "STAGE:"; grep -hE "BATCH|START|DONE|ALL_KERNEL" "\$B/driver.log" 2>/dev/null | tail -3 | sed 's/^/  /'
echo ""
echo "FINISHED-ARM RESULTS:"
for d in "\$B"/*/; do
  n=\$(basename "\$d")
  grep -hE "arm K[012]\] (lit_acc|acc_by_hops)" "\$d/run.log" 2>/dev/null | sed "s#^#  [\$n] #"
done
echo ""
echo "CURRENTLY TRAINING (latest step of each active config):"
for active in \$(pgrep -af "agi_demo.kernel.run" | grep -v pgrep | grep -oE "out-dir [^ ]+" | awk '{print \$2}'); do
  echo "  -> \$(basename \$active)"
  grep -hE "arm K[012]\]| step |advance" "\$active/run.log" 2>/dev/null | tail -2 | sed 's/^/      /'
done
echo ""
echo "GPU:"; nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null | sed 's/^/  /'
REMOTE
  sleep "$INTERVAL"
done
