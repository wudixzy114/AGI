#!/usr/bin/env bash
# Live dashboard for the remote 3B run. Run locally:
#     bash agi_demo/watch_remote.sh
# Auto-refreshes every few seconds over SSH. Ctrl-C to stop (does NOT affect the remote run).
#
# Shows: which config/arm is training, latest step + loss + accuracy, curriculum level,
# finished-config eval tables, and live GPU utilization — all pulled from the remote logs.

HOST="${1:-ea-main2}"
BASE="/media/cfs/xiezongyu.1/AGI/agi_demo/outputs/3b_causal"
INTERVAL="${2:-5}"

while true; do
  clear
  echo "=============================================================="
  echo " 3B remote run — live  (host: $HOST, refresh ${INTERVAL}s, Ctrl-C to stop)"
  echo " $(date '+%H:%M:%S')"
  echo "=============================================================="
  ssh -o ConnectTimeout=15 "$HOST" bash -s <<REMOTE 2>/dev/null
cd /media/cfs/xiezongyu.1/AGI    # so relative out-dir paths from pgrep resolve
B="$BASE"
echo ""
echo "STAGE:"; grep -hE "START|DONE|ALL_" "\$B/master.log" 2>/dev/null | tail -2 | sed 's/^/  /'
echo ""
echo "FINISHED CONFIGS (eval accuracy by hop):"
for d in "\$B"/*/; do
  n=\$(basename "\$d")
  grep -hE "acc_by_hops|mean acc_by_hops" "\$d/run.log" 2>/dev/null | sed "s#^#  [\$n] #"
done
echo ""
echo "CURRENTLY TRAINING (latest step of the active config):"
active=\$(pgrep -af "agi_demo.run" | grep -v pgrep | grep -oE "out-dir [^ ]+" | awk '{print \$2}' | head -1)
if [ -n "\$active" ]; then
  echo "  -> \$active"
  grep -hE "arm [ABC]\]|step |advance" "\$active/run.log" 2>/dev/null | tail -4 | sed 's/^/    /'
else
  echo "  (no active training process — run may be finished)"
fi
echo ""
echo "GPU:"; nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null | sed 's/^/  /'
REMOTE
  sleep "$INTERVAL"
done
