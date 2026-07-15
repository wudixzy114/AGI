#!/usr/bin/env bash
# Submit a job matrix to the resource-aware scheduler on the B200 (or any GPU host).
#   bash agi_demo/scheduler/submit.sh [host] [jobs_spec] [max_concurrent]
# Defaults: ea-main2, jobs_kernel.json, 8 concurrent.
#
# The scheduler packs jobs onto the GPU by MEASURED free memory (not fixed &/wait batches), launches
# new ones the instant capacity frees, survives its own restart (liveness is derived from the process
# table + metrics.json, not in-memory handles), and reserves memory for warming-up jobs to avoid an
# OOM stampede. Watch progress with the dashboard (it shows the scheduler queue/running/done panel).
set -u
HOST="${1:-ea-main2}"
SPEC="${2:-jobs_kernel.json}"
MAXC="${3:-8}"
REMOTE_DIR="/media/cfs/xiezongyu.1/AGI"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "[submit] syncing kernel + scheduler to $HOST …"
rsync -az -e ssh --exclude='outputs/' --exclude='__pycache__/' --exclude='*.pyc' \
  "$HERE/../" "$HOST:$REMOTE_DIR/agi_demo/" >/dev/null

# derive out_base from the spec so we can report where results land
OUT_BASE=$(python3 -c "import json,sys; print(json.load(open('$HERE/$SPEC'))['out_base'])")

echo "[submit] launching scheduler on $HOST (spec=$SPEC, max_concurrent=$MAXC)"
ssh "$HOST" "cd $REMOTE_DIR && mkdir -p $OUT_BASE && \
  nohup /opt/conda/bin/python -u agi_demo/scheduler/schedule.py agi_demo/scheduler/$SPEC \
    --workdir $REMOTE_DIR --max-concurrent $MAXC \
    > $OUT_BASE/scheduler.log 2>&1 & echo \"scheduler pid=\$!\""

echo "[submit] done. Watch with:"
echo "    bash agi_demo/dashboard/dashboard.sh $HOST $REMOTE_DIR/$OUT_BASE"
echo "  or tail the scheduler log:"
echo "    ssh $HOST 'tail -f $REMOTE_DIR/$OUT_BASE/scheduler.log'"
