#!/usr/bin/env bash
# Stable live dashboard for any AGI run (kernel or 3B). Run locally:
#     bash agi_demo/dashboard/dashboard.sh [host] [run_dir] [interval_s]
# Defaults: ea-main2, the kernel run dir, 6s.
#
# Design for stability + zero-maintenance:
#   * The remote JSON is built by collect_state.py (stdlib python, json.dumps) — NOT a bash
#     heredoc — so a stray quote/brace in a log line can never corrupt it.
#   * The HTML is DATA-DRIVEN: it renders each config by the shape its metrics declare. Change
#     params / arms / mode / add configs → the dashboard just reflects it. No edits here ever.
#   * A failed SSH pull keeps the LAST good state and marks it stale (never a blank page).
#   * Ctrl-C stops the watcher only; the remote training run is untouched.
set -u
HOST="${1:-ea-main2}"
RUN_DIR="${2:-/media/cfs/xiezongyu.1/AGI/agi_demo/outputs/kernel}"
INTERVAL="${3:-6}"

HERE="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$HERE/dashboard.html"
COLLECTOR="$HERE/collect_state.py"
CHARTJS="$HERE/chart.umd.min.js"
OUT="/tmp/agi_dashboard"
REMOTE_COLLECTOR="/tmp/agi_collect_state.py"
mkdir -p "$OUT"

# The rendered HTML lives in $OUT and references chart.umd.min.js by RELATIVE path, so the vendored
# Chart.js must sit next to it. Copy it in (once, and refresh if the source changed).
if [ -f "$CHARTJS" ]; then
  cp -f "$CHARTJS" "$OUT/chart.umd.min.js"
else
  echo "warning: $CHARTJS not found — charts will show a load error until it exists"
fi

# Ship the collector to the remote once (and whenever it changes).
scp -q "$COLLECTOR" "$HOST:$REMOTE_COLLECTOR" 2>/dev/null \
  || { echo "warning: could not upload collector to $HOST (will retry in loop)"; }

echo "dashboard -> $OUT/dashboard.html   host=$HOST   run=$RUN_DIR   refresh=${INTERVAL}s"
echo "(Ctrl-C stops the watcher only; the remote run keeps going)"

first=1
while true; do
  # pull state; on failure keep the previous state.json and mark stale
  if raw=$(ssh -o ConnectTimeout=15 "$HOST" "cd /media/cfs/xiezongyu.1/AGI && python3 $REMOTE_COLLECTOR '$RUN_DIR'" 2>/dev/null) && [ -n "$raw" ]; then
    printf '%s' "$raw" > "$OUT/state.json"
    STALE=0
  else
    # remote collector may be missing (fresh notebook) — re-upload for the next tick
    scp -q "$COLLECTOR" "$HOST:$REMOTE_COLLECTOR" 2>/dev/null || true
    STALE=1
  fi

  # inline state.json into the template (python .replace on the sentinel line only -> always valid)
  python3 - "$TEMPLATE" "$OUT/state.json" "$OUT/dashboard.html" "$STALE" <<'PY'
import sys, json, re
tpl_p, state_p, out_p, stale = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
tpl = open(tpl_p).read()
try:
    raw = open(state_p).read().strip()
    st = json.loads(raw)
except Exception:
    st = {"ts":"(no data yet)","gpu":None,"stage":[],"active":[],"configs":{},"run_dir":""}
if stale == "1":
    st["__stale"] = True
inj = "var S = " + json.dumps(st) + "; //__STATE__"
html = re.sub(r"var S = .*//__STATE__", lambda _: inj, tpl, count=1)
open(out_p, "w").write(html)
PY

  if [ $first -eq 1 ]; then
    command -v open >/dev/null && open "$OUT/dashboard.html" 2>/dev/null
    first=0
  fi
  sleep "$INTERVAL"
done
