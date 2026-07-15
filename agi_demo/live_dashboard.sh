#!/usr/bin/env bash
# Live BROWSER dashboard for the remote run. Run locally:
#     bash agi_demo/live_dashboard.sh
# Every few seconds: SSH-pulls remote logs into state.json, then inlines that JSON into a copy of
# live_template.html (data baked in -> no fetch -> no file:// CORS issue). Auto-refreshes in the
# browser. Ctrl-C to stop (the remote run is unaffected).

HOST="${1:-ea-main2}"
REMOTE_BASE="${3:-/media/cfs/xiezongyu.1/AGI/agi_demo/outputs/3b_causal}"
OUT="/tmp/agi_live"
INTERVAL="${2:-6}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$HERE/live_template.html"
mkdir -p "$OUT"

echo "live dashboard -> $OUT/live.html  (refreshing every ${INTERVAL}s; Ctrl-C to stop)"
first=1
while true; do
  ssh -o ConnectTimeout=15 "$HOST" bash -s <<REMOTE 2>/dev/null > "$OUT/state.json"
cd /media/cfs/xiezongyu.1/AGI    # so relative out-dir paths from pgrep resolve
B="$REMOTE_BASE"
echo '{'
echo '"ts":"'\$(date +%H:%M:%S)'",'
# GPU: sample util 3x (~1.5s) and report MAX so a between-kernel 0% trough or a CPU-only
# probe phase doesn't misleadingly read as "idle". Memory shown from the same probe.
gpu=\$(for i in 1 2 3; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1; sleep 0.5; done | sort -t, -k1 -n | tail -1 | awk -F, '{printf "%s%% util, %d MiB (peak of 3 samples)", \$1, \$2}')
echo '"gpu":"'\$gpu'",'
# ALL currently-training configs (parallel-aware), not just the first
actives=\$(pgrep -af "agi_demo.run" | grep -v pgrep | grep -oE "out-dir [^ ]+" | awk '{print \$2}' | sort -u)
echo '"active":['
first=1
for a in \$actives; do [ \$first -eq 0 ] && echo ','; first=0; echo '"'\$(basename "\$a")'"'; done
echo '],'
# per-active live info: latest step lines + ema progress curve
echo '"live":{'
firstl=1
for a in \$actives; do
  [ \$firstl -eq 0 ] && echo ','; firstl=0
  n=\$(basename "\$a")
  echo "\"\$n\":{"
  echo '"steps":['
  grep -hE "step |advance" "\$a/run.log" 2>/dev/null | tail -4 | \
    sed 's/"/ /g' | awk '{printf "%s\"%s\"", (NR>1?",":""), \$0} END{print ""}'
  echo '],"progress":['
  # accept either ema_acc= (chain tasks) or ref_ema= (session task) as the y-value
  grep -hE "step +[0-9]+/" "\$a/run.log" 2>/dev/null | \
    sed -E 's/.*step +([0-9]+)\/.*(ema_acc|ref_ema)=([0-9.]+).*/[\1,\3]/' | grep '^\[' | \
    awk '{printf "%s%s",(NR>1?",":""),\$0} END{print ""}'
  echo ']}'
done
echo '},'
echo '"configs":{'
firstc=1
for d in "\$B"/*/; do
  n=\$(basename "\$d")
  [ -f "\$d/metrics.json" ] || continue
  [ \$firstc -eq 0 ] && echo ','
  firstc=0
  echo "\"\$n\":"
  cat "\$d/metrics.json"
done
echo '}}'
REMOTE

  # Inline the pulled JSON into the template. Python .replace touches only the token,
  # leaving all JS (no backticks / no ${}) intact -> valid HTML every time.
  python3 - "$TEMPLATE" "$OUT/state.json" "$OUT/live.html" <<'PY'
import sys, json, re
tpl_path, state_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
tpl = open(tpl_path).read()
raw = open(state_path).read().strip()
try:
    json.loads(raw)          # validate; if the pull was mid-write, fall back to empty
except Exception:
    raw = '{"ts":"(pull error)","gpu":"","active":"none","steps":[],"progress":[],"configs":{}}'
# Replace the entire seeded line (var S = {...}; //__STATE__) with the real inlined data.
html = re.sub(r"var S = .*//__STATE__", "var S = " + raw + ";", tpl, count=1)
open(out_path, "w").write(html)
PY

  if [ $first -eq 1 ]; then
    command -v open >/dev/null && open "$OUT/live.html"
    first=0
  fi
  sleep "$INTERVAL"
done
