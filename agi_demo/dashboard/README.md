# Training monitor (stable, data-driven)

One dashboard for every AGI run. Built once — it adapts to param/mode/arm changes on its own,
so you never edit it again.

## Daily use

```bash
bash agi_demo/dashboard/dashboard.sh                 # kernel run on ea-main2, opens in browser
bash agi_demo/dashboard/dashboard.sh ea-main2 /media/cfs/xiezongyu.1/AGI/agi_demo/outputs/kernel 6
#                                    ^host      ^remote run dir                                   ^refresh s
```

It opens `/tmp/agi_dashboard/dashboard.html`. **Two independent refresh layers:**
- **Driver pull** (the `refresh s` arg above): how often `dashboard.sh` re-pulls remote data and
  rewrites the HTML on disk. Default 6s.
- **In-browser auto-reload** (header control): how often the *tab* reloads to show the newest HTML.
  Pick from the dropdown (手动 / 3s / 5s / 10s / 30s / 60s) — a live countdown shows the next reload,
  the choice is remembered, and the **⟳** button reloads immediately. Default 5s; "手动" turns it off.

**Ctrl-C stops the watcher only — the remote training run is untouched.** Toggle light/dark with the
button (remembered).

## Why it never needs edits

- **Data-driven rendering.** The dashboard reads whatever `metrics.json` files exist under the run
  dir and draws each config by the *shape* its metrics declare (`dashboard.metric`):
  `ref_lit` → arm bars (+ an auto memory-timescale card if a probe is present), `acc_by_hops` →
  accuracy-vs-depth lines. Add a config, change `d_model`/`decays`/`session_hops`, add an arm, switch
  session↔arith → the dashboard just reflects it. No dashboard code changes, ever.
- **Self-describing metrics.** `kernel/run.py` writes a `dashboard` block (title/subtitle/kernel dims,
  derived from the run's own params) into each `metrics.json`, so chart titles are always current and
  correct — never a hardcoded name map.
- **Stable by construction.**
  - Remote JSON is built by `collect_state.py` (stdlib, `json.dumps`) — not a bash heredoc — so a
    stray quote/brace in a log line can't corrupt it (the old dashboard's main failure mode).
  - Chart.js is **vendored locally** (`chart.umd.min.js`) — no CDN, works with no internet.
  - A failed SSH pull keeps the last good view and marks it *stale* (never a blank page).

## Files
- `dashboard.sh` — the driver you run (SSH-pull → inline JSON → open/refresh HTML).
- `collect_state.py` — runs on the remote; emits one JSON blob (GPU, stage, active configs with live
  progress, every finished config's metrics.json).
- `dashboard.html` — the data-driven renderer (dataviz palette, validated; light + dark).
- `chart.umd.min.js` — vendored Chart.js (no network dependency).

## Colors (dataviz palette, validated)
Arms: **K0 = muted gray** (severed-read control, deliberately not a series hue), **K1 = blue**,
**K2 = orange** (worst-adjacent CVD ΔE ≈ 56, well clear). Timescale fan uses a single-hue blue ramp
ordered by γ (magnitude encoding). Chance is a red hairline; the Project-3 wall (0.10) a warning line.
