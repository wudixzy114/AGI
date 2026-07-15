# Scheduler Postmortem — Project 5 Phase A (2026-07-15)

A full-flow record of what went wrong with the resource-aware scheduler (`schedule.py`) while running
the P5A memory-overwrite sweep on the B200, every fix attempted, what each fix did and didn't solve,
and the final decision. Written for whoever next reaches for `schedule.py` — **read this before you
trust its liveness/relaunch logic.**

## TL;DR

The confirmed failure was **duplicate dispatch to the same output directory**, not training or
memory. The original code had several independent ways to produce it: no single-scheduler lock,
`pgrep` failures collapsed to an empty result, restart-critical ownership state kept only in memory,
and a fire-once launcher that only skipped completed (not currently running) jobs. The exact
historical trigger cannot be proven from the retained local artifacts, but the code races are
deterministic and sufficient to explain the symptom.

The earlier claim that NUL bytes proved "mutual SIGKILL" was wrong. A duplicate runner opened
`run.log` with `>` and truncated it while the original process retained its old file offset. Later
writes by the original create a sparse NUL-filled gap. That is strong evidence of duplicate writers,
not of a signal. The later `kill_stale()` workaround did explicitly send SIGKILL and therefore made
a mistaken retry destructive.

## Symptom timeline (what the user saw)

1. **"控制台没有任何任务, GPU 占用 0, 但后台任务在跑."** → NOT a scheduler bug. The local
   `dashboard.sh` was pointed at the old `kernel/` run dir, not `p5a/`; its `state.json` was also
   stale (11:22) because the ea-ssh tunnel token had expired. Fix: kill the stale driver, restart
   `dashboard.sh ea-main2 .../outputs/p5a 5`. (Lesson: the dashboard driver is per-run-dir; switching
   experiments requires restarting it with the right dir.)
2. **"第一部分跑完了但 Done 还是 0, 看不到数据."** → The real bug surfaced. metrics.json count = 0
   because no job ever finished — they were being **relaunched and reset** mid-run.
3. **"跑的任务又失败崩掉了."** → Same root cause, now visible as processes dying at low step counts
   with progress that had gone backwards.

## What was proven

**Isolation test that settled it:** ran one multi-seed job standalone, outside the scheduler:
```
python -m agi_demo.kernel.run --session --session-hops 0 --allow-overwrite --arm K2 \
  --session-len 12 --n-slots 4 --steps 150 --lr 3e-3 --seeds 0,1,2 --out-dir /tmp/iso_test
→ METRICS WRITTEN, ref_acc 0.997, seeds [0,1,2] all completed.
```
So **run.py's multi-seed loop is fine.** The failure only happened under orchestration.

Two processes were launched against the same `--out-dir`. The NUL gap in `run.log` is explained by
the second runner truncating the file while the first continued writing at its previous offset.
Both processes then competed for GPU resources and output files. No retained artifact identifies
which duplicate-dispatch path fired first.

## Deterministic defects found in the code audit

1. `submit.sh` could start multiple scheduler daemons for the same `out_base`; there was no lock.
   Both instances could scan the same queue and admit the same job in the same polling interval.
2. `launched_at`, `attempts`, and ownership lived only in RAM despite the "stateless restart" claim.
   After restart, a missed scan classified an existing job as never launched and immediately queued it.
3. `sh()` converted every process-scan exception into `""`, making "liveness unavailable"
   indistinguishable from "zero live jobs".
4. `launch_all.py` skipped only jobs with `metrics.json`. Re-running it while jobs were still active
   launched duplicates for every unfinished output directory.
5. `kill_stale()` used a regex-based `pkill -9` before retry. A false retry therefore killed the
   healthy generation it had failed to identify.

## Fixes attempted, in order

### Fix 1 — `seen_running` latch (close the launch race)
**Change:** once pgrep confirms a job live, latch `seen_running[name]=True` so a subsequent pgrep miss
doesn't immediately re-queue it.
**What it solved:** the *initial* double-admit at launch (pgrep lagging a just-forked process).
**What it did NOT solve:** it *created* a new gap — a job seen-then-briefly-missed was excluded from
`inflight` (guarded by `not seen_running`) yet not in `run`, so it fell straight into `queued` and was
relaunched anyway. Net: still re-admitted healthy jobs. (Unit-tested the latch in isolation — passed —
but the interaction with the classification branches was the hole.)

### Fix 2 — `occupies_slot()` state model + `absent_grace_s`
**Change:** replaced the scattered run/inflight/queued branches with one function: a job occupies a
slot if visibly running OR within launch warmup OR (seen_running AND last-seen within
`absent_grace_s`). Added `last_seen[name]` refreshed every visible tick. Initial `absent_grace_s=45`.
**What it solved:** short (≤45 s) pgrep flicker no longer re-queues a job. Unit-tested: a job stays
occupied through a simulated flicker and is only freed after sustained absence; genuine crashes still
free the slot and retry.
**What it did NOT solve:** a time threshold cannot establish process ownership, and it did nothing
about concurrent scheduler instances or state loss after restart.

### Fix 3 — `kill_stale()` before launch + `absent_grace_s=300`
**Change:** (a) before every (re)launch, `pkill -9 -f -- '--out-dir <dir>$'` to clear any survivor
bound to that out_dir, so a relaunch can't collide with an original. (b) Raised grace to 5 min so a
multi-minute stall waits the process out instead of relaunching.
**What it solved:** a retry no longer left two same-directory processes alive.
**What it did NOT solve:** it achieved that by killing the prior generation, even when the retry
decision was wrong. `pkill -f` also treated paths as regex and could overmatch. This changed a
duplicate-write bug into an explicit mid-task termination path.

## Final decision — delete the complexity for this workload

Measured fact: single job ≈ 1.6–2 GB; **14 jobs × ~2 GB ≈ 28 GB vs 183 GB available.** Everything
fits at once. The scheduler's reason-to-exist (pack jobs when they can't all fit) does not apply, so
all of its fragile machinery is pure downside here.

**`agi_demo/scheduler/launch_all.py`** — fire every job once, detached, log to
`<out_dir>/run.log`, then **exit**. The original version only skipped `metrics.json`; that was not
actually idempotent while a job was still running. The repaired version also takes the output-tree
dispatcher lock and skips output directories found in the live process table.

## Guidance for next time

- **Default to `launch_all.py`** whenever the whole sweep fits in VRAM at once (check:
  `Σ est_mem_mb ≪ GPU total`). It is the robust choice precisely because it has no moving parts.
- **Reserve `schedule.py`** for genuinely memory-constrained sweeps (jobs that cannot all co-reside).
  It now uses exact procfs argv/PID identity, persistent attempt state, and a single-dispatcher lock;
  it never kills an existing process before retrying.
- **The dashboard driver is per-run-dir.** Switching experiments (kernel → p5a) requires restarting
  `dashboard.sh` with the new dir, or you'll watch a stale/empty directory. A "latest active dir"
  auto-pick would remove this footgun.
- **`pkill -f` with paths is dangerous** — path separators and dots are regex, and prefix out-dir
  names (`slots2` vs `slots12`) risk over-match. Anchor carefully or match on an exact, quoted token.
- **NUL gaps in a concurrently written/truncated log are sparse-file holes**, not proof of SIGKILL.
  Check for a second process or a second `>` opener.
- **The scheduler runs on the remote**, so a local tunnel flap cannot directly make its `pgrep`
  return empty. A local observation gap and a remote liveness failure are different events.

## Hardening implemented after this incident
- **Exact generation guard:** each launch persists PID plus `/proc/<pid>/stat` start ticks, and a
  restart adopts any exact `--out-dir` owner before it considers admission.
- **Single dispatcher:** scheduler and fire-once launcher share a lock scoped to `out_base`.
- **No automatic killing:** duplicate owners become a visible `conflict`; the scheduler pauses that
  job instead of trying to choose and kill one.
- **Dashboard auto-selects the newest active run dir** so switching experiments needs no restart.
