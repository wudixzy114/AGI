# Resource-aware job scheduler

Submit a whole job matrix once; the scheduler **packs jobs onto the GPU by measured free memory**
and launches new ones the instant capacity frees — no more hand-batched `&`/`wait` (which idled the
GPU while fast jobs waited for slow ones). This replaces `agi_demo/kernel/run_kernel.sh`.

## Use

```bash
bash agi_demo/scheduler/submit.sh                       # ea-main2, jobs_kernel.json, 8 concurrent
bash agi_demo/scheduler/submit.sh ea-main2 jobs_kernel.json 10
bash agi_demo/dashboard/dashboard.sh ea-main2 /media/cfs/xiezongyu.1/AGI/agi_demo/outputs/kernel_sched
```

The dashboard shows a **Scheduler** panel (done/running/queued/failed, free memory, what's next).

## Change the sweep = edit one JSON

`jobs_kernel.json` is the whole matrix. Each job:
```json
{"name":"session_hops0","priority":10,"est_mem_mb":5000,"args":["--session","--session-hops","0", …]}
```
- `name` → its output subdir (`out_base/name`) and its liveness key.
- `priority` → queue order (higher first).
- `est_mem_mb` → conservative VRAM estimate used for admission.
- `args` → appended to `base_cmd`; `--out-dir` is added automatically.

Add/remove jobs, retune `est_mem_mb`, reorder by `priority` — nothing else changes. The dashboard
auto-adapts too (it renders by metric shape).

## Admission policy (why it won't OOM or thrash)

Admit job J iff **all** hold:
- `running < max_concurrent` (a hard pool cap),
- `est_mem(J) ≤ free_mem − reserved − mem_safety_mb`,
- `gpu_util < util_ceiling`.

`reserved` = Σ `est_mem` of jobs launched within the last `warmup_s` seconds — they may not have
allocated their memory yet, so the scheduler **reserves** it to prevent a launch stampede that would
OOM the box. Tunable via flags (`--max-concurrent`, `--mem-safety-mb`, `--warmup-s`, `--util-ceiling`).

## Robust by construction

- **Stateless liveness** — "running" is derived from the process table (a process whose cmdline
  carries the job's `--out-dir`), "done" from `metrics.json` existing. So the scheduler can be killed
  and restarted and it re-derives the world from disk + `ps` — it never double-launches a job.
- **Crash-loop guard** — a job launched but neither running nor done within `grace_s` is retried up
  to `max_attempts`, then marked `failed` (won't relaunch forever).
- **Detached launch** — each job runs under `setsid` (falls back to `nohup`) via a small generated
  `.run.sh`, so jobs outlive the scheduler; logs stream to `<out_dir>/run.log`.
- **Atomic state** — `scheduler_state.json` is written via a temp file + `os.replace`, so the
  dashboard never reads a half-written file.

## Files
- `schedule.py` — the daemon (stdlib only; runs on the remote under nohup).
- `jobs_kernel.json` — the declarative job matrix (edit this to change the sweep).
- `submit.sh` — rsync + launch the scheduler on the host.
