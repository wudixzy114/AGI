#!/usr/bin/env python3
"""Resource-aware job scheduler for the B200 (or any single-GPU box).

WHY: kernel jobs use ~4 GB / 183 GB and are launch-bound (sequential scan), so hand-batched
`&`+`wait` barriers waste the GPU — fast jobs idle waiting for slow ones. This scheduler packs as
many jobs onto the GPU as MEASURED headroom allows, launching new ones the instant capacity frees.

DESIGN (robust by construction — survives its own restart, never double-launches):
  * STATELESS liveness. "running" = a process whose cmdline contains this job's --out-dir (pgrep);
    "done" = <out_dir>/metrics.json exists. So if the scheduler is killed and restarted, it simply
    re-derives the world from the filesystem + process table. No Popen handles to lose.
  * ADMISSION = memory headroom + concurrency cap + util ceiling. Admit job J iff
      running < max_concurrent
      AND est_mem(J) <= free_mem - reserved - safety_margin
      AND gpu_util < util_ceiling
    `reserved` = sum of est_mem for jobs launched within the last `warmup_s` seconds — they may not
    have allocated their memory yet, so we RESERVE it to avoid a launch stampede that OOMs the box.
  * CRASH-LOOP GUARD. A job that is launched but then neither running nor done within `grace_s` is
    retried up to `max_attempts`, then marked failed (won't relaunch forever).
  * Detached launch (`setsid`), so jobs outlive the scheduler; logs stream to <out_dir>/run.log.
  * Writes scheduler_state.json each tick for the dashboard.

Jobs are declared in a JSON spec (see jobs_kernel.json); nothing about the job matrix is hardcoded
here — change the spec, not this file.

Usage (on the remote, under nohup):
  python3 schedule.py jobs_kernel.json [--max-concurrent 8] [--mem-safety-mb 8000] \
      [--warmup-s 40] [--poll-s 10] [--util-ceiling 96]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode("utf-8", "replace")
    except Exception:
        return ""


def gpu_stat():
    """(free_mb, used_mb, total_mb, util_pct). Peak-of-3 util so a between-kernel trough
    doesn't read as idle. Returns None on failure (caller treats as 'no headroom info')."""
    best = None
    for _ in range(3):
        out = sh("nvidia-smi --query-gpu=memory.free,memory.used,memory.total,utilization.gpu "
                 "--format=csv,noheader,nounits").strip()
        if out:
            p = [int(x.strip()) for x in out.splitlines()[0].split(",")]
            if best is None or p[3] > best[3]:
                best = p
        time.sleep(0.3)
    if not best:
        return None
    return {"free": best[0], "used": best[1], "total": best[2], "util": best[3]}


def running_outdirs():
    """Set of --out-dir paths that currently have a live training process (the liveness source)."""
    out = sh("pgrep -af 'kernel.run|agi_demo.run' | grep -v pgrep | grep -v schedule.py")
    dirs = set()
    for m in re.finditer(r"--out-dir\s+(\S+)", out):
        dirs.add(os.path.normpath(m.group(1)))
    return dirs


def load_spec(path):
    with open(path) as f:
        spec = json.load(f)
    base = spec["base_cmd"]
    out_base = spec["out_base"]
    defaults = spec.get("defaults", {})
    jobs = []
    for j in spec["jobs"]:
        jobs.append({
            "name": j["name"],
            "args": j["args"],
            "est_mem_mb": j.get("est_mem_mb", defaults.get("est_mem_mb", 6000)),
            "priority": j.get("priority", 0),
            "out_dir": os.path.normpath(os.path.join(out_base, j["name"])),
        })
    return base, out_base, jobs


def kill_stale(out_dir):
    """Kill any process still bound to this out_dir before (re)launching. Prevents the new
    process from colliding with a survivor of a prior launch — that collision SIGKILLs one of
    them mid-write (leaves NUL bytes in run.log and orphaned CUDA contexts). Idempotent."""
    # match the exact '--out-dir <dir>' token; pkill -f matches the full cmdline
    subprocess.call(f"pkill -9 -f -- '--out-dir {out_dir}$' 2>/dev/null; "
                    f"pkill -9 -f -- '--out-dir {out_dir} ' 2>/dev/null", shell=True)


def launch(base_cmd, job, workdir):
    """Launch a job fully detached so it outlives the scheduler. We write the command to a small
    runner script and exec THAT (avoids fragile shell escaping through nested quoting), then
    setsid+background it. Logs stream to <out_dir>/run.log."""
    kill_stale(job["out_dir"])          # clear any survivor bound to this out_dir first
    os.makedirs(job["out_dir"], exist_ok=True)
    full = f"{base_cmd} {' '.join(job['args'])} --out-dir {job['out_dir']}"
    runner = os.path.join(job["out_dir"], ".run.sh")
    with open(runner, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {workdir}\n")
        f.write(f"exec {full} > {job['out_dir']}/run.log 2>&1\n")
    os.chmod(runner, 0o755)
    # setsid detaches into its own session; if setsid is unavailable, fall back to plain nohup
    setsid = "setsid" if _have("setsid") else "nohup"
    subprocess.Popen(f"{setsid} bash {runner} >/dev/null 2>&1 &", shell=True)


def _have(prog):
    return subprocess.call(f"command -v {prog} >/dev/null 2>&1", shell=True) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--workdir", default="/media/cfs/xiezongyu.1/AGI")
    ap.add_argument("--max-concurrent", type=int, default=8)
    ap.add_argument("--mem-safety-mb", type=int, default=8000)
    ap.add_argument("--warmup-s", type=int, default=40)
    ap.add_argument("--poll-s", type=int, default=10)
    ap.add_argument("--util-ceiling", type=int, default=96)
    ap.add_argument("--grace-s", type=int, default=90, help="a launched job must appear running within this")
    ap.add_argument("--absent-grace-s", type=int, default=300,
                    help="a job seen running may vanish from pgrep this long before we treat it as "
                         "crashed and relaunch. Set generously (default 5min): a node/tunnel stall "
                         "makes pgrep briefly return nothing, and a premature relaunch collides with "
                         "the still-alive original (mutual SIGKILL). Better to wait out the stall.")
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--state", default=None, help="state json path (default: <out_base>/scheduler_state.json)")
    args = ap.parse_args()

    base_cmd, out_base, jobs = load_spec(args.spec)
    state_path = args.state or os.path.join(out_base, "scheduler_state.json")
    os.makedirs(out_base, exist_ok=True)

    launched_at = {}      # name -> ts of last launch (reservation + grace tracking)
    attempts = {}         # name -> count
    seen_running = {}     # name -> True once pgrep has confirmed it live (closes the launch race)
    last_seen = {}        # name -> ts pgrep last saw it live (absent-grace tolerates flicker)
    print(f"[sched] {len(jobs)} jobs | max_conc={args.max_concurrent} "
          f"safety={args.mem_safety_mb}MB warmup={args.warmup_s}s poll={args.poll_s}s", flush=True)

    while True:
        now = time.time()
        g = gpu_stat()
        running = running_outdirs()

        def is_done(j):
            return os.path.exists(os.path.join(j["out_dir"], "metrics.json"))

        def is_running(j):
            return j["out_dir"] in running

        done = [j for j in jobs if is_done(j)]
        run_now = {j["name"] for j in jobs if is_running(j) and not is_done(j)}
        for nm in run_now:
            seen_running[nm] = True     # latch first time pgrep confirms it live
            last_seen[nm] = now         # refresh liveness timestamp every tick it's visible

        # Robust status model (fixes the re-admit bug): a job counts as OCCUPYING a slot if it is
        # either visible to pgrep now, OR launched-and-warming, OR was seen running and hasn't been
        # ABSENT longer than absent_grace_s. pgrep routinely misses a live proc for a tick or two
        # (fork/cmdline flicker, between-seed churn); the OLD code dropped such a job straight back
        # into `queued` and relaunched it, spawning duplicates that overwrote run.log and reset
        # progress. We only treat a job as crashed/finished-without-metrics after it has been gone
        # from pgrep for a sustained window.
        def occupies_slot(j):
            nm = j["name"]
            if nm in run_now:
                return True                                   # visibly running
            if nm not in launched_at:
                return False                                  # never launched
            if is_done(j):
                return False                                  # finished (has metrics.json)
            if now - launched_at[nm] <= args.warmup_s:
                return True                                   # just launched, still starting up
            if seen_running.get(nm) and now - last_seen.get(nm, 0) <= args.absent_grace_s:
                return True                                   # seen live recently; pgrep flicker
            return False                                      # gone long enough -> not occupying

        occupied = [j for j in jobs if occupies_slot(j)]
        run = occupied                                        # for slot accounting + state output
        run_names = {j["name"] for j in occupied}

        # reserved memory = est_mem of every occupying job still inside its warmup window (may not
        # have allocated its VRAM yet — reserve so we don't over-admit into a launch stampede)
        reserved = sum(j["est_mem_mb"] for j in occupied
                       if j["name"] in launched_at and now - launched_at[j["name"]] < args.warmup_s)

        # classify the rest: anything not done and not occupying a slot is either a retry or queued
        queued, failed = [], []
        for j in jobs:
            if is_done(j) or j["name"] in run_names:
                continue
            nm = j["name"]
            la = launched_at.get(nm)
            # launched, then disappeared from pgrep for > absent_grace (or never appeared within
            # grace) -> crashed / exited without metrics.json. Retry unless out of attempts.
            crashed = la is not None and (
                (seen_running.get(nm) and now - last_seen.get(nm, 0) > args.absent_grace_s)
                or (not seen_running.get(nm) and now - la > args.grace_s))
            if crashed and attempts.get(nm, 0) >= args.max_attempts:
                failed.append(j)
                continue
            queued.append(j)

        failed_names = {j["name"] for j in failed}

        # admission: highest priority first, pack by measured headroom
        admitted_this_tick = []
        if g is not None:
            eff_free = g["free"] - reserved - args.mem_safety_mb
            slots = args.max_concurrent - len(run)
            for j in sorted(queued, key=lambda x: -x["priority"]):
                if slots <= 0:
                    break
                if g["util"] >= args.util_ceiling:
                    break
                if j["est_mem_mb"] <= eff_free:
                    launch(base_cmd, j, args.workdir)
                    launched_at[j["name"]] = now
                    attempts[j["name"]] = attempts.get(j["name"], 0) + 1
                    eff_free -= j["est_mem_mb"]
                    slots -= 1
                    admitted_this_tick.append(j["name"])

        # persist state for the dashboard
        state = {
            "ts": time.strftime("%H:%M:%S"),
            "gpu": g,
            "reserved_mb": reserved,
            "policy": {"max_concurrent": args.max_concurrent, "mem_safety_mb": args.mem_safety_mb,
                       "warmup_s": args.warmup_s, "util_ceiling": args.util_ceiling},
            "counts": {"total": len(jobs), "done": len(done), "running": len(run),
                       "queued": len(queued), "failed": len(failed)},
            "running": sorted(run_names),
            "queued": [j["name"] for j in sorted(queued, key=lambda x: -x["priority"])],
            "done": [j["name"] for j in done],
            "failed": sorted(failed_names),
            "just_admitted": admitted_this_tick,
        }
        tmp = state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, state_path)   # atomic — dashboard never reads a half-written file

        if admitted_this_tick:
            print(f"[sched {state['ts']}] admitted {admitted_this_tick} | "
                  f"free={g['free'] if g else '?'}MB reserved={reserved}MB "
                  f"running={len(run)} queued={len(queued)}", flush=True)

        # exit when nothing is left to do
        if len(done) + len(failed) == len(jobs):
            print(f"[sched] all jobs settled: {len(done)} done, {len(failed)} failed "
                  f"({sorted(failed_names)})", flush=True)
            break
        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
