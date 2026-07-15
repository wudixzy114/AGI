#!/usr/bin/env python3
"""Capped launcher — run a job sweep with a concurrency limit gated on CPU RAM.

WHY (P5A lesson, the SECOND root cause): the real bottleneck on this notebook is CPU RAM (~20 GB
total), NOT GPU memory (183 GB). launch_all.py fires every job at once; 12 simultaneous cold-starts
(model init + CUDA context + dataloader) spiked past 20 GB and the kernel OOM-killer culled them
(no Python traceback — that's the OOM-kill signature). A single job is ~1.8 GB steady but peaks
higher at startup. See POSTMORTEM_p5a_scheduler.md.

This launcher is the minimal correct tool: keep at most --max-concurrent jobs running, and only
start a new one when free RAM > --min-free-mb. No pgrep-liveness, no auto-relaunch, no kill — it
only ever STARTS jobs (the fragile relaunch/kill machinery is what caused the earlier collisions).
It polls until every job either has metrics.json or is no longer catchable, then exits.

Idempotent: skips jobs that already have metrics.json, so re-running resumes a partial sweep.

Usage (detached on the remote):
  nohup python3 run_capped.py jobs_p5a.json --max-concurrent 5 --min-free-mb 6000 \
      --workdir /media/cfs/xiezongyu.1/AGI > capped.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode("utf-8", "replace")
    except Exception:
        return ""


def free_mb():
    """Available CPU RAM in MB (MemAvailable from /proc/meminfo — the honest 'can I allocate' number)."""
    out = sh("awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo")
    try:
        return int(out.strip())
    except Exception:
        return 0


def live_outdirs():
    """out_dirs with a live training process (best-effort; only used to count concurrency, never to kill)."""
    out = sh("pgrep -af 'kernel.run|agi_demo.run' | grep -v pgrep | grep -v run_capped")
    import re
    return {os.path.normpath(m.group(1)) for m in re.finditer(r"--out-dir\s+(\S+)", out)}


def launch(base_cmd, job, workdir):
    os.makedirs(job["out_dir"], exist_ok=True)
    full = f"{base_cmd} {' '.join(job['args'])} --out-dir {job['out_dir']}"
    runner = os.path.join(job["out_dir"], ".run.sh")
    with open(runner, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {workdir}\n")
        f.write(f"exec {full} > {job['out_dir']}/run.log 2>&1\n")
    os.chmod(runner, 0o755)
    setsid = "setsid" if sh("command -v setsid") else "nohup"
    subprocess.Popen(f"{setsid} bash {runner} >/dev/null 2>&1 &", shell=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--workdir", default="/media/cfs/xiezongyu.1/AGI")
    ap.add_argument("--max-concurrent", type=int, default=5,
                    help="max jobs running at once (CPU-RAM bound: ~1.8GB steady + startup peak/job)")
    ap.add_argument("--min-free-mb", type=int, default=6000,
                    help="don't start a new job unless at least this much CPU RAM is free (covers the "
                         "startup spike so a cold-start can't OOM the box)")
    ap.add_argument("--poll-s", type=int, default=20)
    ap.add_argument("--start-stagger-s", type=int, default=8,
                    help="min gap between two starts so their init spikes don't overlap")
    args = ap.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)
    base_cmd = spec["base_cmd"]
    out_base = spec["out_base"]
    jobs = [dict(j, out_dir=os.path.normpath(os.path.join(out_base, j["name"]))) for j in spec["jobs"]]
    os.makedirs(out_base, exist_ok=True)

    def done(j):
        return os.path.exists(os.path.join(j["out_dir"], "metrics.json"))

    print(f"[capped] {len(jobs)} jobs | max_concurrent={args.max_concurrent} "
          f"min_free={args.min_free_mb}MB stagger={args.start_stagger_s}s", flush=True)
    started = set()
    last_start = 0.0
    while True:
        live = live_outdirs()
        n_live = sum(1 for j in jobs if j["out_dir"] in live and not done(j))
        pending = [j for j in jobs if not done(j) and j["out_dir"] not in live and j["name"] not in started]
        n_done = sum(1 for j in jobs if done(j))

        if n_done + n_live == len(jobs) and not pending:
            print(f"[capped] settled: {n_done} done, {n_live} still running, none pending — exiting "
                  "(re-run to resume any that die).", flush=True)
            break

        now = time.time()
        fm = free_mb()
        # start one job per tick (staggered) if under the cap and RAM allows
        if (pending and n_live < args.max_concurrent and fm >= args.min_free_mb
                and now - last_start >= args.start_stagger_s):
            j = sorted(pending, key=lambda x: -x.get("priority", 0))[0]
            launch(base_cmd, j, args.workdir)
            started.add(j["name"]); last_start = now
            print(f"[capped {time.strftime('%H:%M:%S')}] start {j['name']} | "
                  f"live={n_live+1} done={n_done} free={fm}MB pending={len(pending)-1}", flush=True)
        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
