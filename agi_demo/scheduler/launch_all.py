#!/usr/bin/env python3
"""Launch every currently missing job once, without polling or automatic retries.

This launcher shares the scheduler's output-tree lock and exact procfs ownership check. Re-running
it while jobs are active skips those jobs instead of creating duplicate writers for an output dir.
"""
from __future__ import annotations

import argparse
import os

try:
    from .schedule import AlreadyLocked, acquire_run_lock, launch_job, live_processes_by_outdir, load_spec
except ImportError:  # Direct script execution: python agi_demo/scheduler/launch_all.py
    from schedule import AlreadyLocked, acquire_run_lock, launch_job, live_processes_by_outdir, load_spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--workdir", default="/media/cfs/xiezongyu.1/AGI")
    args = ap.parse_args()

    workdir = os.path.abspath(args.workdir)
    _, out_base, jobs = load_spec(args.spec, workdir)
    os.makedirs(out_base, exist_ok=True)
    try:
        lock = acquire_run_lock(os.path.join(out_base, ".scheduler.lock"))
    except AlreadyLocked as exc:
        print(f"refusing concurrent launch: {exc}")
        return 2

    launched, completed, active = [], [], []
    try:
        live = live_processes_by_outdir()
        for job in jobs:
            if os.path.exists(os.path.join(job["out_dir"], "metrics.json")):
                completed.append(job["name"])
                continue
            refs = live.get(job["out_dir"], [])
            if refs:
                active.append((job["name"], [ref.pid for ref in refs]))
                continue
            process = launch_job(job, workdir)
            launched.append((job["name"], process.pid))
    finally:
        lock.close()

    print(f"launched {len(launched)}: {launched}")
    if active:
        print(f"skipped (already active) {len(active)}: {active}")
    if completed:
        print(f"skipped (already complete) {len(completed)}: {completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
