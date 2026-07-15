#!/usr/bin/env python3
"""Collector — walk a run directory, emit ONE valid JSON state blob to stdout.

Runs on the remote (stdlib only, works on py3.6+). Replaces the old bash-heredoc JSON
assembly (which broke whenever a log line contained a quote/brace). Everything here goes
through json.dumps, so the output is always valid JSON — the single biggest stability fix.

Usage (invoked by dashboard.sh over SSH):
    python3 collect_state.py <run_dir>

Output schema (all keys always present, so the HTML never has to guard for missing fields):
{
  "ts": "10:33:20", "run_dir": "...",
  "gpu": {"util": 85, "mem_used": 4298, "mem_total": 183359} | null,
  "stage": ["…last few driver.log markers…"],
  "active": [ {name, arm, step, total, pct, latest, progress:[[step,acc],…]} ],
  "configs": { "<name>": <that config's metrics.json, parsed> }
}
The HTML renderer is DATA-DRIVEN: it decides how to draw each config from the SHAPE of its
metrics (ref_acc/lit_acc -> bars, acc_by_hops -> lines, timescales -> fan). So changing arms,
params, adding configs, or switching mode needs NO dashboard edits — new data just appears.
"""
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


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def gpu_state():
    out = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
             "--format=csv,noheader,nounits").strip()
    if not out:
        return None
    line = out.splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    try:
        return {"util": int(parts[0]), "mem_used": int(parts[1]), "mem_total": int(parts[2])}
    except Exception:
        return None


# match "step  1200/4000 ... (ema_acc|ref_ema)=0.104"  (either y-metric)
STEP_RE = re.compile(r"step\s+(\d+)\s*/\s*(\d+).*?(?:ema_acc|ref_ema)\s*=\s*([0-9.]+)")
ARM_RE = re.compile(r"\[arm\s+([A-Za-z0-9]+)\]")


def parse_live(run_log):
    """Latest step/total, the newest arm label, latest log line, and the ema progress curve."""
    try:
        with open(run_log) as f:
            lines = f.readlines()
    except Exception:
        return None
    step = total = None
    arm = None
    latest = ""
    progress = []
    for ln in lines:
        a = ARM_RE.search(ln)
        if a:
            arm = a.group(1)
        m = STEP_RE.search(ln)
        if m:
            step, total = int(m.group(1)), int(m.group(2))
            progress.append([step, float(m.group(3))])
            latest = ln.strip()
    if step is None and arm is None:
        return None
    return {"arm": arm, "step": step, "total": total,
            "pct": round(100.0 * step / total, 1) if step and total else None,
            "latest": latest, "progress": progress}


def active_configs(run_dir):
    """Names of configs with a live training process (parallel-aware), via pgrep out-dir."""
    out = sh("pgrep -af 'kernel.run|agi_demo.run' | grep -v pgrep")
    names = []
    for m in re.finditer(r"out-dir\s+(\S+)", out):
        names.append(os.path.basename(m.group(1).rstrip("/")))
    # de-dup preserving order
    seen, uniq = set(), []
    for n in names:
        if n not in seen:
            seen.add(n); uniq.append(n)
    return uniq


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "agi_demo/outputs/kernel"
    state = {
        "ts": time.strftime("%H:%M:%S"),
        "run_dir": run_dir,
        "gpu": gpu_state(),
        "stage": [],
        "active": [],
        "configs": {},
    }

    driver_log = os.path.join(run_dir, "driver.log")
    if os.path.exists(driver_log):
        markers = [ln.strip() for ln in sh("grep -hE 'BATCH|START|DONE |ALL_.*DONE' "
                                           + "'%s'" % driver_log).splitlines()]
        state["stage"] = markers[-4:]

    # scheduler_state.json (written by scheduler/schedule.py) — surfaced in the dashboard if present
    sched = read_json(os.path.join(run_dir, "scheduler_state.json"))
    state["scheduler"] = sched

    actives = set(active_configs(run_dir))

    # every subdir is a config; attach metrics.json if present, live info if it's training
    if os.path.isdir(run_dir):
        for name in sorted(os.listdir(run_dir)):
            d = os.path.join(run_dir, name)
            if not os.path.isdir(d):
                continue
            m = read_json(os.path.join(d, "metrics.json"))
            if m is not None:
                state["configs"][name] = m
            if name in actives:
                live = parse_live(os.path.join(d, "run.log"))
                if live:
                    live["name"] = name
                    state["active"].append(live)

    sys.stdout.write(json.dumps(state))


if __name__ == "__main__":
    main()
