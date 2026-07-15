#!/usr/bin/env python3
"""Resource-aware, restart-safe scheduler for a single Linux GPU host.

The scheduler never kills a job. A retry is allowed only after the exact process previously
associated with an output directory has disappeared and no other live process owns that directory.
One advisory lock per output tree prevents two scheduler/launcher instances from admitting the
same job concurrently.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProcessRef:
    pid: int
    start_ticks: int
    pgid: int
    argv: tuple[str, ...]
    out_dir: str


class AlreadyLocked(RuntimeError):
    pass


def atomic_json_write(path: str, value: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def acquire_run_lock(path: str):
    """Hold this returned file object for the lifetime of the scheduler/launcher."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lock_file = open(path, "a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.seek(0)
        owner = lock_file.read().strip() or "unknown"
        lock_file.close()
        raise AlreadyLocked(f"another scheduler/launcher holds {path} (pid {owner})") from exc
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    return lock_file


def gpu_stat() -> dict[str, int] | None:
    """Return the busiest of three samples, or None when GPU telemetry is unavailable."""
    best = None
    command = [
        "nvidia-smi",
        "--query-gpu=memory.free,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    for _ in range(3):
        try:
            out = subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True).strip()
            values = [int(x.strip()) for x in out.splitlines()[0].split(",")]
            sample = {"free": values[0], "used": values[1], "total": values[2], "util": values[3]}
            if best is None or sample["util"] > best["util"]:
                best = sample
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
        time.sleep(0.3)
    return best


def _proc_start_ticks(pid: int) -> int:
    with open(f"/proc/{pid}/stat") as f:
        stat = f.read()
    # comm may contain spaces and parentheses; fields after the final ')' begin at field 3.
    return int(stat[stat.rfind(")") + 2 :].split()[19])


def _absolute_out_dir(value: str, pid: int) -> str:
    if os.path.isabs(value):
        return os.path.normpath(value)
    cwd = os.readlink(f"/proc/{pid}/cwd")
    return os.path.normpath(os.path.join(cwd, value))


def _out_dir_from_argv(argv: tuple[str, ...], pid: int) -> str | None:
    for index, token in enumerate(argv):
        if token == "--out-dir" and index + 1 < len(argv):
            return _absolute_out_dir(argv[index + 1], pid)
        if token.startswith("--out-dir="):
            return _absolute_out_dir(token.split("=", 1)[1], pid)
    return None


def live_processes_by_outdir(proc_root: str = "/proc") -> dict[str, list[ProcessRef]]:
    """Read argv directly from procfs; an empty result is never synthesized from command failure."""
    if not os.path.isdir(proc_root):
        raise RuntimeError(f"procfs is unavailable at {proc_root}")
    result: dict[str, list[ProcessRef]] = {}
    for entry in os.scandir(proc_root):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            with open(os.path.join(proc_root, entry.name, "cmdline"), "rb") as f:
                raw = f.read()
            argv = tuple(part.decode("utf-8", "surrogateescape") for part in raw.split(b"\0") if part)
            if not argv:
                continue
            out_dir = _out_dir_from_argv(argv, pid)
            if out_dir is None:
                continue
            ref = ProcessRef(
                pid=pid,
                start_ticks=_proc_start_ticks(pid),
                pgid=os.getpgid(pid),
                argv=argv,
                out_dir=out_dir,
            )
            result.setdefault(out_dir, []).append(ref)
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, ValueError):
            # Processes can exit while /proc is being scanned. A tracked live process is rediscovered
            # on the next tick; no retry happens merely because a scan entry was unreadable.
            continue
    return result


def recorded_process_alive(record: dict[str, Any]) -> bool:
    """Conservatively verify a saved PID generation when the full procfs scan missed it."""
    pid = record.get("pid")
    if not isinstance(pid, int):
        return False
    root_alive = True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        root_alive = False
    except PermissionError:
        return True
    expected_start = record.get("start_ticks")
    if root_alive and expected_start is None:
        return True
    if root_alive:
        try:
            if _proc_start_ticks(pid) == expected_start:
                return True
        except (PermissionError, OSError, ValueError):
            # An unreadable live identity is not evidence that it exited.
            return True

    # The session leader can exit before multiprocessing workers. Never relaunch into a surviving
    # process group created for this job.
    pgid = record.get("pgid")
    if isinstance(pgid, int) and pgid > 0:
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            pass
        except PermissionError:
            return True
    return False


def load_spec(path: str, workdir: str) -> tuple[list[str], str, list[dict[str, Any]]]:
    with open(path) as f:
        spec = json.load(f)
    base_command = shlex.split(spec["base_cmd"])
    if not base_command:
        raise ValueError("base_cmd must not be empty")
    out_base = spec["out_base"]
    out_base_abs = os.path.normpath(out_base if os.path.isabs(out_base) else os.path.join(workdir, out_base))
    defaults = spec.get("defaults", {})
    jobs = []
    names, out_dirs = set(), set()
    for item in spec["jobs"]:
        name = item["name"]
        out_dir = os.path.normpath(os.path.join(out_base_abs, name))
        if os.path.commonpath([out_base_abs, out_dir]) != out_base_abs:
            raise ValueError(f"job output escapes out_base: {name}")
        if name in names or out_dir in out_dirs:
            raise ValueError(f"duplicate job name or output directory: {name}")
        names.add(name)
        out_dirs.add(out_dir)
        command = base_command + [str(arg) for arg in item["args"]] + ["--out-dir", out_dir]
        jobs.append(
            {
                "name": name,
                "args": item["args"],
                "est_mem_mb": item.get("est_mem_mb", defaults.get("est_mem_mb", 6000)),
                "priority": item.get("priority", 0),
                "out_dir": out_dir,
                "command": command,
            }
        )
    return base_command, out_base_abs, jobs


def load_job_state(path: str) -> dict[str, dict[str, Any]]:
    try:
        with open(path) as f:
            state = json.load(f)
        records = state.get("job_state", {})
        return records if isinstance(records, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def reconcile_jobs(
    jobs: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    live: dict[str, list[ProcessRef]],
    max_attempts: int,
    now: float,
) -> dict[str, str]:
    """Reconcile durable ownership with procfs without ever terminating a process."""
    statuses: dict[str, str] = {}
    for job in jobs:
        name, out_dir = job["name"], job["out_dir"]
        refs = live.get(out_dir, [])
        record = records.setdefault(name, {"attempts": 0, "status": "queued"})
        metrics_exist = os.path.exists(os.path.join(out_dir, "metrics.json"))

        if len(refs) > 1:
            owners = [
                {
                    "pid": ref.pid,
                    "start_ticks": ref.start_ticks,
                    "pgid": ref.pgid if ref.pgid == ref.pid else None,
                }
                for ref in refs
            ]
            record.update(
                {"status": "conflict", "pids": sorted(ref.pid for ref in refs), "owners": owners}
            )
            statuses[name] = "conflict"
            continue
        if len(refs) == 1:
            ref = refs[0]
            record.update(
                {
                    "status": "finishing" if metrics_exist else "running",
                    "pid": ref.pid,
                    "start_ticks": ref.start_ticks,
                    "pgid": ref.pgid if ref.pgid == ref.pid else None,
                    "last_seen_at": now,
                }
            )
            record.pop("pids", None)
            record.pop("owners", None)
            statuses[name] = record["status"]
            continue
        if metrics_exist:
            record.update({"status": "done", "finished_at": now})
            statuses[name] = "done"
            continue

        previous = record.get("status")
        if previous == "conflict" and any(
            recorded_process_alive(owner) for owner in record.get("owners", [])
        ):
            record["status"] = "uncertain"
            statuses[name] = "uncertain"
            continue
        if previous in {"running", "finishing", "launching", "uncertain"} and recorded_process_alive(record):
            record["status"] = "uncertain"
            statuses[name] = "uncertain"
            continue
        if previous in {"running", "finishing", "launching", "conflict", "uncertain"}:
            record.update({"status": "exited", "exited_at": now})
        record.pop("pids", None)
        record.pop("owners", None)
        if record.get("attempts", 0) >= max_attempts:
            record["status"] = "failed"
            statuses[name] = "failed"
        else:
            record["status"] = "queued"
            statuses[name] = "queued"
    return statuses


def launch_job(job: dict[str, Any], workdir: str) -> subprocess.Popen:
    os.makedirs(job["out_dir"], exist_ok=True)
    log_path = os.path.join(job["out_dir"], "run.log")
    log = open(log_path, "ab", buffering=0)
    marker = f"\n[scheduler launch {time.strftime('%Y-%m-%d %H:%M:%S')}] {shlex.join(job['command'])}\n"
    log.write(marker.encode())
    try:
        process = subprocess.Popen(
            job["command"],
            cwd=workdir,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log.close()
    return process


def make_state(
    jobs: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    statuses: dict[str, str],
    gpu: dict[str, int] | None,
    reserved_mb: int,
    args: argparse.Namespace,
    admitted: list[str],
    error: str | None = None,
) -> dict[str, Any]:
    groups = {status: sorted(name for name, value in statuses.items() if value == status)
              for status in {"queued", "running", "finishing", "done", "failed", "conflict", "uncertain", "unknown"}}
    running = sorted(groups["running"] + groups["finishing"] + groups["uncertain"])
    return {
        "schema_version": 2,
        "ts": time.strftime("%H:%M:%S"),
        "gpu": gpu,
        "reserved_mb": reserved_mb,
        "policy": {
            "max_concurrent": args.max_concurrent,
            "mem_safety_mb": args.mem_safety_mb,
            "warmup_s": args.warmup_s,
            "util_ceiling": args.util_ceiling,
            "max_attempts": args.max_attempts,
        },
        "counts": {
            "total": len(jobs),
            "done": len(groups["done"]),
            "running": len(running),
            "queued": len(groups["queued"]),
            "failed": len(groups["failed"]),
            "conflict": len(groups["conflict"]),
            "unknown": len(groups["unknown"]),
        },
        "running": running,
        "queued": groups["queued"],
        "done": groups["done"],
        "failed": groups["failed"],
        "conflict": groups["conflict"],
        "uncertain": groups["uncertain"],
        "unknown": groups["unknown"],
        "just_admitted": admitted,
        "job_state": records,
        "error": error,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--workdir", default="/media/cfs/xiezongyu.1/AGI")
    ap.add_argument("--max-concurrent", type=int, default=8)
    ap.add_argument("--mem-safety-mb", type=int, default=8000)
    ap.add_argument("--warmup-s", type=int, default=40)
    ap.add_argument("--poll-s", type=float, default=10)
    ap.add_argument("--util-ceiling", type=int, default=96)
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--state", default=None)
    # Kept as no-op compatibility flags. Direct child PID ownership replaces time-based guessing.
    ap.add_argument("--grace-s", type=int, default=90, help=argparse.SUPPRESS)
    ap.add_argument("--absent-grace-s", type=int, default=300, help=argparse.SUPPRESS)
    args = ap.parse_args()

    workdir = os.path.abspath(args.workdir)
    _, out_base, jobs = load_spec(args.spec, workdir)
    state_path = os.path.abspath(args.state) if args.state else os.path.join(out_base, "scheduler_state.json")
    os.makedirs(out_base, exist_ok=True)
    try:
        lock = acquire_run_lock(os.path.join(out_base, ".scheduler.lock"))
    except AlreadyLocked as exc:
        print(f"[sched] refusing duplicate instance: {exc}", flush=True)
        return 2

    records = load_job_state(state_path)
    children: dict[int, subprocess.Popen] = {}
    print(f"[sched] {len(jobs)} jobs | max_conc={args.max_concurrent} | pid={os.getpid()}", flush=True)
    try:
        while True:
            # Reap children launched by this scheduler. Adopted jobs are reaped by their original
            # parent/init and are still tracked through procfs plus their persisted process group.
            for pid, child in list(children.items()):
                if child.poll() is not None:
                    children.pop(pid)
            now = time.time()
            gpu = gpu_stat()
            try:
                live = live_processes_by_outdir()
            except RuntimeError as exc:
                # Unknown liveness must stop admission. It must never be interpreted as zero jobs.
                statuses = {job["name"]: records.get(job["name"], {}).get("status", "unknown")
                            for job in jobs}
                state = make_state(jobs, records, statuses, gpu, 0, args, [], str(exc))
                atomic_json_write(state_path, state)
                print(f"[sched] liveness unavailable; admission paused: {exc}", flush=True)
                time.sleep(args.poll_s)
                continue

            statuses = reconcile_jobs(jobs, records, live, args.max_attempts, now)
            # Count actual processes, including duplicates and jobs finishing after metrics.json.
            active_count = sum(len(live.get(job["out_dir"], [])) for job in jobs)
            active_count += sum(
                1 for job in jobs
                if statuses[job["name"]] == "uncertain" and not live.get(job["out_dir"])
            )
            reserved = sum(
                job["est_mem_mb"]
                for job in jobs
                if statuses[job["name"]] in {"running", "finishing"}
                and now - records[job["name"]].get("launched_at", 0) < args.warmup_s
            )
            admitted = []
            if gpu is not None:
                effective_free = gpu["free"] - reserved - args.mem_safety_mb
                slots = max(0, args.max_concurrent - active_count)
                queued = sorted((job for job in jobs if statuses[job["name"]] == "queued"),
                                key=lambda job: -job["priority"])
                for job in queued:
                    if slots <= 0 or gpu["util"] >= args.util_ceiling:
                        break
                    if job["est_mem_mb"] > effective_free:
                        continue
                    try:
                        process = launch_job(job, workdir)
                    except OSError as exc:
                        record = records[job["name"]]
                        record["launch_error"] = str(exc)
                        print(f"[sched] failed to launch {job['name']}: {exc}", flush=True)
                        continue
                    record = records[job["name"]]
                    try:
                        start_ticks = _proc_start_ticks(process.pid)
                    except (FileNotFoundError, ProcessLookupError, OSError, ValueError):
                        start_ticks = None
                    record.update(
                        {
                            "attempts": record.get("attempts", 0) + 1,
                            "status": "launching",
                            "pid": process.pid,
                            "start_ticks": start_ticks,
                            "pgid": process.pid,
                            "launched_at": now,
                            "command": job["command"],
                        }
                    )
                    children[process.pid] = process
                    statuses[job["name"]] = "running"
                    effective_free -= job["est_mem_mb"]
                    slots -= 1
                    admitted.append(job["name"])

            state = make_state(jobs, records, statuses, gpu, reserved, args, admitted)
            atomic_json_write(state_path, state)
            if admitted:
                print(f"[sched {state['ts']}] admitted {admitted}", flush=True)
            settled = all(statuses[job["name"]] in {"done", "failed"} for job in jobs)
            if settled:
                print(f"[sched] all jobs settled: {state['counts']['done']} done, "
                      f"{state['counts']['failed']} failed", flush=True)
                return 0
            time.sleep(args.poll_s)
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
