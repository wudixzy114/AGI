import json
import os
import tempfile
import unittest
from unittest import mock

try:
    from . import launch_all, schedule
except ImportError:
    import launch_all
    import schedule


class SchedulerStateTests(unittest.TestCase):
    def job(self, root, name="job"):
        return {
            "name": name,
            "out_dir": os.path.join(root, name),
            "est_mem_mb": 100,
            "priority": 1,
            "command": ["python", "train.py", "--out-dir", os.path.join(root, name)],
        }

    def ref(self, job, pid=123, start_ticks=456):
        return schedule.ProcessRef(pid, start_ticks, pid, tuple(job["command"]), job["out_dir"])

    def test_restart_adopts_an_existing_process(self):
        with tempfile.TemporaryDirectory() as root:
            job = self.job(root)
            records = {}
            statuses = schedule.reconcile_jobs(
                [job], records, {job["out_dir"]: [self.ref(job)]}, max_attempts=2, now=10
            )
            self.assertEqual(statuses, {"job": "running"})
            self.assertEqual(records["job"]["pid"], 123)
            self.assertEqual(records["job"]["start_ticks"], 456)
            self.assertEqual(records["job"]["pgid"], 123)

    def test_missing_owned_process_retries_without_killing_anything(self):
        with tempfile.TemporaryDirectory() as root:
            job = self.job(root)
            records = {"job": {"status": "running", "attempts": 1, "pid": 123}}
            statuses = schedule.reconcile_jobs([job], records, {}, max_attempts=2, now=20)
            self.assertEqual(statuses, {"job": "queued"})
            self.assertEqual(records["job"]["attempts"], 1)
            self.assertEqual(records["job"]["exited_at"], 20)

    def test_scan_miss_does_not_requeue_a_still_live_saved_pid(self):
        with tempfile.TemporaryDirectory() as root:
            job = self.job(root)
            records = {"job": {"status": "running", "attempts": 1, "pid": 123, "start_ticks": 456}}
            with mock.patch.object(schedule, "recorded_process_alive", return_value=True):
                statuses = schedule.reconcile_jobs([job], records, {}, max_attempts=2, now=25)
            self.assertEqual(statuses, {"job": "uncertain"})
            self.assertEqual(records["job"]["attempts"], 1)

    def test_attempt_limit_survives_restart(self):
        with tempfile.TemporaryDirectory() as root:
            state_path = os.path.join(root, "state.json")
            schedule.atomic_json_write(
                state_path, {"job_state": {"job": {"status": "running", "attempts": 2}}}
            )
            records = schedule.load_job_state(state_path)
            statuses = schedule.reconcile_jobs([self.job(root)], records, {}, max_attempts=2, now=30)
            self.assertEqual(statuses, {"job": "failed"})

    def test_duplicate_processes_are_reported_and_never_requeued(self):
        with tempfile.TemporaryDirectory() as root:
            job = self.job(root)
            refs = [self.ref(job, 123), self.ref(job, 124)]
            records = {}
            statuses = schedule.reconcile_jobs(
                [job], records, {job["out_dir"]: refs}, max_attempts=2, now=40
            )
            self.assertEqual(statuses, {"job": "conflict"})
            self.assertEqual(records["job"]["pids"], [123, 124])

    def test_scan_miss_after_conflict_does_not_launch_a_third_copy(self):
        with tempfile.TemporaryDirectory() as root:
            job = self.job(root)
            refs = [self.ref(job, 123), self.ref(job, 124)]
            records = {}
            schedule.reconcile_jobs([job], records, {job["out_dir"]: refs}, 2, 40)
            with mock.patch.object(schedule, "recorded_process_alive", return_value=True):
                statuses = schedule.reconcile_jobs([job], records, {}, 2, 50)
            self.assertEqual(statuses, {"job": "uncertain"})

    def test_metrics_are_done_only_after_the_process_exits(self):
        with tempfile.TemporaryDirectory() as root:
            job = self.job(root)
            os.makedirs(job["out_dir"])
            with open(os.path.join(job["out_dir"], "metrics.json"), "w") as f:
                json.dump({}, f)
            records = {}
            live = {job["out_dir"]: [self.ref(job)]}
            self.assertEqual(
                schedule.reconcile_jobs([job], records, live, 2, 50), {"job": "finishing"}
            )
            self.assertEqual(schedule.reconcile_jobs([job], records, {}, 2, 60), {"job": "done"})


class SchedulerLockTests(unittest.TestCase):
    def test_second_dispatcher_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, ".scheduler.lock")
            first = schedule.acquire_run_lock(path)
            try:
                with self.assertRaises(schedule.AlreadyLocked):
                    schedule.acquire_run_lock(path)
            finally:
                first.close()


class LaunchAllTests(unittest.TestCase):
    def test_active_output_directory_is_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            job = SchedulerStateTests().job(root)
            ref = SchedulerStateTests().ref(job)
            lock = mock.Mock()
            argv = ["launch_all.py", "jobs.json", "--workdir", root]
            with (
                mock.patch("sys.argv", argv),
                mock.patch.object(launch_all, "load_spec", return_value=(["python"], root, [job])),
                mock.patch.object(launch_all, "acquire_run_lock", return_value=lock),
                mock.patch.object(
                    launch_all, "live_processes_by_outdir", return_value={job["out_dir"]: [ref]}
                ),
                mock.patch.object(launch_all, "launch_job") as launch,
            ):
                self.assertEqual(launch_all.main(), 0)
            launch.assert_not_called()
            lock.close.assert_called_once()


class SpecTests(unittest.TestCase):
    def test_commands_use_absolute_output_directories_and_argv_tokens(self):
        with tempfile.TemporaryDirectory() as root:
            spec_path = os.path.join(root, "jobs.json")
            with open(spec_path, "w") as f:
                json.dump(
                    {
                        "base_cmd": "python -u -m package.run",
                        "out_base": "outputs",
                        "jobs": [{"name": "a", "args": ["--value", "with space"]}],
                    },
                    f,
                )
            _, out_base, jobs = schedule.load_spec(spec_path, root)
            self.assertEqual(out_base, os.path.join(root, "outputs"))
            self.assertEqual(jobs[0]["command"][-2:], ["--out-dir", os.path.join(root, "outputs/a")])
            self.assertIn("with space", jobs[0]["command"])

    def test_job_output_cannot_escape_the_locked_output_tree(self):
        with tempfile.TemporaryDirectory() as root:
            spec_path = os.path.join(root, "jobs.json")
            with open(spec_path, "w") as f:
                json.dump(
                    {"base_cmd": "python train.py", "out_base": "outputs",
                     "jobs": [{"name": "../outside", "args": []}]},
                    f,
                )
            with self.assertRaisesRegex(ValueError, "escapes out_base"):
                schedule.load_spec(spec_path, root)


if __name__ == "__main__":
    unittest.main()
