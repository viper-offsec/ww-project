"""Contamination guards: detect foreign CPU/GPU activity around a measurement.

"Ours" = the experiment-runner process and all its descendants (engine
servers, load generator, samplers). Anything else that uses more than
GUARD_CPU_PCT % of a core, or any foreign GPU compute process, contaminates a run.

  python -m ww.guards monitor --root-pid 1234 --out guard.jsonl   # until SIGTERM
"""
import argparse
import json
import signal
import subprocess
import sys
import time

import psutil

from ww import settings as S

_running = True


def _stop(*_):
    global _running
    _running = False


def our_pids(root_pid: int) -> set:
    try:
        root = psutil.Process(root_pid)
        return {root_pid} | {c.pid for c in root.children(recursive=True)}
    except psutil.NoSuchProcess:
        return set()


def gpu_compute_pids() -> list:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return []
    res = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0].isdigit():
            res.append((int(parts[0]), parts[1] if len(parts) > 1 else ""))
    return res


class CpuTracker:
    def __init__(self):
        self.procs = {}

    def busy(self, exclude: set, threshold: float) -> list:
        busy = []
        seen = set()
        for p in psutil.process_iter(["pid", "name"]):
            pid = p.info["pid"]
            seen.add(pid)
            if pid not in self.procs:
                self.procs[pid] = p
                try:
                    p.cpu_percent(None)       # prime
                except psutil.Error:
                    pass
                continue
            if pid in exclude:
                continue
            try:
                pct = self.procs[pid].cpu_percent(None)
            except psutil.Error:
                continue
            if pct >= threshold:
                busy.append([pid, p.info["name"], round(pct, 1)])
        for pid in list(self.procs):
            if pid not in seen:
                del self.procs[pid]
        return busy


def check_once(root_pid: int, tracker: CpuTracker, threshold: float = S.GUARD_CPU_PCT) -> dict:
    ours = our_pids(root_pid)
    foreign_gpu = [[pid, name] for pid, name in gpu_compute_pids() if pid not in ours]
    return {"t_ns": time.time_ns(), "foreign_cpu": tracker.busy(ours, threshold),
            "foreign_gpu": foreign_gpu, "load1": round(psutil.getloadavg()[0], 2)}


def wait_until_clean(root_pid: int, max_wait: float = S.GUARD_MAX_WAIT_S) -> dict:
    """Block until no foreign activity is seen for 5 s (or max_wait elapses)."""
    tracker = CpuTracker()
    tracker.busy(set(), 1e9)   # prime counters
    t0 = time.time()
    last = None
    while time.time() - t0 < max_wait:
        time.sleep(5)
        last = check_once(root_pid, tracker)
        if not last["foreign_cpu"] and not last["foreign_gpu"]:
            return {"clean": True, "waited_s": round(time.time() - t0, 1), "last": last}
    return {"clean": False, "waited_s": round(time.time() - t0, 1), "last": last}


def monitor(root_pid: int, out: str, interval: float) -> int:
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    tracker = CpuTracker()
    tracker.busy(set(), 1e9)
    with open(out, "w") as f:
        while _running:
            time.sleep(interval)
            f.write(json.dumps(check_once(root_pid, tracker)) + "\n")
            f.flush()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("monitor")
    m.add_argument("--root-pid", type=int, required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--interval", type=float, default=5.0)
    c = sub.add_parser("check")
    c.add_argument("--root-pid", type=int, default=0)
    args = ap.parse_args(argv)
    if args.cmd == "monitor":
        return monitor(args.root_pid, args.out, args.interval)
    import os
    print(json.dumps(wait_until_clean(args.root_pid or os.getpid(), max_wait=15)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
