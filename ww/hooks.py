"""Experiment-runner lifecycle hooks (called from experiment/RunnerConfig.py).

Per run:  kill leftovers -> wait until machine is clean -> start engine ->
60 s warm-up (disjoint requests) -> start samplers -> measured phase (fixed
request set) -> stop samplers -> stop engine -> cool down -> compute metrics.

Hooks never raise: any failure is recorded in run_info.json and the run is
marked invalid (valid=0), so scripts/requeue_invalid.py can schedule it again.
"""
import json
import os
import shutil
import signal
import subprocess
import time
import traceback
from pathlib import Path

import psutil

from ww import engines as E
from ww import guards
from ww import runmetrics
from ww import settings as S

_STATE = {}


def _log(msg: str) -> None:
    print(f"[ww {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _factors(context) -> dict:
    r = context.execute_run
    return {k: str(r[k]) for k in ("model", "engine", "profile", "context")}


def _save_info(run_dir: Path) -> None:
    (run_dir / "run_info.json").write_text(json.dumps(_STATE.get("info", {}), indent=2, default=str))


def _err(run_dir: Path, what: str) -> None:
    _STATE.setdefault("info", {}).setdefault("errors", []).append(what)
    _STATE.setdefault("info", {}).setdefault("tracebacks", []).append(traceback.format_exc())
    _STATE["failed"] = True
    _log(f"ERROR {what}: {traceback.format_exc().splitlines()[-1]}")
    _save_info(run_dir)


def units_for(model: str, profile: str, context: str) -> int:
    """Number of units (sessions for Chat) in the measured phase."""
    key = f"{model}/{profile}_{context}"
    if S.N_PER_CELL_FILE.exists():
        table = json.loads(S.N_PER_CELL_FILE.read_text())
        if key in table:
            return int(table[key])
    default_requests = {"small": 160, "medium": 128, "large": 96}[context]
    return default_requests // (S.CHAT_TURNS if profile == "chat" else 1)


# --------------------------------------------------------------------- experiment level
def before_experiment(experiment_path: Path) -> None:
    os.environ["WW_ROOT_PID"] = str(os.getpid())
    os.environ["WW_EXPERIMENT_PATH"] = str(experiment_path)
    experiment_path.mkdir(parents=True, exist_ok=True)
    E.kill_leftovers()
    env = {"engines": E.engine_versions(), "settings": {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(S).items()
        if k.isupper() and not k.startswith("_")}}
    for cmd in (["nvidia-smi"], ["lscpu"], ["uname", "-a"], ["free", "-g"]):
        try:
            env[" ".join(cmd)] = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        except Exception as e:  # noqa: BLE001
            env[" ".join(cmd)] = str(e)
    (experiment_path / "environment.json").write_text(json.dumps(env, indent=2, default=str))
    idle = experiment_path / "idle_baseline.json"
    if not idle.exists():
        _log("recording 60 s idle baseline")
        csv_path = experiment_path / "idle_nvml.csv"
        p = subprocess.Popen([S.runner_python(), "-m", "ww.nvml_sampler", "--out", str(csv_path)],
                             cwd=S.REPO_ROOT)
        time.sleep(60)
        p.send_signal(signal.SIGTERM)
        p.wait(timeout=30)
        rows = csv_path.read_text().splitlines()[1:]
        cols = [r.split(",") for r in rows if r]
        t0, t1 = int(cols[5][0]), int(cols[-1][0])
        g = runmetrics.gpu_energy(csv_path, t0, t1)
        idle.write_text(json.dumps({"power_W": g["E_gpu_J"] / ((t1 - t0) / 1e9),
                                    "temp_C": g["gpu_temp_max_C"],
                                    "cpu_util_pct": g["cpu_util_mean_pct"]}, indent=2))
    _log(f"idle baseline: {idle.read_text().strip()}")


# --------------------------------------------------------------------- run level
def start_run(context) -> None:
    run_dir = Path(context.run_dir)
    for child in run_dir.iterdir():            # a re-run starts from an empty directory
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    f = _factors(context)
    _STATE.clear()
    _STATE["info"] = {"factors": f, "errors": [], "t_start_run": time.time()}
    _STATE["procs"] = {}
    _log(f"run {context.execute_run['__run_id']}: {f}")
    try:
        E.kill_leftovers()
        root = int(os.environ.get("WW_ROOT_PID", os.getppid()))
        pre = guards.wait_until_clean(root)
        _STATE["info"]["precheck"] = pre
        if not pre["clean"]:
            _log(f"WARNING machine not clean after {pre['waited_s']} s: {pre['last']}")
        proc = E.start_engine(f["engine"], f["model"], run_dir / "engine.log")
        _STATE["engine"] = proc
        _STATE["info"]["engine_start_s"] = round(E.wait_ready(f["engine"], proc), 2)
        _log(f"{f['engine']} ready in {_STATE['info']['engine_start_s']} s; warm-up {S.WARMUP_SECONDS} s")
        wl = S.workload_file(f["model"], f["profile"], f["context"], warmup=True)
        r = subprocess.run([S.runner_python(), "-m", "ww.loadgen", "--url", E.base_url(f["engine"]),
                            "--model", S.MODELS[f["model"]]["served_name"], "--workload", str(wl),
                            "--duration", str(S.WARMUP_SECONDS), "--concurrency", str(S.CONCURRENCY),
                            "--out-dir", str(run_dir / "warmup")],
                           cwd=S.REPO_ROOT, capture_output=True, text=True)
        _STATE["info"]["warmup_rc"] = r.returncode
        if r.returncode != 0:
            _STATE["info"]["errors"].append(f"warmup_rc_{r.returncode}")
            _log(f"warm-up problems: {r.stdout[-300:]} {r.stderr[-300:]}")
        time.sleep(5)
    except Exception:  # noqa: BLE001
        _err(run_dir, "start_run")
    _save_info(run_dir)


def start_measurement(context) -> None:
    if _STATE.get("failed"):
        return
    run_dir = Path(context.run_dir)
    f = _factors(context)
    try:
        py = S.runner_python()
        procs = _STATE["procs"]
        procs["nvml"] = subprocess.Popen([py, "-m", "ww.nvml_sampler", "--out", str(run_dir / "nvml.csv"),
                                          "--interval", str(S.SAMPLE_INTERVAL_S)], cwd=S.REPO_ROOT)
        procs["metrics"] = subprocess.Popen([py, "-m", "ww.metrics_scraper", "--url", E.base_url(f["engine"]),
                                             "--out", str(run_dir / "metrics.jsonl"),
                                             "--interval", str(S.METRICS_SCRAPE_INTERVAL_S)], cwd=S.REPO_ROOT)
        procs["guard"] = subprocess.Popen([py, "-m", "ww.guards", "monitor",
                                           "--root-pid", os.environ.get("WW_ROOT_PID", str(os.getppid())),
                                           "--out", str(run_dir / "guard.jsonl")], cwd=S.REPO_ROOT)
        if S.ENERGIBRIDGE_BIN.exists():
            procs["energibridge"] = subprocess.Popen(
                [str(S.ENERGIBRIDGE_BIN), "-g", "-i", "100", "-o", str(run_dir / "energibridge.csv"),
                 "--", "sleep", "86400"], stdout=open(run_dir / "energibridge.out", "w"),
                stderr=subprocess.STDOUT)
        time.sleep(S.PRE_ROLL_S)
    except Exception:  # noqa: BLE001
        _err(run_dir, "start_measurement")


def interact(context) -> None:
    if _STATE.get("failed"):
        return
    run_dir = Path(context.run_dir)
    f = _factors(context)
    try:
        units = units_for(f["model"], f["profile"], f["context"])
        _STATE["info"]["units"] = units
        _log(f"measured phase: {units} units at concurrency {S.CONCURRENCY}")
        r = subprocess.run([S.runner_python(), "-m", "ww.loadgen", "--url", E.base_url(f["engine"]),
                            "--model", S.MODELS[f["model"]]["served_name"],
                            "--workload", str(S.workload_file(f["model"], f["profile"], f["context"])),
                            "--units", str(units), "--concurrency", str(S.CONCURRENCY),
                            "--out-dir", str(run_dir / "measure")],
                           cwd=S.REPO_ROOT, capture_output=True, text=True)
        _STATE["info"]["measure_rc"] = r.returncode
        (run_dir / "measure_stdout.txt").write_text(r.stdout[-5000:] + "\n" + r.stderr[-5000:])
    except Exception:  # noqa: BLE001
        _err(run_dir, "interact")


def _terminate(p: subprocess.Popen, name: str) -> None:
    if p is None or p.poll() is not None:
        return
    if name == "energibridge":      # let EnergiBridge finish normally by ending its child
        try:
            for c in psutil.Process(p.pid).children():
                c.terminate()
        except psutil.Error:
            pass
    else:
        p.send_signal(signal.SIGTERM)
    try:
        p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        p.kill()


def stop_measurement(context) -> None:
    time.sleep(S.POST_ROLL_S)
    for name, p in list(_STATE.get("procs", {}).items()):
        try:
            _terminate(p, name)
        except Exception:  # noqa: BLE001
            pass


def _gpu_temp() -> float:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return float("nan")


def stop_run(context) -> None:
    run_dir = Path(context.run_dir)
    f = _factors(context)
    try:
        E.stop_engine(_STATE.get("engine"), f["engine"])
    except Exception:  # noqa: BLE001
        _err(run_dir, "stop_engine")
        E.kill_leftovers()
    base = None
    try:
        idle = Path(os.environ["WW_EXPERIMENT_PATH"]) / "idle_baseline.json"
        base = json.loads(idle.read_text()).get("temp_C")
    except Exception:  # noqa: BLE001
        pass
    t0 = time.time()
    while True:
        waited = time.time() - t0
        temp = _gpu_temp()
        if waited >= S.COOLDOWN_MAX_S:
            break
        if waited >= S.COOLDOWN_MIN_S and (base is None or temp <= base + S.COOLDOWN_TEMP_DELTA_C):
            break
        time.sleep(5)
    _STATE["info"]["cooldown_s"] = round(time.time() - t0, 1)
    _STATE["info"]["cooldown_end_temp_C"] = temp
    _save_info(run_dir)


def populate_run_data(context) -> dict:
    run_dir = Path(context.run_dir)
    f = _factors(context)
    try:
        res = runmetrics.compute(run_dir, f["profile"], f["context"])
    except Exception:  # noqa: BLE001
        _err(run_dir, "populate_run_data")
        res = {k: None for k in runmetrics.DATA_COLUMNS}
        res.update({"valid": 0, "invalid_reason": "metrics_error"})
    (run_dir / "run_metrics.json").write_text(json.dumps(res, indent=2))
    _log(f"result: valid={res.get('valid')} {res.get('invalid_reason')} "
         f"E_tok={res.get('E_tok_J')} J/tok window={res.get('window_s')} s")
    return {k: ("" if res.get(k) is None else res.get(k)) for k in runmetrics.DATA_COLUMNS}
