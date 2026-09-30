"""Pilot: calibrate the measured-phase size N of every cell (Phase 1 in the
reference study). For each engine and cell: 20 s warm-up, then a short run of
N0 requests; throughput of the *faster* engine defines N so that its measured
window lasts >= MIN_WINDOW_SECONDS (x1.1 safety), rounded up to a multiple of
the concurrency. Writes workloads/n_per_cell.json and logs/pilot_report.json.

  python scripts/pilot.py [--models llama,qwen] [--n0 32]
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ww import engines as E  # noqa: E402
from ww import settings as S  # noqa: E402
from ww.runmetrics import engine_metrics  # noqa: E402


def cells_for(model: str):
    if model == "qwen":
        return [(p, c) for p in S.QWEN_CELLS["profiles"] for c in S.QWEN_CELLS["contexts"]]
    return [(p, c) for p in S.PROFILES for c in S.CONTEXTS]


def loadgen(engine, model, wl, out, units=0, duration=0):
    cmd = [S.runner_python(), "-m", "ww.loadgen", "--url", E.base_url(engine),
           "--model", S.MODELS[model]["served_name"], "--workload", str(wl),
           "--concurrency", str(S.CONCURRENCY), "--out-dir", str(out)]
    cmd += ["--duration", str(duration)] if duration else ["--units", str(units)]
    r = subprocess.run(cmd, cwd=S.REPO_ROOT, capture_output=True, text=True)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"error": (r.stdout + r.stderr)[-500:]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama,qwen")
    ap.add_argument("--engines", default=",".join(S.ENGINES))
    ap.add_argument("--n0", type=int, default=32, help="requests per pilot cell")
    args = ap.parse_args(argv)
    root = S.LOGS_DIR / f"pilot_{time.strftime('%Y%m%d_%H%M%S')}"
    root.mkdir(parents=True, exist_ok=True)
    report = {}
    for model in args.models.split(","):
        for engine in args.engines.split(","):
            E.kill_leftovers()
            proc = E.start_engine(engine, model, root / f"engine_{engine}_{model}.log")
            try:
                start_s = E.wait_ready(engine, proc)
                print(f"{engine}/{model} ready in {start_s:.0f} s", flush=True)
                for profile, context in cells_for(model):
                    key = f"{model}/{profile}_{context}"
                    d = root / engine / f"{model}_{profile}_{context}"
                    loadgen(engine, model, S.workload_file(model, profile, context, warmup=True), d / "warmup", duration=20)
                    scrape = subprocess.Popen([S.runner_python(), "-m", "ww.metrics_scraper", "--url",
                                               E.base_url(engine), "--out", str(d / "metrics.jsonl")], cwd=S.REPO_ROOT)
                    units = max(1, args.n0 // (S.CHAT_TURNS if profile == "chat" else 1))
                    s = loadgen(engine, model, S.workload_file(model, profile, context), d / "measure", units=units)
                    scrape.terminate()
                    scrape.wait(timeout=20)
                    em = engine_metrics(d / "metrics.jsonl", s.get("t_start_ns", 0), s.get("t_end_ns", 1 << 62))
                    entry = report.setdefault(key, {})
                    entry[engine] = {"req_per_s": s.get("req_per_s"), "window_s": s.get("window_s"),
                                     "errors": s.get("n_errors"), "ttft_ms_median": s.get("ttft_ms_median"),
                                     "kv_peak_pct": em["kv_peak_pct"], "preemptions": em["preemptions"],
                                     "engine_start_s": round(start_s, 1)}
                    print(key, engine, json.dumps(entry[engine]), flush=True)
            finally:
                E.stop_engine(proc, engine)
                time.sleep(30)

    n_table = json.loads(S.N_PER_CELL_FILE.read_text()) if S.N_PER_CELL_FILE.exists() else {}
    for key, per_engine in report.items():
        rps = max((v["req_per_s"] or 0) for v in per_engine.values())
        if rps <= 0:
            continue
        profile = key.split("/")[1].split("_")[0]
        per_unit = S.CHAT_TURNS if profile == "chat" else 1
        n_req = math.ceil(1.10 * S.MIN_WINDOW_SECONDS * rps)
        units = math.ceil(n_req / per_unit)
        units = int(math.ceil(units / S.CONCURRENCY) * S.CONCURRENCY)
        cap = S.POOL_REQUESTS // per_unit
        if units > cap:
            print(f"WARNING {key}: needs {units} units, pool has {cap}; capping", flush=True)
            units = cap
        n_table[key] = units
        per_engine["units_selected"] = units
    S.N_PER_CELL_FILE.write_text(json.dumps(n_table, indent=2, sort_keys=True))
    (root / "pilot_report.json").write_text(json.dumps(report, indent=2))
    (S.LOGS_DIR / "pilot_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(n_table, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
