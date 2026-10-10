"""Derive per-run columns that need the raw measurement files.

For every run of data/Run_Table.csv this recomputes, from the raw 10 Hz NVML samples,
the power-integral GPU energy used in the sensitivity analysis (Section 4.5), and adds
the block (repetition) index and the thermal state of every run (cool-down end temperature
and the temperature the run started from). Output: data/Run_Table_derived.csv (one row per run).

Usage: python analysis/01_derive.py <raw experiment dir>
       (e.g. ../experiment-data-backup/ww_vllm_sglang_gx10)
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ww.runmetrics import _integrate_power, _num  # noqa: E402

raw = Path(sys.argv[1])
rows = list(csv.DictReader(open(ROOT / "data" / "Run_Table.csv")))
out = []
for r in rows:
    run_dir = raw / r["__run_id"]
    reqs = [json.loads(line) for line in open(run_dir / "measure" / "requests.jsonl") if line.strip()]
    t0 = min(q["t_dispatch_ns"] for q in reqs)
    t1 = max(q["t_done_ns"] for q in reqs)
    ts, pw = [], []
    with open(run_dir / "nvml.csv") as f:
        for s in csv.DictReader(f):
            ts.append(int(s["t_ns"]))
            pw.append(_num(s["power_mW"]))
    e_pint = _integrate_power(ts, pw, t0, t1)
    n_out = float(r["n_output_tokens"])
    info = json.loads((run_dir / "run_info.json").read_text())
    out.append({
        "__run_id": r["__run_id"],
        "t_start_run": info["t_start_run"],
        "cooldown_s": info["cooldown_s"],
        "cooldown_end_temp_C": info["cooldown_end_temp_C"],
        "block": int(r["__run_id"].rsplit("_", 1)[1]) + 1,
        "E_gpu_pint_J": round(e_pint, 4),
        "E_tok_pint_J": round(e_pint / n_out, 6),
        "P_gpu_pint_W": round(e_pint / float(r["window_s"]), 4),
        "EDP_pint_Js": round(e_pint * float(r["window_s"]), 3),
        "pint_vs_counter_pct": round(100 * (e_pint / float(r["E_gpu_J"]) - 1), 3),
    })

# temperature each run starts from = cool-down end temperature of the previous run in
# execution order (the first run starts from the idle baseline of the experiment)
idle_T = json.loads((raw / "idle_baseline.json").read_text())["temp_C"]
prev_T = idle_T
for o in sorted(out, key=lambda o: o["t_start_run"]):
    o["temp_before_run_C"] = prev_T
    prev_T = o["cooldown_end_temp_C"]

dst = ROOT / "data" / "Run_Table_derived.csv"
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print(f"wrote {dst} ({len(out)} runs)")
