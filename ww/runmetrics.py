"""Turn the raw files of one run into the run-table data columns.

Raw files in <run_dir>/:
  measure/requests.jsonl  per-request timestamps and usage (ww.loadgen)
  nvml.csv                10 Hz NVML samples incl. cumulative energy (ww.nvml_sampler)
  energibridge.csv        EnergiBridge samples (cross-check, optional)
  metrics.jsonl           engine /metrics scrapes (ww.metrics_scraper)
  guard.jsonl             foreign-activity monitor (ww.guards)
  run_info.json           engine start time, precheck result, factors

Measurement window = [first request dispatch, last request completion].
"""
import csv
import json
import math
import statistics
from pathlib import Path

from ww import settings as S

DATA_COLUMNS = [
    "valid", "invalid_reason",
    "n_requests", "n_errors", "n_output_tokens", "n_input_tokens", "input_tokens_mean", "input_dev_pct",
    "window_s",
    "E_gpu_J", "E_gpu_eb_J", "E_total_J", "E_req_J", "E_tok_J", "eta_tok_per_J", "P_gpu_mean_W", "EDP_Js",
    "T_req_rps", "T_gen_tps",
    "ttft_ms_median", "tpot_ms_median", "e2e_ms_median", "ttft_ms_p90", "tpot_ms_p90",
    "kv_peak_pct", "prefix_hit_pct", "preemptions",
    "gpu_temp_start_C", "gpu_temp_max_C", "sm_clock_mean_MHz", "gpu_util_mean_pct", "cpu_util_mean_pct",
    "engine_start_s", "contaminated",
]

# fraction (0-1) of the KV pool held by requests (evictable cached blocks excluded);
# first gauge present wins (SGLang 0.5.x keeps sglang:token_usage at 0, full_token_usage is live)
KV_GAUGES = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc",
             "sglang:full_token_usage", "sglang:token_usage")


def _read_jsonl(p: Path) -> list:
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _num(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _pct(values, q):
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def _interp(ts, vs, t):
    """Linear interpolation of a monotone series at time t (ns)."""
    if not ts:
        return None
    if t <= ts[0]:
        return vs[0]
    if t >= ts[-1]:
        return vs[-1]
    lo, hi = 0, len(ts) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ts[mid] <= t:
            lo = mid
        else:
            hi = mid
    f = (t - ts[lo]) / (ts[hi] - ts[lo]) if ts[hi] != ts[lo] else 0.0
    return vs[lo] + f * (vs[hi] - vs[lo])


def _integrate_power(ts_ns, p_mw, t0, t1):
    """Trapezoidal integral of power (mW) over [t0, t1] (ns) -> Joules."""
    pts = [(t, p) for t, p in zip(ts_ns, p_mw) if p is not None]
    if len(pts) < 2:
        return None
    e = 0.0
    for (ta, pa), (tb, pb) in zip(pts, pts[1:]):
        a, b = max(ta, t0), min(tb, t1)
        if b <= a:
            continue
        pa_c = pa + (pb - pa) * (a - ta) / (tb - ta)
        pb_c = pa + (pb - pa) * (b - ta) / (tb - ta)
        e += (pa_c + pb_c) / 2.0 / 1000.0 * (b - a) / 1e9
    return e


def gpu_energy(nvml_csv: Path, t0: int, t1: int) -> dict:
    ts, en, pw, temp, clk, gutil, cpu = [], [], [], [], [], [], []
    with open(nvml_csv) as f:
        for r in csv.DictReader(f):
            ts.append(int(r["t_ns"]))
            en.append(_num(r["energy_mJ"]))
            pw.append(_num(r["power_mW"]))
            temp.append(_num(r["temp_C"]))
            clk.append(_num(r["sm_clock_MHz"]))
            gutil.append(_num(r["gpu_util_pct"]))
            cpu.append(_num(r["cpu_util_pct"]))
    res = {}
    e_pairs = [(t, e) for t, e in zip(ts, en) if e is not None]
    if len(e_pairs) >= 2:
        et, ev = zip(*e_pairs)
        res["E_gpu_J"] = (_interp(list(et), list(ev), t1) - _interp(list(et), list(ev), t0)) / 1000.0
    else:
        res["E_gpu_J"] = _integrate_power(ts, pw, t0, t1)
    inside = [i for i, t in enumerate(ts) if t0 <= t <= t1]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.fmean(xs), 3) if xs else None
    res["gpu_temp_start_C"] = next((temp[i] for i in inside if temp[i] is not None), None)
    res["gpu_temp_max_C"] = max((temp[i] for i in inside if temp[i] is not None), default=None)
    res["sm_clock_mean_MHz"] = mean([clk[i] for i in inside])
    res["gpu_util_mean_pct"] = mean([gutil[i] for i in inside])
    res["cpu_util_mean_pct"] = mean([cpu[i] for i in inside])
    res["_power_integral_J"] = _integrate_power(ts, pw, t0, t1)
    return res


def energibridge_energy(eb_csv: Path, t0: int, t1: int):
    if not eb_csv.exists():
        return None
    ts, pw = [], []
    with open(eb_csv) as f:
        rd = csv.DictReader(f)
        col = next((c for c in (rd.fieldnames or []) if c.startswith("GPU0_POWER")), None)
        if col is None:
            return None
        for r in rd:
            t = _num(r.get("Time"))
            if t is None:
                continue
            ts.append(int(t * 1e6))          # EnergiBridge Time is epoch ms
            pw.append(_num(r[col]))
    return _integrate_power(ts, pw, t0, t1)


def engine_metrics(metrics_jsonl: Path, t0: int, t1: int) -> dict:
    rows = [r for r in _read_jsonl(metrics_jsonl) if "m" in r]
    if not rows:
        return {"kv_peak_pct": None, "_hit_counters": None, "preemptions": None}
    win = [r for r in rows if t0 - 1e9 <= r["t_ns"] <= t1 + 1e9] or rows
    gauge = next((g for g in KV_GAUGES if any(g in r["m"] for r in win)), None)
    kv = [r["m"][gauge] for r in win if gauge and gauge in r["m"]]
    first, last = win[0]["m"], win[-1]["m"]

    def delta(pred):
        keys = [k for k in last if pred(k) and k.endswith("_total")]
        if not keys:
            return None
        return sum(last[k] - first.get(k, 0.0) for k in keys)
    hits = delta(lambda k: "prefix_cache_hits" in k and "external" not in k)
    queries = delta(lambda k: "prefix_cache_queries" in k and "external" not in k)
    if hits is None:   # SGLang: cached vs prompt token counters
        hits = delta(lambda k: k == "sglang:cached_tokens_total")
        queries = delta(lambda k: k == "sglang:prompt_tokens_total")
    hit_gauge = [r["m"]["sglang:cache_hit_rate"] for r in win if "sglang:cache_hit_rate" in r["m"]]
    preempt = delta(lambda k: "preempt" in k or "retract" in k)
    retract_gauge = [v for r in win for k, v in r["m"].items() if "retract" in k and not k.endswith("_total")]
    if retract_gauge:   # SGLang exposes retracted requests as a gauge
        preempt = max(preempt or 0.0, max(retract_gauge))
    return {
        "kv_peak_pct": round(100.0 * max(kv), 3) if kv else None,
        "_kv_gauge": gauge,
        "_hit_counters": (hits, queries),
        "_hit_gauge_mean": statistics.fmean(hit_gauge) if hit_gauge else None,
        "preemptions": preempt,
    }


def contamination(guard_jsonl: Path, t0: int, t1: int) -> bool:
    for r in _read_jsonl(guard_jsonl):
        if t0 <= r.get("t_ns", 0) <= t1 and (r.get("foreign_cpu") or r.get("foreign_gpu")):
            return True
    return False


def compute(run_dir: Path, profile: str, context: str) -> dict:
    run_dir = Path(run_dir)
    out = {k: None for k in DATA_COLUMNS}
    info = json.loads((run_dir / "run_info.json").read_text()) if (run_dir / "run_info.json").exists() else {}
    out["engine_start_s"] = info.get("engine_start_s")
    reqs = _read_jsonl(run_dir / "measure" / "requests.jsonl")
    reasons = list(info.get("errors", []))
    if not reqs:
        out["valid"] = 0
        out["invalid_reason"] = ";".join(reasons + ["no_requests"])
        return out
    ok = [r for r in reqs if not r["error"]]
    t0 = min(r["t_dispatch_ns"] for r in reqs)
    t1 = max(r["t_done_ns"] for r in reqs)
    window = (t1 - t0) / 1e9
    n_out = sum(r["completion_tokens"] or 0 for r in ok)
    n_in = sum(r["prompt_tokens"] or 0 for r in ok)
    target = S.CONTEXTS[context]
    out.update({
        "n_requests": len(ok), "n_errors": len(reqs) - len(ok),
        "n_output_tokens": n_out, "n_input_tokens": n_in,
        "input_tokens_mean": round(n_in / len(ok), 2) if ok else None,
        "window_s": round(window, 3),
    })
    if ok:
        out["input_dev_pct"] = round(100.0 * (out["input_tokens_mean"] - target) / target, 3)
        ttft = [(r["t_first_token_ns"] - r["t_dispatch_ns"]) / 1e6 for r in ok if r["t_first_token_ns"]]
        tpot = [(r["t_last_token_ns"] - r["t_first_token_ns"]) / 1e6 / (r["completion_tokens"] - 1)
                for r in ok if r["t_first_token_ns"] and r["completion_tokens"] and r["completion_tokens"] > 1]
        e2e = [(r["t_done_ns"] - r["t_dispatch_ns"]) / 1e6 for r in ok]
        out.update({
            "ttft_ms_median": round(statistics.median(ttft), 3) if ttft else None,
            "tpot_ms_median": round(statistics.median(tpot), 3) if tpot else None,
            "e2e_ms_median": round(statistics.median(e2e), 3),
            "ttft_ms_p90": round(_pct(ttft, 0.9), 3) if ttft else None,
            "tpot_ms_p90": round(_pct(tpot, 0.9), 3) if tpot else None,
            "T_req_rps": round(len(ok) / window, 5),
            "T_gen_tps": round(n_out / window, 3),
        })
        cached = [r.get("cached_tokens") for r in ok]
        if all(c is not None for c in cached) and n_in:
            out["prefix_hit_pct"] = round(100.0 * sum(cached) / n_in, 3)

    g = gpu_energy(run_dir / "nvml.csv", t0, t1) if (run_dir / "nvml.csv").exists() else {}
    for k in ("E_gpu_J", "gpu_temp_start_C", "gpu_temp_max_C", "sm_clock_mean_MHz",
              "gpu_util_mean_pct", "cpu_util_mean_pct"):
        out[k] = g.get(k)
    out["E_gpu_eb_J"] = energibridge_energy(run_dir / "energibridge.csv", t0, t1)
    em = engine_metrics(run_dir / "metrics.jsonl", t0, t1)
    out["kv_peak_pct"] = em["kv_peak_pct"]
    out["preemptions"] = em["preemptions"]
    if out["prefix_hit_pct"] is None:
        hq = em.get("_hit_counters")
        if hq and hq[0] is not None and hq[1]:
            out["prefix_hit_pct"] = round(100.0 * hq[0] / hq[1], 3)
        elif em.get("_hit_gauge_mean") is not None:
            out["prefix_hit_pct"] = round(100.0 * em["_hit_gauge_mean"], 3)

    E = out["E_gpu_J"]
    out["E_total_J"] = E          # GX10: only GPU energy is observable in software
    if E and ok:
        out.update({
            "E_req_J": round(E / len(ok), 5),
            "E_tok_J": round(E / n_out, 6) if n_out else None,
            "eta_tok_per_J": round(n_out / E, 5),
            "P_gpu_mean_W": round(E / window, 4),
            "EDP_Js": round(E * window, 3),
        })
        out["E_gpu_J"] = round(E, 4)
        out["E_total_J"] = round(E, 4)
    if out["E_gpu_eb_J"] is not None:
        out["E_gpu_eb_J"] = round(out["E_gpu_eb_J"], 4)

    out["contaminated"] = int(contamination(run_dir / "guard.jsonl", t0, t1))
    if out["n_errors"]:
        reasons.append("request_errors")
    if any((r["completion_tokens"] or 0) != S.OUTPUT_TOKENS for r in ok):
        reasons.append("output_length")
    if out["preemptions"]:
        reasons.append("preemptions")
    if out["contaminated"]:
        reasons.append("contaminated")
    if out.get("input_dev_pct") is not None and abs(out["input_dev_pct"]) > 5:
        reasons.append("input_length")
    if E is None:
        reasons.append("no_energy")
    out["valid"] = 0 if reasons else 1
    out["invalid_reason"] = ";".join(reasons)
    return out


if __name__ == "__main__":
    import sys
    print(json.dumps(compute(Path(sys.argv[1]), sys.argv[2], sys.argv[3]), indent=2))
