"""Carbon footprint of the experiment following the Green Software Foundation SCI specification.

    SCI = (E * I + M) per R

E   operational energy of the whole experiment (all 260 runs incl. engine start, warm-up, cool-down):
    - GPU: measured with NVML's cumulative energy counter, differenced between consecutive runs;
    - rest of the system (CPU, memory, storage, NIC, power supply): not observable on the GB10, modelled
      as a constant power = published idle wall power of GB10 systems - our measured idle GPU power.
I   carbon intensity of the Dutch grid in 2025 (Electricity Maps, lifecycle).
M   embodied emissions of the GX10 (no published product carbon footprint; proxy: Apple Mac Studio).
R   functional units: one valid experimental run; one million generated output tokens.

Usage: python analysis/04_sci.py <raw experiment dir> [report dir]
Writes analysis/output/sci.csv and, if a report dir is given, <report>/tables/sci.tex.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
raw = Path(sys.argv[1])
report = Path(sys.argv[2]) if len(sys.argv) > 2 else None

# ---------------------------------------------------------------- assumptions (with sources)
IDLE_WALL_W = {"low": 22.0,   # Tom's Hardware (2026): DGX Spark headless idle after NIC hot-plug update
               "mid": 37.0,   # Tom's Hardware (2025/2026): DGX Spark idle before the update
               "high": 45.0}  # ServeTheHome (2025): DGX Spark idle 40-45 W
GRID_G_PER_KWH = {"low": 262.2, "mid": 262.2,   # Electricity Maps, NL 2025, flow-traced (consumption), lifecycle
                  "high": 273.8}                # Electricity Maps, NL 2025, production-based
# Apple Mac Studio PER (March 2025): 276 kg (M4 Max) and 382 kg (M3 Ultra) life cycle; 41 % product use
EMBODIED_KG = {"low": 276 * 0.59, "mid": (276 + 382) / 2 * 0.59, "high": 382 * 0.59}
LIFETIME_H = 4 * 365 * 24      # 4-year expected lifetime (as in Apple's PER use scenario)

# ---------------------------------------------------------------- measured inputs
rows = list(csv.DictReader(open(ROOT / "data" / "Run_Table.csv")))
runs = []
for r in rows:
    d = raw / r["__run_id"]
    info = json.loads((d / "run_info.json").read_text())
    with open(d / "nvml.csv") as f:
        rd = csv.DictReader(f)
        first = next(rd)
        last = first
        for last in rd:
            pass
    runs.append({"t_start": info["t_start_run"], "cool_s": info["cooldown_s"],
                 "e0_mJ": float(first["energy_mJ"]), "t0": int(first["t_ns"]) / 1e9, "t1": int(last["t_ns"]) / 1e9})
runs.sort(key=lambda x: x["t0"])
e_cycles_J = sum((b["e0_mJ"] - a["e0_mJ"]) / 1000 for a, b in zip(runs, runs[1:]))
t_cycles_s = runs[-1]["t0"] - runs[0]["t0"]
t_exp_s = (runs[-1]["t1"] + runs[-1]["cool_s"]) - min(x["t_start"] for x in runs)   # first engine start -> last cool-down
e_gpu_kwh = e_cycles_J * (t_exp_s / t_cycles_s) / 3.6e6                            # extrapolate the uncovered ~1 %
idle_gpu_w = json.loads((raw / "idle_baseline.json").read_text())["power_W"]
n_runs = sum(1 for r in rows if r["valid"] in ("1", "1.0"))
n_tok = sum(float(r["n_output_tokens"]) for r in rows)
t_exp_h = t_exp_s / 3600

out = []
for sc in ("low", "mid", "high"):
    p_rest = IDLE_WALL_W[sc] - idle_gpu_w
    e_rest = p_rest * t_exp_h / 1000
    e_tot = e_gpu_kwh + e_rest
    o_kg = e_tot * GRID_G_PER_KWH[sc] / 1000
    m_kg = EMBODIED_KG[sc] * t_exp_h / LIFETIME_H
    c_kg = o_kg + m_kg
    out.append({"scenario": sc, "duration_h": round(t_exp_h, 2), "E_gpu_kWh": round(e_gpu_kwh, 3),
                "P_rest_W": round(p_rest, 1), "E_rest_kWh": round(e_rest, 3), "E_kWh": round(e_tot, 3),
                "I_g_per_kWh": GRID_G_PER_KWH[sc], "O_kg": round(o_kg, 3), "TE_kg": round(EMBODIED_KG[sc], 1),
                "M_kg": round(m_kg, 3), "C_kg": round(c_kg, 3), "R_runs": n_runs, "R_Mtok": round(n_tok / 1e6, 3),
                "SCI_g_per_run": round(1000 * c_kg / n_runs, 2), "SCI_g_per_Mtok": round(1000 * c_kg / (n_tok / 1e6), 1)})

dst = ROOT / "analysis" / "output" / "sci.csv"
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
for o in out:
    print(o)

if report:
    lo, mid, hi = out
    f2 = lambda k, nd=2: " & ".join(f"{o[k]:.{nd}f}" for o in out)
    lines = [r"\begin{table}[!t]",
             r"\caption{Carbon footprint of the experiment (SCI) in three scenarios. Measured: GPU energy, "
             r"duration, $R$. Modelled: rest-of-system power, grid intensity, embodied emissions (see text).}",
             r"\label{tab:sci}", r"\scriptsize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{lrrr}", r"\toprule",
             r"\textbf{Component} & \textbf{Low} & \textbf{Mid} & \textbf{High} \\", r"\midrule",
             rf"$E_{{\mathrm{{GPU}}}}$, measured (kWh) & {f2('E_gpu_kWh')} \\",
             rf"Rest-of-system power (W) & {f2('P_rest_W', 1)} \\",
             rf"$E_{{\mathrm{{rest}}}}$, modelled (kWh) & {f2('E_rest_kWh')} \\",
             rf"$E$ (kWh) & {f2('E_kWh')} \\",
             rf"$I$ (gCO$_2$e/kWh) & {f2('I_g_per_kWh', 1)} \\",
             rf"$E \cdot I$ (kgCO$_2$e) & {f2('O_kg')} \\",
             rf"$M$ (kgCO$_2$e) & {f2('M_kg')} \\",
             r"\midrule",
             rf"Total $C = E \cdot I + M$ (kgCO$_2$e) & {f2('C_kg')} \\",
             rf"SCI per run (gCO$_2$e/run) & {f2('SCI_g_per_run', 1)} \\",
             rf"SCI per $10^6$ output tokens (gCO$_2$e) & {f2('SCI_g_per_Mtok', 0)} \\",
             r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (report / "tables" / "sci.tex").write_text("\n".join(lines) + "\n")
    print("wrote", report / "tables" / "sci.tex")
