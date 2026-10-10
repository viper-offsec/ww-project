# WattWatchers — vLLM vs. SGLang: energy and performance under controlled serving workloads

Replication package of the Green Lab 2026/2027 experiment (VU Amsterdam) by team WattWatchers.
It compares the **vLLM** and **SGLang** inference engines serving open-weight LLMs on an
**ASUS Ascent GX10 (NVIDIA GB10 Grace-Blackwell, 128 GB unified memory)**, measuring GPU energy,
serving performance and KV-cache behaviour across prompt profiles and context sizes.

**Run table:** [`data/Run_Table.csv`](data/Run_Table.csv) — one row per run (260 runs, all valid), produced by
experiment-runner; the raw measurements behind every row are in
[`data/raw/ww_vllm_sglang_gx10.tar.xz`](data/raw/ww_vllm_sglang_gx10.tar.xz).

## Main results

| Condition | Energy per output token (SGLang vs. vLLM) | End-to-end latency | Verdict (RQ4) |
|---|---|---|---|
| 512 input tokens | −3 % … +6 % (not significant) | ±1 % | practically equivalent |
| 2,048 input tokens | −2 % … −7 % | ±1 % | profile-dependent |
| 8,192 input tokens | −9 % … −13 % (Llama), −9 % … −14 % (Qwen) | −2 % … −9 % | SGLang dominates |

SGLang has a longer time-to-first-token (up to 2.3×) but a shorter time per output token (up to −27 %) and
more than twice the host CPU utilisation; both engines reach the same prefix-cache hit rates. Full results:
[`analysis/output/`](analysis/output/).

Carbon footprint of the whole experiment (SCI, Netherlands grid): about 1.1 kgCO2e (0.86–1.33), i.e. 4.3 gCO2e per
run; GPU energy measured with the NVML counter over all 55 h (1.31 kWh), rest of the system and embodied emissions
modelled from published data ([`analysis/04_sci.py`](analysis/04_sci.py)).

## Design at a glance

| Element | Setting |
|---|---|
| Main factor | Engine: vLLM 0.28.0, SGLang 0.5.19 (torch 2.13 / CUDA 13.0) |
| Moderator factors | Prompt profile: API, Chat, Agentic; Context size: 512, 2,048, 8,192 input tokens |
| Subjects | Llama-3.1-8B-Instruct (full 2×3×3 factorial); Qwen2.5-7B-Instruct (8 corner cells: {API, Agentic} × {512, 8192} × 2 engines) |
| Repetitions | 10 per cell, 260 runs, randomized complete blocks (every block = one full replicate) |
| Held constant | 128 output tokens/request (`min_tokens`=`max_tokens`, `ignore_eos`), concurrency 8, greedy decoding, seed 0, KV pool 81,920 tokens, max 16 running sequences, max context 16,384, BF16, prefix caching on |
| Energy | GPU energy from NVML's cumulative energy counter (10 Hz); integral of the NVML power readings as sensitivity analysis (EnergiBridge logs the same readings); GB10 exposes no CPU/system energy in software |
| Window | First request dispatch → last request completion of the measured phase |

Prompt profiles differ **only** in their prefix-reuse structure (measured on the generated request sets):

| Profile | Structure | Reusable prefix |
|---|---|---|
| API | fixed system prompt + unique user prompt | ≈1–6 % |
| Chat | 4-turn sessions, full history resent, canned 128-token replies, 64-token follow-ups | ≈60–75 % (session-local) |
| Agentic | global tool-schema prefix (75 % of the prompt) + unique task | ≈75 % (global) |

## Repository layout

```
ww/                     experiment logic (engines, load generator, samplers, guards, metrics)
  settings.py           every experimental constant (single source of truth)
  engines.py            matched vLLM / SGLang launch commands, health checks, shutdown
  loadgen.py            closed-loop replay client (identical requests to both engines)
  nvml_sampler.py       10 Hz NVML energy/power/temperature/clock + host CPU utilisation
  metrics_scraper.py    engine /metrics (KV usage, prefix-cache counters, preemptions)
  guards.py             foreign CPU/GPU activity detection (contaminated runs are re-queued)
  runmetrics.py         raw files -> run-table data columns
  runtable.py           run table: factorial + Qwen corners, block-randomised order
  hooks.py              experiment-runner lifecycle hooks
experiment/RunnerConfig.py   experiment-runner configuration (thin, delegates to ww/)
workloads/generate_workloads.py   builds the request sets with exact token budgets
scripts/                firstlight, smoke test, pilot calibration, run-table export, requeue
patches/                EnergiBridge build fix for ARM hosts
tools/experiment-runner      S2-group/experiment-runner @ 56620fa (git submodule, as used)
tools/EnergiBridge           tdurieux/EnergiBridge @ 2925da7 (git submodule) + patches/energibridge-arm-build.patch
data/Run_Table_planned.csv   the planned run table (260 runs)
data/Run_Table.csv           the final run table (260 valid runs, one row per run)
data/Run_Table_derived.csv   per-run columns derived from the raw samples (power-integral energy, block index)
analysis/01_derive.py   raw NVML samples -> data/Run_Table_derived.csv
analysis/02_stats.R     descriptive statistics, Shapiro-Wilk, Mann-Whitney U, Cliff's delta, ART-ANOVA/ART-C, RQ4 rule, sensitivity analyses
analysis/03_report.py   figures and LaTeX tables of the report (+ Q-Q plots in analysis/output/)
analysis/04_sci.py      carbon footprint of the experiment (Green Software Foundation SCI)
analysis/R_session_info.txt   R and package versions used for the analysis
analysis/output/        all statistical results as CSV
data/raw/ww_vllm_sglang_gx10.tar.xz   raw per-run data (NVML samples, per-request records, /metrics scrapes,
                        guard samples, engine logs, EnergiBridge CSVs) + environment.json + experiment-runner run table
data/calibration/       pilot report, requests per cell (n_per_cell.json), workload manifests
data/workloads/request_sets.tar.xz   the exact request sets of all cells (JSONL, one request/session per line)
logs/                   main experiment log, pilot, smoke tests, first-light, end-to-end test, installation logs
```

Clone with the tools: `git clone --recurse-submodules https://github.com/viper-offsec/ww-project.git`.
The exact request sets replayed in every cell (measured and warm-up, both models) are in
[`data/workloads/request_sets.tar.xz`](data/workloads/request_sets.tar.xz); they were generated from ShareGPT and
Glaive function-calling v2 (see the datasets' own licenses) by `workloads/generate_workloads.py` with fixed seeds,
and `data/calibration/manifest_*.json` records their token statistics.

## Reproducing

```bash
# on the testbed, inside ~/greenlab_exp (see scripts/setup_gx10.sh for the full environment)
python3 -m venv .venv-vllm   && .venv-vllm/bin/pip install vllm==0.28.0 ninja
python3 -m venv .venv-sglang && .venv-sglang/bin/pip install sglang==0.5.19 ninja
python3 -m venv .venv-runner && .venv-runner/bin/pip install -r ww-experiment/requirements-runner.txt
git clone https://github.com/S2-group/experiment-runner.git && git -C experiment-runner checkout 56620fa

cd ww-experiment
../.venv-runner/bin/python workloads/generate_workloads.py --model llama
../.venv-runner/bin/python workloads/generate_workloads.py --model qwen
../.venv-runner/bin/python scripts/smoke_test.py --engine vllm --model llama
../.venv-runner/bin/python scripts/pilot.py          # calibrates units per cell -> workloads/n_per_cell.json
cd .. && .venv-runner/bin/python experiment-runner/experiment-runner/ ww-experiment/experiment/RunnerConfig.py
# after an interruption: requeue invalid runs, then start the same command again (it resumes)
.venv-runner/bin/python ww-experiment/scripts/requeue_invalid.py experiments/ww_vllm_sglang_gx10
```

## Analysis

```bash
# needs the raw per-run directories (experiments/ww_vllm_sglang_gx10/) for step 1 only
python3 analysis/01_derive.py <raw experiment dir>
Rscript analysis/02_stats.R          # R >= 4.3 with ARTool and effsize
python3 analysis/03_report.py <report dir>   # pandas, numpy, matplotlib
python3 analysis/04_sci.py <raw experiment dir> [report dir]   # SCI footprint -> analysis/output/sci.csv
```

## Run-table columns

| Column | Meaning |
|---|---|
| `__run_id`, `__done` | experiment-runner bookkeeping |
| `model`, `engine`, `profile`, `context` | factor levels |
| `valid`, `invalid_reason` | 1 if no request errors, all outputs 128 tokens, no preemptions, no foreign activity, input within ±5 % |
| `n_requests`, `n_errors`, `n_output_tokens`, `n_input_tokens`, `input_tokens_mean`, `input_dev_pct` | workload actually served |
| `window_s` | measured window Δt_run (s) |
| `E_gpu_J` (M1) | GPU energy from the NVML energy counter over the window |
| `E_gpu_eb_J` | EnergiBridge power samples integrated over the window (not used in the analysis; the sensitivity analysis uses `E_gpu_pint_J` in `data/Run_Table_derived.csv`) |
| `E_total_J` | = `E_gpu_J` on the GX10 (no CPU/system energy counters) |
| `E_req_J` (M2), `E_tok_J` (M3), `P_gpu_mean_W` (M4) | energy per request, per output token, mean GPU power |
| `T_req_rps` (M5), `T_gen_tps` (M6) | request and output-token throughput |
| `ttft_ms_median` (M7), `tpot_ms_median` (M8), `e2e_ms_median` (M13), `*_p90` | per-run latency summaries |
| `eta_tok_per_J` (M9), `EDP_Js` (M10) | energy efficiency and energy-delay product |
| `kv_peak_pct` (M11), `prefix_hit_pct` (M12) | peak KV-pool utilisation, prompt tokens served from the prefix cache |
| `preemptions`, `contaminated` | validity checks |
| `gpu_temp_*`, `sm_clock_mean_MHz`, `gpu_util_mean_pct`, `cpu_util_mean_pct`, `engine_start_s` | control variables |
