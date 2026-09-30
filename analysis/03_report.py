"""Figures and LaTeX tables for the report (Section 6), from the outputs of 02_stats.R.

Usage: python analysis/03_report.py [report dir]   (default: ../ww-assignment2)
Writes <report>/figures/*.pdf and <report>/tables/*.tex.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "output"
REPORT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "ww-assignment2"
FIG, TAB = REPORT / "figures", REPORT / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

rt = pd.read_csv(ROOT / "data" / "Run_Table.csv").merge(pd.read_csv(ROOT / "data" / "Run_Table_derived.csv"),
                                                       on="__run_id")
desc = pd.read_csv(OUT / "descriptives.csv")
simple = pd.read_csv(OUT / "simple_effects.csv")
base = pd.read_csv(OUT / "rq1_rq2.csv")
art = pd.read_csv(OUT / "rq3_art_anova.csv")
rq4 = pd.read_csv(OUT / "rq4_matrix.csv")
rq4p = pd.read_csv(OUT / "sens_rq4_matrix_pint.csv")

PROFILES = ["api", "chat", "agentic"]
CONTEXTS = ["small", "medium", "large"]
PNAME = {"api": "API", "chat": "Chat", "agentic": "Agentic"}
CNAME = {"small": "Small", "medium": "Medium", "large": "Large"}
CTOK = {"small": "512", "medium": "2,048", "large": "8,192"}
ENAME = {"vllm": "vLLM", "sglang": "SGLang"}
VCOL = {"vllm": "#9ecae1", "sglang": "#fdd49e"}      # violin fill per engine
MEAN_KW = dict(marker="D", color="#1f3fbf", markersize=3.2, linestyle="none", zorder=5)
REF_KW = dict(color="#d62728", linestyle="--", linewidth=1.0, zorder=1)

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "axes.linewidth": 0.6, "pdf.fonttype": 42})


def cellv(model, profile, context, engine, metric):
    d = rt[(rt.model == model) & (rt.profile == profile) & (rt.context == context) & (rt.engine == engine)]
    return d[metric].to_numpy(dtype=float)


def violin_box(ax, data, positions, colors):
    """Apsan-style distribution: light violin, black box plot, blue diamond at the mean."""
    for x, d, c in zip(positions, data, colors):
        if np.ptp(d) > 0:
            v = ax.violinplot([d], positions=[x], widths=0.8, showextrema=False)
            for b in v["bodies"]:
                b.set_facecolor(c)
                b.set_edgecolor("none")
                b.set_alpha(0.9)
    ax.boxplot(data, positions=positions, widths=0.38, patch_artist=True, showfliers=True,
               boxprops=dict(facecolor="white", edgecolor="black", linewidth=0.7),
               medianprops=dict(color="black", linewidth=0.9), whiskerprops=dict(linewidth=0.7),
               capprops=dict(linewidth=0.7), flierprops=dict(marker="o", markersize=2, markeredgewidth=0.4))
    ax.plot(positions, [np.mean(d) for d in data], **MEAN_KW)


# ------------------------------------------------------------------ Figure: baseline cell (RQ1, RQ2)
def fig_baseline():
    panels = [("E_tok_J", r"$E_{tok}$ (J/token)"), ("P_gpu_mean_W", r"$\bar{P}_{GPU}$ (W)"),
              ("window_s", r"$\Delta t_{run}$ (s)"), ("cpu_util_mean_pct", "CPU util. (%)"),
              ("T_gen_tps", r"$T_{gen}$ (tokens/s)"), ("ttft_ms_median", r"$t_{TTFT}$ (ms)"),
              ("tpot_ms_median", r"$t_{TPOT}$ (ms)"), ("kv_peak_pct", r"$U_{KV}$ (%)")]
    fig, axes = plt.subplots(2, 4, figsize=(3.4, 2.55))
    for ax, (m, label) in zip(axes.flat, panels):
        data = [cellv("llama", "api", "small", e, m) for e in ("vllm", "sglang")]
        violin_box(ax, data, [0, 1], [VCOL["vllm"], VCOL["sglang"]])
        ax.set_title(label, fontsize=6.5, pad=2)
        ax.set_xticks([0, 1], ["vLLM", "SGLang"], rotation=35, ha="right")
        ax.set_xlim(-0.6, 1.6)
        ax.tick_params(length=2, pad=1)
        ax.margins(y=0.15)
    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.6)
    fig.savefig(FIG / "baseline.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Figure: E_tok of the 18 Llama cells (RQ3)
def fig_energy(metric="E_tok_J", fname="energy.pdf", ylabel="Energy per output token (J/token)"):
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 3.9), sharex=True)
    pos = np.arange(6)
    for ax, c in zip(axes, CONTEXTS):
        ax.axvspan(1.5, 3.5, color="#e6e6e6", zorder=0)
        data, cols = [], []
        for p in PROFILES:
            for e in ("vllm", "sglang"):
                data.append(cellv("llama", p, c, e, metric))
                cols.append(VCOL[e])
        ax.axhline(np.median(data[0]), **REF_KW)
        violin_box(ax, data, pos, cols)
        ax.text(0.985, 0.9, f"{CNAME[c]} ({CTOK[c]} tokens)", transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="black", lw=0.5))
        ax.margins(y=0.18)
        ax.tick_params(length=2, pad=1)
    axes[-1].set_xticks(pos, ["vLLM", "SGLang"] * 3)
    for i, p in enumerate(PROFILES):
        axes[-1].text(2 * i + 0.5, -0.30, PNAME[p], transform=axes[-1].get_xaxis_transform(), ha="center",
                      va="top", fontsize=7, fontweight="bold")
    fig.supylabel(ylabel, fontsize=7, x=0.02)
    h = [plt.Line2D([], [], **REF_KW), plt.Line2D([], [], **MEAN_KW),
         plt.Rectangle((0, 0), 1, 1, fc=VCOL["vllm"]), plt.Rectangle((0, 0), 1, 1, fc=VCOL["sglang"])]
    fig.legend(h, ["vLLM API median", "Mean value", "vLLM", "SGLang"], loc="lower center", ncol=4,
               frameon=True, fontsize=6, handlelength=1.4, columnspacing=0.9, bbox_to_anchor=(0.54, 0.0))
    fig.tight_layout(pad=0.3, h_pad=0.4, rect=(0.03, 0.07, 1, 1))
    fig.savefig(FIG / fname)
    plt.close(fig)


# ------------------------------------------------------------------ Figure: interaction plots of the engine gap (RQ3)
def fig_interaction():
    panels = [("E_tok_J", r"$E_{tok}$"), ("ttft_ms_median", r"$t_{TTFT}$"),
              ("tpot_ms_median", r"$t_{TPOT}$"), ("kv_peak_pct", r"$U_{KV}$")]
    style = {"api": ("#1f77b4", "o"), "chat": ("#2ca02c", "s"), "agentic": ("#d62728", "^")}
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.75))
    x = np.arange(3)
    for ax, (m, label) in zip(axes.flat, panels):
        ax.axhline(0, color="black", linewidth=0.6)
        for p in PROFILES:
            col, mk = style[p]
            s = simple[(simple.metric == m) & (simple.model == "llama") & (simple.profile == p)]
            y = [s[s.context == c].diff_pct.iloc[0] for c in CONTEXTS]
            ax.plot(x, y, color=col, marker=mk, markersize=3.2, linewidth=1.0, label=f"{PNAME[p]} (Llama)")
            q = simple[(simple.metric == m) & (simple.model == "qwen") & (simple.profile == p)]
            if len(q):
                xq = [0, 2]
                yq = [q[q.context == c].diff_pct.iloc[0] for c in ("small", "large")]
                ax.plot(xq, yq, color=col, marker=mk, markersize=3.2, linewidth=0.8, linestyle=":",
                        markerfacecolor="white", label=f"{PNAME[p]} (Qwen)")
        ax.set_title(label, fontsize=7, pad=2)
        ax.set_xticks(x, ["Small", "Medium", "Large"])
        ax.set_xlim(-0.3, 2.3)
        ax.tick_params(length=2, pad=1)
        ax.grid(axis="y", linewidth=0.3, color="#cccccc")
    fig.supylabel("SGLang vs. vLLM, difference of medians (%)", fontsize=7, x=0.02)
    hs, ls = axes[0, 0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="lower center", ncol=3, fontsize=5.8, frameon=True, handlelength=2.0,
               columnspacing=0.8, bbox_to_anchor=(0.54, 0.0))
    fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.6, rect=(0.03, 0.13, 1, 1))
    fig.savefig(FIG / "interaction.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Figure: energy-latency trade-off (RQ4)
VERD = {"sglang": "SGLang dominates", "vllm": "vLLM dominates", "none": "equivalent"}


def verdict_text(r):
    if r.outcome == "equivalent":
        return "Equivalent"
    if r.outcome == "trade-off":
        return f"Trade-off, EDP: {ENAME[r.preferred]}"
    return f"{ENAME[r.preferred]} dominates"


def fig_tradeoff():
    fig, axes = plt.subplots(3, 3, figsize=(3.4, 3.5))
    for i, p in enumerate(PROFILES):
        for j, c in enumerate(CONTEXTS):
            ax = axes[i, j]
            for e, mk in (("vllm", "o"), ("sglang", "^")):
                ax.scatter(cellv("llama", p, c, e, "e2e_ms_median") / 1000, cellv("llama", p, c, e, "eta_tok_per_J"),
                           s=9, marker=mk, facecolor=VCOL[e], edgecolor="black", linewidth=0.4, label=ENAME[e],
                           zorder=3)
            r = rq4[(rq4.model == "llama") & (rq4.profile == p) & (rq4.context == c)].iloc[0]
            ax.set_title(verdict_text(r), fontsize=5.8, pad=2,
                         color={"sglang": "#b35806", "vllm": "#08519c"}.get(r.preferred, "black"))
            ax.tick_params(length=1.5, pad=1, labelsize=5.2)
            ax.margins(0.15)
            if j == 0:
                ax.set_ylabel(f"{PNAME[p]}\n" + r"$\eta$ (tok/J)", fontsize=6.2)
            if i == 2:
                ax.set_xlabel(f"$t_{{E2E}}$ (s)\n{CNAME[c]}", fontsize=6.2)
    hs, ls = axes[0, 0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="lower center", ncol=2, fontsize=6.2, frameon=True, bbox_to_anchor=(0.55, 0.0))
    fig.tight_layout(pad=0.3, h_pad=0.45, w_pad=0.35, rect=(0, 0.045, 1, 1))
    fig.savefig(FIG / "tradeoff.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ LaTeX helpers
MAGCOL = {"large": "efL", "medium": "efM", "small": "efS", "negligible": "efN"}


def fmt_p(p):
    return r"$<$1e$-$4" if p < 1e-4 else r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def signed(x, nd=1):
    s = f"{x:+.{nd}f}"
    return s.replace("-", "$-$")


def num(x, nd=3):
    return f"{x:,.{nd}f}".replace(",", "{,}").replace("-", "$-$")


# ------------------------------------------------------------------ Table: descriptive statistics of E_tok
def tab_etok():
    def block(model, conds, header_groups):
        rows = {k: [] for k in ("Mean", "Min", "Median", "Max", "Std", "CV")}
        best = []
        for p, c in conds:
            meds = {}
            for e in ("vllm", "sglang"):
                d = desc[(desc.model == model) & (desc.profile == p) & (desc.context == c) & (desc.engine == e)
                         & (desc.metric == "E_tok_J")].iloc[0]
                rows["Mean"].append(d["mean"]); rows["Min"].append(d["min"]); rows["Median"].append(d["median"])
                rows["Max"].append(d["max"]); rows["Std"].append(d["sd"]); rows["CV"].append(d["cv"])
                meds[e] = d["median"]
            best.append(min(meds, key=meds.get))
        n = len(conds) * 2
        lines = [r"\toprule"]
        lines.append(" & " + " & ".join(rf"\multicolumn{{{2 * k}}}{{c}}{{\textbf{{{g}}}}}" for g, k in header_groups)
                     + r" \\")
        cm, col = [], 2
        for g, k in header_groups:
            cm.append(rf"\cmidrule(lr){{{col}-{col + 2 * k - 1}}}")
            col += 2 * k
        lines.append(" ".join(cm))
        lines.append(" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{CNAME[c]}}}" for _, c in conds) + r" \\")
        lines.append(" & " + " & ".join(["V", "S"] * len(conds)) + r" \\")
        lines.append(r"\midrule")
        for k, vals in rows.items():
            cells = []
            for i, v in enumerate(vals):
                txt = num(v, 3)
                if k == "Median" and ("vllm", "sglang")[i % 2] == best[i // 2]:
                    txt = r"\cellcolor{best}" + txt
                cells.append(txt)
            lines.append(rf"\textbf{{{k}}} & " + " & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        return n, lines

    n1, l1 = block("llama", [(p, c) for p in PROFILES for c in CONTEXTS], [(PNAME[p], 3) for p in PROFILES])
    n2, l2 = block("qwen", [(p, c) for p in ("api", "agentic") for c in ("small", "large")],
                   [("API", 2), ("Agentic", 2)])
    out = [r"\begin{table*}[!t]", r"\caption{Energy per output token $E_{\mathrm{tok}}$ (J/token) per cell: "
           r"Llama-3.1-8B (top) and Qwen2.5-7B replication (bottom). V: vLLM, S: SGLang, Std: standard "
           r"deviation, CV: coefficient of variation; the lower median of each condition is highlighted.}",
           r"\label{tab:etok}", r"\scriptsize", r"\setlength{\tabcolsep}{2.6pt}", r"\centering",
           rf"\begin{{tabular}}{{l{'r' * n1}}}", *l1, r"\end{tabular}", r"\\[4pt]",
           rf"\begin{{tabular}}{{l{'r' * n2}}}", *l2, r"\end{tabular}", r"\end{table*}"]
    (TAB / "etok.tex").write_text("\n".join(out) + "\n")


# ------------------------------------------------------------------ Table: baseline tests (RQ1, RQ2)
def tab_baseline():
    names = {"E_tok_J": r"$E_{\mathrm{tok}}$ (J/tok)", "cpu_util_mean_pct": r"Host CPU (\%)",
             "E_total_J": r"$E_{\mathrm{total}}$ (J)", "E_req_J": r"$E_{\mathrm{req}}$ (J/req)",
             "P_gpu_mean_W": r"$\bar{P}_{\mathrm{GPU}}$ (W)", "window_s": r"$\Delta t_{\mathrm{run}}$ (s)",
             "T_req_rps": r"$T_{\mathrm{req}}$ (req/s)", "T_gen_tps": r"$T_{\mathrm{gen}}$ (tok/s)",
             "ttft_ms_median": r"$t_{\mathrm{TTFT}}$ (ms)", "tpot_ms_median": r"$t_{\mathrm{TPOT}}$ (ms)",
             "kv_peak_pct": r"$U_{\mathrm{KV}}$ (\%)"}
    order = [("RQ1.1", "E_tok_J"), ("RQ1", "E_total_J"), ("RQ1", "E_req_J"), ("RQ1.3", "P_gpu_mean_W"),
             ("RQ1.3", "window_s"), ("RQ1.2", "cpu_util_mean_pct"), ("RQ2.1", "T_req_rps"), ("RQ2.1", "T_gen_tps"),
             ("RQ2.2", "ttft_ms_median"), ("RQ2.3", "tpot_ms_median"), ("RQ2.4", "kv_peak_pct")]
    lines = [r"\begin{table}[!t]", r"\caption{Engine comparison in the baseline condition (Llama, API $\times$ "
             r"Small; $n = 10$ runs per engine): medians, difference of medians with bootstrap 95\% CI, "
             r"Holm-adjusted Mann-Whitney $p$ (RQ1.1--1.2 and RQ2 families; -- = descriptive), and Cliff's "
             r"$\delta$ (SGLang vs.\ vLLM), coloured by magnitude.}", r"\label{tab:baseline}", r"\scriptsize",
             r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"\textbf{RQ} & \textbf{Metric} & \textbf{vLLM} & \textbf{SGLang} & \textbf{$\Delta$\% [95\% CI]} & "
             r"\textbf{$p_{\mathrm{Holm}}$} & \textbf{$\delta$} \\", r"\midrule"]
    for rq, m in order:
        r = base[base.metric == m].iloc[0]
        nd = 3 if r.med_v < 1 else 2 if r.med_v < 100 else 1
        ci = f"{signed(r.diff_pct)} [{signed(r.diff_lo)}, {signed(r.diff_hi)}]"
        p = "--" if pd.isna(r.p_holm) else fmt_p(r.p_holm)
        if not pd.isna(r.p_holm) and r.p_holm < 0.05:
            p = rf"\textbf{{{p}}}"
        d = rf"\cellcolor{{{MAGCOL[r.magnitude]}}}{signed(r.delta, 2)}"
        lines.append(f"{rq} & {names[m]} & {num(r.med_v, nd)} & {num(r.med_s, nd)} & {ci} & {p} & {d} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "baseline.tex").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ Table: ART-ANOVA (RQ3)
def tab_art():
    terms = ["engine", "profile", "context", "engine:profile", "engine:context", "profile:context",
             "engine:profile:context"]
    tname = {"engine": r"Engine ($\alpha$)", "profile": r"Profile ($\beta$)", "context": r"Context ($\gamma$)",
             "engine:profile": r"Engine $\times$ Profile ($\alpha\beta$), RQ3.1",
             "engine:context": r"Engine $\times$ Context ($\alpha\gamma$), RQ3.2",
             "profile:context": r"Profile $\times$ Context ($\beta\gamma$)",
             "engine:profile:context": r"Engine $\times$ Profile $\times$ Context ($\alpha\beta\gamma$), RQ3.3"}
    mets = [("E_tok_J", r"$E_{\mathrm{tok}}$"), ("ttft_ms_median", r"$t_{\mathrm{TTFT}}$"),
            ("tpot_ms_median", r"$t_{\mathrm{TPOT}}$"), ("kv_peak_pct", r"$U_{\mathrm{KV}}$ (RQ3.4)"),
            ("E_tok_pint_J", r"$E_{\mathrm{tok}}$ (power int.)")]
    lines = [r"\begin{table*}[!t]", r"\caption{Three-way ART-ANOVA on the 18 Llama cells ($n = 180$ runs): "
             r"$F$ statistic and partial $\eta^2_p$ per term and outcome. All 35 terms are significant with "
             r"$p < 10^{-4}$ (df$_1$ = 1, 2, or 4; df$_2$ = 162). The last column is the sensitivity analysis "
             r"with the power-integral energy estimate.}", r"\label{tab:art}", r"\footnotesize",
             r"\begin{tabular}{l" + "rr" * len(mets) + "}", r"\toprule",
             r"\textbf{Term} & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{\textbf{{{n}}}}}" for _, n in mets) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(mets))),
             " & " + " & ".join([r"$F$ & $\eta^2_p$"] * len(mets)) + r" \\", r"\midrule"]
    for t in terms:
        cells = []
        for m, _ in mets:
            r = art[(art.metric == m) & (art.term == t)].iloc[0]
            assert r.p < 1e-4, (m, t, r.p)
            cells.append(f"{r.F:.1f} & {r.eta2p:.2f}")
        name = tname[t]
        if t.startswith("engine:"):
            name = rf"\textbf{{{name}}}"
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TAB / "art.tex").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ Table: per-condition engine effects (Apsan Table 5 style)
def tab_effects():
    mets = [("E_tok_J", r"$E_{\mathrm{tok}}$"), ("E_tok_pint_J", r"$E_{\mathrm{tok}}^{\mathrm{P}}$"),
            ("P_gpu_mean_W", r"$\bar{P}_{\mathrm{GPU}}$"), ("window_s", r"$\Delta t_{\mathrm{run}}$"),
            ("ttft_ms_median", r"$t_{\mathrm{TTFT}}$"), ("tpot_ms_median", r"$t_{\mathrm{TPOT}}$"),
            ("e2e_ms_median", r"$t_{\mathrm{E2E}}$"), ("kv_peak_pct", r"$U_{\mathrm{KV}}$"),
            ("prefix_hit_pct", r"$H_{\mathrm{prefix}}$"), ("cpu_util_mean_pct", r"CPU")]
    lines = [r"\begin{table*}[!t]", r"\caption{Engine effect per condition: difference of medians of SGLang "
             r"relative to vLLM (\%), coloured by the magnitude of Cliff's $\delta$; bold: Holm-adjusted "
             r"Mann-Whitney $p < 0.05$ (Holm over the conditions of each model). $E_{\mathrm{tok}}^{\mathrm{P}}$: "
             r"power-integral estimate. Negative values mean that SGLang is lower.}", r"\label{tab:effects}",
             r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lll" + "r" * len(mets) + "}",
             r"\toprule", r"\textbf{Model} & \textbf{Profile} & \textbf{Context} & "
             + " & ".join(n for _, n in mets) + r" \\", r"\midrule"]
    first = True
    for model in ("llama", "qwen"):
        if not first:
            lines.append(r"\midrule")
        first = False
        for p in PROFILES:
            for c in CONTEXTS:
                s = simple[(simple.model == model) & (simple.profile == p) & (simple.context == c)]
                if s.empty:
                    continue
                cells = []
                for m, _ in mets:
                    r = s[s.metric == m].iloc[0]
                    txt = signed(r.diff_pct)
                    if r.p_holm < 0.05:
                        txt = rf"\textbf{{{txt}}}"
                    cells.append(rf"\cellcolor{{{MAGCOL[r.magnitude]}}}{txt}")
                mname = {"llama": "Llama", "qwen": "Qwen"}[model]
                lines.append(f"{mname} & {PNAME[p]} & {CNAME[c]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule",
              r"\multicolumn{" + str(3 + len(mets)) + r"}{l}{Effect size (Cliff's $\delta$): "
              r"\colorbox{efL}{Large} \colorbox{efM}{Medium} \colorbox{efS}{Small} \colorbox{efN}{Negligible}}"
              r" \\", r"\end{tabular}", r"\end{table*}"]
    (TAB / "effects.tex").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ Table: RQ4 decision matrix
def tab_rq4():
    fill = {"sglang": "domS", "vllm": "domV", "none": "efN"}
    lines = [r"\begin{table}[!t]", r"\caption{RQ4 decision matrix (Llama): outcome of the dominance rule per "
             r"condition with the difference of medians of SGLang vs.\ vLLM in efficiency $\eta$ and latency "
             r"$t_{\mathrm{E2E}}$ (bold: Holm $p < 0.05$ and $|\delta| \geq 0.147$). \ding{51}: same verdict in "
             r"the Qwen replication; $\dagger$: different verdict with the power-integral energy estimate.}",
             r"\label{tab:rq4}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.25}",
             r"\begin{tabular}{l" + "c" * 3 + "}", r"\toprule",
             r"& " + " & ".join(rf"\textbf{{{CNAME[c]}}} ({CTOK[c]})" for c in CONTEXTS) + r" \\", r"\midrule"]
    for p in PROFILES:
        cells = []
        for c in CONTEXTS:
            r = rq4[(rq4.model == "llama") & (rq4.profile == p) & (rq4.context == c)].iloc[0]
            rp = rq4p[(rq4p.model == "llama") & (rq4p.profile == p) & (rq4p.context == c)].iloc[0]
            q = rq4[(rq4.model == "qwen") & (rq4.profile == p) & (rq4.context == c)]
            mark = ""
            if len(q) and (q.iloc[0].outcome, q.iloc[0].preferred) == (r.outcome, r.preferred):
                mark += r" \ding{51}"
            if (rp.outcome, rp.preferred) != (r.outcome, r.preferred):
                mark += r"$^{\dagger}$"
            head = {"equivalent": "Equivalent", "trade-off": f"Trade-off (EDP: {ENAME.get(r.preferred, '')})",
                    "dominance": f"{ENAME.get(r.preferred, '')} dominates"}[r.outcome]
            sig_e = r.eta_p_holm < 0.05 and abs(r.eta_delta) >= 0.147
            sig_l = r.e2e_p_holm < 0.05 and abs(r.e2e_delta) >= 0.147
            e = signed(r.eta_diff_pct)
            lt = signed(r.e2e_diff_pct)
            e = rf"\textbf{{{e}}}" if sig_e else e
            lt = rf"\textbf{{{lt}}}" if sig_l else lt
            cells.append(rf"\cellcolor{{{fill[r.preferred]}}}\makecell{{\textbf{{{head}}}{mark}\\"
                         rf"$\eta$ {e}\%, $t_{{\mathrm{{E2E}}}}$ {lt}\%}}")
        lines.append(rf"\textbf{{{PNAME[p]}}} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "rq4.tex").write_text("\n".join(lines) + "\n")


def fig_qq():
    """Normal Q-Q plots per cell (replication package only)."""
    from statistics import NormalDist
    cells = rt.groupby(["model", "profile", "context", "engine"])
    for metric in ("E_tok_J", "ttft_ms_median", "tpot_ms_median", "e2e_ms_median"):
        fig, axes = plt.subplots(4, 7, figsize=(12, 7))
        for ax, ((m, p, c, e), d) in zip(axes.flat, cells):
            x = np.sort(d[metric].to_numpy(dtype=float))
            n = len(x)
            q = [NormalDist().inv_cdf((i + 0.5) / n) for i in range(n)]
            ax.plot(q, x, "o", markersize=3)
            if x.std() > 0:
                ax.plot(q, x.mean() + x.std(ddof=1) * np.array(q), "r-", linewidth=0.8)
            ax.set_title(f"{m} {p} {c} {e}", fontsize=6)
            ax.tick_params(labelsize=5)
        for ax in list(axes.flat)[len(cells):]:
            ax.axis("off")
        fig.suptitle(f"Normal Q-Q plots: {metric}")
        fig.tight_layout()
        fig.savefig(OUT / f"qq_{metric}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    fig_qq()
    fig_baseline()
    fig_energy()
    fig_interaction()
    fig_tradeoff()
    tab_etok()
    tab_baseline()
    tab_art()
    tab_effects()
    tab_rq4()
    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))
    print("tables:", sorted(p.name for p in TAB.glob("*.tex")))
