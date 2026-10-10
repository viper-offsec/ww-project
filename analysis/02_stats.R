# Statistical analysis of the vLLM vs SGLang experiment (Section 4.5 of the report).
# Input : data/Run_Table.csv, data/Run_Table_derived.csv
# Output: analysis/output/*.csv
# Usage : Rscript analysis/02_stats.R   (from the ww-experiment directory)
.libPaths(c("~/Library/R/library", .libPaths()))
suppressPackageStartupMessages({ library(ARTool); library(effsize) })
set.seed(20260927)
dir.create("analysis/output", showWarnings = FALSE)

rt <- merge(read.csv("data/Run_Table.csv", check.names = FALSE), read.csv("data/Run_Table_derived.csv", check.names = FALSE), by = "__run_id")
names(rt)[1] <- "run_id"
stopifnot(nrow(rt) == 260, all(rt$valid == 1))
rt$engine  <- factor(rt$engine,  levels = c("vllm", "sglang"))
rt$profile <- factor(rt$profile, levels = c("api", "chat", "agentic"))
rt$context <- factor(rt$context, levels = c("small", "medium", "large"))

# ---------------------------------------------------------------- helpers
compare <- function(d, metric) {
  v <- d[d$engine == "vllm", metric]; s <- d[d$engine == "sglang", metric]
  w  <- suppressWarnings(wilcox.test(s, v, exact = NULL))
  cd <- cliff.delta(s, v)                       # delta > 0: SGLang larger
  boot <- replicate(5000, median(sample(s, replace = TRUE)) / median(sample(v, replace = TRUE)))
  data.frame(metric = metric, n_v = length(v), n_s = length(s),
             med_v = median(v), med_s = median(s),
             diff_pct = 100 * (median(s) / median(v) - 1),
             diff_lo = 100 * (quantile(boot, 0.025) - 1), diff_hi = 100 * (quantile(boot, 0.975) - 1),
             U = unname(w$statistic), p_raw = w$p.value,
             delta = unname(cd$estimate), delta_lo = cd$conf.int[1], delta_hi = cd$conf.int[2],
             magnitude = as.character(cd$magnitude), row.names = NULL)
}
cell <- function(m, p, c) rt[rt$model == m & rt$profile == p & rt$context == c, ]
gmean <- function(x) exp(mean(log(x)))

# ---------------------------------------------------------------- normality
metrics_all <- c("E_tok_J", "E_tok_pint_J", "P_gpu_mean_W", "window_s", "cpu_util_mean_pct", "T_gen_tps",
                 "ttft_ms_median", "tpot_ms_median", "e2e_ms_median", "kv_peak_pct", "prefix_hit_pct", "EDP_Js")
sw <- do.call(rbind, lapply(split(rt, list(rt$model, rt$profile, rt$context, rt$engine), drop = TRUE), function(d)
  do.call(rbind, lapply(metrics_all, function(mt) {
    x <- d[[mt]]
    p <- if (length(unique(x)) > 2) shapiro.test(x)$p.value else NA
    data.frame(model = d$model[1], profile = d$profile[1], context = d$context[1], engine = d$engine[1],
               metric = mt, W_p = p)
  }))))
write.csv(sw, "analysis/output/shapiro.csv", row.names = FALSE)

# ---------------------------------------------------------------- descriptive stats per cell
desc <- do.call(rbind, lapply(split(rt, list(rt$model, rt$profile, rt$context, rt$engine), drop = TRUE), function(d)
  do.call(rbind, lapply(metrics_all, function(mt) {
    x <- d[[mt]]
    data.frame(model = d$model[1], profile = d$profile[1], context = d$context[1], engine = d$engine[1],
               metric = mt, n = length(x), mean = mean(x), min = min(x), q1 = quantile(x, .25), median = median(x),
               q3 = quantile(x, .75), max = max(x), sd = sd(x), cv = sd(x) / mean(x), row.names = NULL)
  }))))
write.csv(desc, "analysis/output/descriptives.csv", row.names = FALSE)

# ---------------------------------------------------------------- RQ1 + RQ2 (Llama, API x Small)
base <- cell("llama", "api", "small")
# RQ1 family: H0 1.1 (E_tok), 1.2 (host CPU), 1.3a (mean GPU power), 1.3b (run duration); Holm over the four
rq1 <- rbind(compare(base, "E_tok_J"), compare(base, "cpu_util_mean_pct"),
             compare(base, "P_gpu_mean_W"), compare(base, "window_s"))
rq1$p_holm <- p.adjust(rq1$p_raw, "holm"); rq1$rq <- c("RQ1.1", "RQ1.2", "RQ1.3a", "RQ1.3b")
rq1_desc <- rbind(compare(base, "E_total_J"), compare(base, "E_req_J"))
rq1_desc$p_holm <- NA; rq1_desc$rq <- "RQ1 (descriptive)"
# RQ1.3 exact log-ratio decomposition on geometric means
v <- base[base$engine == "vllm", ]; s <- base[base$engine == "sglang", ]
dec <- data.frame(ln_E = log(gmean(s$E_gpu_J) / gmean(v$E_gpu_J)),
                  ln_P = log(gmean(s$P_gpu_mean_W) / gmean(v$P_gpu_mean_W)),
                  ln_t = log(gmean(s$window_s) / gmean(v$window_s)))
dec$check <- dec$ln_P + dec$ln_t - dec$ln_E
dec$share_P <- dec$ln_P / dec$ln_E; dec$share_t <- dec$ln_t / dec$ln_E
# bootstrap 95% CI of the three log-ratios (runs resampled within each engine)
bt <- t(replicate(5000, {
  vs <- v[sample(nrow(v), replace = TRUE), ]; ss <- s[sample(nrow(s), replace = TRUE), ]
  c(ln_E = log(gmean(ss$E_gpu_J) / gmean(vs$E_gpu_J)), ln_P = log(gmean(ss$P_gpu_mean_W) / gmean(vs$P_gpu_mean_W)),
    ln_t = log(gmean(ss$window_s) / gmean(vs$window_s)))
}))
for (k in c("ln_E", "ln_P", "ln_t")) {
  dec[[paste0(k, "_lo")]] <- quantile(bt[, k], 0.025); dec[[paste0(k, "_hi")]] <- quantile(bt[, k], 0.975)
}
write.csv(dec, "analysis/output/rq1_decomposition.csv", row.names = FALSE)
# the same decomposition for every condition (used in the discussion)
decall <- do.call(rbind, lapply(split(rt, list(rt$model, rt$profile, rt$context), drop = TRUE), function(d) {
  v <- d[d$engine == "vllm", ]; s <- d[d$engine == "sglang", ]
  data.frame(model = d$model[1], profile = d$profile[1], context = d$context[1],
             ln_E = log(gmean(s$E_gpu_J) / gmean(v$E_gpu_J)),
             ln_P = log(gmean(s$P_gpu_mean_W) / gmean(v$P_gpu_mean_W)),
             ln_t = log(gmean(s$window_s) / gmean(v$window_s)))
}))
write.csv(decall, "analysis/output/decomposition_all.csv", row.names = FALSE)

rq2 <- do.call(rbind, lapply(c("T_req_rps", "T_gen_tps", "ttft_ms_median", "tpot_ms_median", "kv_peak_pct"),
                             function(mt) compare(base, mt)))
rq2$p_holm <- p.adjust(rq2$p_raw, "holm"); rq2$rq <- c("RQ2.1", "RQ2.1", "RQ2.2", "RQ2.3", "RQ2.4")
write.csv(rbind(rq1, rq1_desc, rq2), "analysis/output/rq1_rq2.csv", row.names = FALSE)

# ---------------------------------------------------------------- RQ3: three-way ART-ANOVA on the 18 Llama cells
ll <- droplevels(rt[rt$model == "llama", ])
art_rows <- list(); con_rows <- list(); align_rows <- list()
for (mt in c("E_tok_J", "ttft_ms_median", "tpot_ms_median", "kv_peak_pct", "E_tok_pint_J")) {
  f <- as.formula(paste(mt, "~ engine * profile * context"))
  m <- art(f, data = ll)
  a <- anova(m)
  art_rows[[mt]] <- data.frame(metric = mt, term = as.character(a$Term), F = a$`F value`, df1 = a$Df,
                               df2 = a$Df.res, p = a$`Pr(>F)`,
                               eta2p = a$`F value` * a$Df / (a$`F value` * a$Df + a$Df.res))
  s <- summary(m)                                     # ART alignment check: F of aligned non-effects ~ 0
  align_rows[[mt]] <- data.frame(metric = mt, max_abs_F_nontarget = max(abs(s$aligned.anova$`F value`)))
  for (term in c("engine:profile", "engine:context", "engine:profile:context")) {
    cc <- as.data.frame(art.con(m, term, interaction = TRUE, adjust = "holm"))
    names(cc)[ncol(cc)] <- "p_holm"
    ctr <- cc[, !(names(cc) %in% c("estimate", "SE", "df", "t.ratio", "p_holm")), drop = FALSE]
    con_rows[[paste(mt, term)]] <- data.frame(metric = mt, term = term,
      contrast = apply(ctr, 1, paste, collapse = " | "), estimate = cc$estimate, SE = cc$SE, df = cc$df,
      t = cc$t.ratio, p_holm = cc$p_holm)
  }
}
write.csv(do.call(rbind, art_rows), "analysis/output/rq3_art_anova.csv", row.names = FALSE)
write.csv(do.call(rbind, con_rows), "analysis/output/rq3_artc_contrasts.csv", row.names = FALSE)
write.csv(do.call(rbind, align_rows), "analysis/output/rq3_art_alignment_check.csv", row.names = FALSE)

# per-cell engine effects (simple effects, used by RQ3 plots and RQ4); Holm per metric over the conditions of a model
simple <- do.call(rbind, lapply(split(rt, list(rt$model, rt$profile, rt$context), drop = TRUE), function(d)
  do.call(rbind, lapply(c("E_tok_J", "E_tok_pint_J", "eta_tok_per_J", "e2e_ms_median", "EDP_Js", "ttft_ms_median",
                          "tpot_ms_median", "kv_peak_pct", "prefix_hit_pct", "P_gpu_mean_W", "window_s",
                          "cpu_util_mean_pct", "T_gen_tps", "ttft_ms_p90", "tpot_ms_p90"),
                        function(mt) cbind(model = d$model[1], profile = d$profile[1], context = d$context[1],
                                           compare(d, mt))))))
simple$p_holm <- ave(simple$p_raw, simple$model, simple$metric, FUN = function(p) p.adjust(p, "holm"))
write.csv(simple, "analysis/output/simple_effects.csv", row.names = FALSE)

# ---------------------------------------------------------------- RQ4 decision rule
verdict <- function(se, eta_col = "eta_tok_per_J") {
  do.call(rbind, lapply(split(se, list(se$model, se$profile, se$context), drop = TRUE), function(d) {
    e <- d[d$metric == eta_col, ]; l <- d[d$metric == "e2e_ms_median", ]; x <- d[d$metric == "EDP_Js", ]
    sig <- function(r) r$p_holm < 0.05 && abs(r$delta) >= 0.147
    # "better": higher eta, lower latency; winner encoded as the engine name
    eb <- if (sig(e)) (if (e$delta > 0) "sglang" else "vllm") else "none"
    lb <- if (sig(l)) (if (l$delta < 0) "sglang" else "vllm") else "none"
    out <- if (eb == "none" && lb == "none") "equivalent" else
           if (eb != "none" && lb != "none" && eb != lb) "trade-off" else "dominance"
    win <- if (out == "dominance") setdiff(c(eb, lb), "none")[1] else
           if (out == "trade-off") (if (x$med_s < x$med_v) "sglang" else "vllm") else "none"
    data.frame(model = d$model[1], profile = d$profile[1], context = d$context[1],
               eta_diff_pct = e$diff_pct, eta_delta = e$delta, eta_p_holm = e$p_holm, eta_better = eb,
               e2e_diff_pct = l$diff_pct, e2e_delta = l$delta, e2e_p_holm = l$p_holm, e2e_better = lb,
               edp_diff_pct = x$diff_pct, edp_delta = x$delta, edp_p_holm = x$p_holm,
               outcome = out, preferred = win)
  }))
}
write.csv(verdict(simple), "analysis/output/rq4_matrix.csv", row.names = FALSE)

# ---------------------------------------------------------------- sensitivity analyses
# (a) power-integral energy instead of the NVML counter
se_p <- simple
se_p <- se_p[se_p$metric != "eta_tok_per_J", ]
tmp <- simple[simple$metric == "E_tok_pint_J", ]
tmp$delta <- -tmp$delta; tmp$diff_pct <- 100 * (tmp$med_v / tmp$med_s - 1); tmp$metric <- "eta_pint"
edp_p <- do.call(rbind, lapply(split(rt, list(rt$model, rt$profile, rt$context), drop = TRUE), function(d)
  cbind(model = d$model[1], profile = d$profile[1], context = d$context[1], compare(d, "EDP_pint_Js"))))
edp_p$p_holm <- ave(edp_p$p_raw, edp_p$model, FUN = function(p) p.adjust(p, "holm")); edp_p$metric <- "EDP_Js"
se_p <- rbind(se_p[se_p$metric != "EDP_Js", ], tmp, edp_p)
write.csv(verdict(se_p, "eta_pint"), "analysis/output/sens_rq4_matrix_pint.csv", row.names = FALSE)
write.csv(compare(base, "E_tok_pint_J"), "analysis/output/sens_rq1_pint.csv", row.names = FALSE)
# (b) IQR outlier filter (1.5 x IQR per cell and metric)
iqr_keep <- function(x) { q <- quantile(x, c(.25, .75)); h <- 1.5 * diff(q); x >= q[1] - h & x <= q[2] + h }
sens_iqr <- do.call(rbind, lapply(split(rt, list(rt$model, rt$profile, rt$context), drop = TRUE), function(d)
  do.call(rbind, lapply(c("E_tok_J", "ttft_ms_median", "tpot_ms_median", "e2e_ms_median", "kv_peak_pct"), function(mt) {
    k <- ave(d[[mt]], d$engine, FUN = function(x) as.numeric(iqr_keep(x))) == 1
    cbind(model = d$model[1], profile = d$profile[1], context = d$context[1], removed = sum(!k),
          compare(d[k, ], mt))
  }))))
sens_iqr$p_holm <- ave(sens_iqr$p_raw, sens_iqr$model, sens_iqr$metric, FUN = function(p) p.adjust(p, "holm"))
write.csv(sens_iqr, "analysis/output/sens_iqr.csv", row.names = FALSE)

# ---------------------------------------------------------------- run-order drift (Spearman rho of E_tok vs block per cell)
drift <- do.call(rbind, lapply(split(rt, list(rt$model, rt$profile, rt$context, rt$engine), drop = TRUE), function(d) {
  ct <- suppressWarnings(cor.test(d$block, d$E_tok_J, method = "spearman"))
  data.frame(model = d$model[1], profile = d$profile[1], context = d$context[1], engine = d$engine[1],
             rho = unname(ct$estimate), p = ct$p.value)
}))
drift$p_holm <- p.adjust(drift$p, "holm")
write.csv(drift, "analysis/output/drift.csv", row.names = FALSE)

# ---------------------------------------------------------------- thermal state (cool-down) checks
# (a) temperature each run starts from (after the previous run's cool-down), by engine
th <- rt[order(rt$t_start_run), ][-1, ]                 # the first run starts from the cold idle baseline
w_pre <- suppressWarnings(wilcox.test(temp_before_run_C ~ engine, data = th, exact = FALSE))
# (b) window-start temperature, by engine (includes the engine's own start-up and warm-up)
w_win <- suppressWarnings(wilcox.test(gpu_temp_start_C ~ engine, data = rt, exact = FALSE))
thermal <- data.frame(
  check = c("cooldown_s", "cooldown_end_temp_C", "temp_before_run_C (vllm)", "temp_before_run_C (sglang)",
            "gpu_temp_start_C (vllm)", "gpu_temp_start_C (sglang)"),
  min = c(min(rt$cooldown_s), min(rt$cooldown_end_temp_C), min(th$temp_before_run_C[th$engine == "vllm"]),
          min(th$temp_before_run_C[th$engine == "sglang"]), min(rt$gpu_temp_start_C[rt$engine == "vllm"]),
          min(rt$gpu_temp_start_C[rt$engine == "sglang"])),
  median = c(median(rt$cooldown_s), median(rt$cooldown_end_temp_C), median(th$temp_before_run_C[th$engine == "vllm"]),
             median(th$temp_before_run_C[th$engine == "sglang"]), median(rt$gpu_temp_start_C[rt$engine == "vllm"]),
             median(rt$gpu_temp_start_C[rt$engine == "sglang"])),
  mean = c(mean(rt$cooldown_s), mean(rt$cooldown_end_temp_C), mean(th$temp_before_run_C[th$engine == "vllm"]),
           mean(th$temp_before_run_C[th$engine == "sglang"]), mean(rt$gpu_temp_start_C[rt$engine == "vllm"]),
           mean(rt$gpu_temp_start_C[rt$engine == "sglang"])),
  max = c(max(rt$cooldown_s), max(rt$cooldown_end_temp_C), max(th$temp_before_run_C[th$engine == "vllm"]),
          max(th$temp_before_run_C[th$engine == "sglang"]), max(rt$gpu_temp_start_C[rt$engine == "vllm"]),
          max(rt$gpu_temp_start_C[rt$engine == "sglang"])),
  p_engine = c(NA, NA, w_pre$p.value, NA, w_win$p.value, NA))
write.csv(thermal, "analysis/output/thermal.csv", row.names = FALSE)
# (c) within-cell Spearman correlation of E_tok with the thermal state
thc <- do.call(rbind, lapply(split(rt, list(rt$model, rt$profile, rt$context, rt$engine), drop = TRUE), function(d) {
  a <- suppressWarnings(cor.test(d$gpu_temp_start_C, d$E_tok_J, method = "spearman", exact = FALSE))
  b <- suppressWarnings(cor.test(d$temp_before_run_C, d$E_tok_J, method = "spearman", exact = FALSE))
  data.frame(model = d$model[1], profile = d$profile[1], context = d$context[1], engine = d$engine[1],
             rho_window_start = unname(a$estimate), p_window_start = a$p.value,
             rho_before_run = unname(b$estimate), p_before_run = b$p.value)
}))
thc$p_window_start_holm <- p.adjust(thc$p_window_start, "holm")
thc$p_before_run_holm <- p.adjust(thc$p_before_run, "holm")
write.csv(thc, "analysis/output/thermal_correlation.csv", row.names = FALSE)
cat("done\n")
