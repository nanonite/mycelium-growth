#!/usr/bin/env Rscript
# Fit the synthetic radial-growth data with the #39 priors and assess recovery.
#
# Run from the repository root:
  #   nix develop --command Rscript R/fit_growth.R
#   nix develop --command Rscript R/fit_growth.R <input_dir> <output_dir>
#
# This is a synthetic parameter-recovery exercise, not a fit to MyceliumSeg.

suppressPackageStartupMessages(library(rethinking))

fit_synthetic_growth <- function(
  input_dir = "data/derived/dish_growth_simulation",
  output_dir = "data/derived/growth_parameter_recovery",
  seed = 20261012L,
  chains = 4L,
  iter = 2000L,
  warmup = 1000L
) {
  required_files <- c("observations.csv", "dish_parameters.csv", "known_parameters.csv")
  missing_files <- required_files[!file.exists(file.path(input_dir, required_files))]
  if (length(missing_files)) {
    stop("Missing simulation inputs: ", paste(missing_files, collapse = ", "), call. = FALSE)
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  observations <- read.csv(file.path(input_dir, "observations.csv"), stringsAsFactors = FALSE)
  dishes <- read.csv(file.path(input_dir, "dish_parameters.csv"), stringsAsFactors = FALSE)
  known <- read.csv(file.path(input_dir, "known_parameters.csv"), stringsAsFactors = FALSE)
  config <- read.csv(file.path(input_dir, "simulation_config.csv"), stringsAsFactors = FALSE)
  max_radius <- as.numeric(config$value[config$parameter == "max_radius_mm"])
  if (length(max_radius) != 1L || !is.finite(max_radius)) {
    max_radius <- 45
  }

  # rethinking 2.21 emits legacy Stan array declarations. Its RStan parser
  # accepts them, while the pinned CmdStan 2.35 compiler rejects them. Use the
  # compatible RStan backend and host GCC for this fit only.
  makevars <- tempfile("growth40-Makevars-")
  writeLines(c("CXX14 = /usr/bin/g++", "CXX17 = /usr/bin/g++"), makevars)
  old_makevars <- Sys.getenv("R_MAKEVARS_USER", unset = NA_character_)
  old_use_cmdstan <- rethinking::ulam_options$use_cmdstan
  Sys.setenv(R_MAKEVARS_USER = makevars)
  assign("use_cmdstan", FALSE, envir = rethinking::ulam_options)
  on.exit({
    assign("use_cmdstan", old_use_cmdstan, envir = rethinking::ulam_options)
    if (is.na(old_makevars)) Sys.unsetenv("R_MAKEVARS_USER") else
      Sys.setenv(R_MAKEVARS_USER = old_makevars)
    unlink(makevars)
  }, add = TRUE)
  rstan::rstan_options(auto_write = TRUE)

  dishes <- dishes[order(dishes$dish_id), , drop = FALSE]
  species_levels <- sort(unique(dishes$species))
  dish_index <- match(observations$dish_id, dishes$dish_id)
  if (anyNA(dish_index)) stop("Observation rows reference unknown dishes.", call. = FALSE)

  model_data <- list(
    R_obs = as.numeric(observations$R_obs_mm),
    time_days = as.numeric(observations$time_days),
    dish_obs = as.integer(dish_index),
    n_obs = as.integer(nrow(observations)),
    n_dish = as.integer(nrow(dishes)),
    n_species = as.integer(length(species_levels)),
    species_dish = as.integer(match(dishes$species, species_levels)),
    is_15C_dish = as.numeric(dishes$temperature_C == 15),
    max_radius_mm = max_radius
  )

  # These are the #39 prior distributions. Explicit lower bounds turn each
  # zero-centered Normal scale prior into the intended HalfNormal prior.
  model <- alist(
    R_obs ~ normal(mu, sigma_seg_mm),
      vector[n_obs]: mu <- fmin(max_radius_mm,
        elt_multiply(
          exp(alpha[species_dish[dish_obs]] + beta_T_15 * is_15C_dish[dish_obs] +
                z_k[dish_obs] * sigma_dish_log_k),
          fmax(0, time_days - exp(gamma[species_dish[dish_obs]] +
            delta_T_15 * is_15C_dish[dish_obs] + z_tau[dish_obs] * sigma_dish_log_tau))
        )) + bias_mm,

      vector[n_species]: alpha ~ normal(mu_log_k, sigma_species_log_k),
      vector[n_species]: gamma ~ normal(mu_log_tau, sigma_species_log_tau),
      vector[n_dish]: z_k ~ normal(0, 1),
      vector[n_dish]: z_tau ~ normal(0, 1),

    mu_log_k ~ normal(log(3), 0.55),
    sigma_species_log_k ~ normal(0, 0.40),
    sigma_dish_log_k ~ normal(0, 0.20),
    beta_T_15 ~ normal(0, 0.35),
    mu_log_tau ~ normal(log(3), 0.45),
    sigma_species_log_tau ~ normal(0, 0.35),
    sigma_dish_log_tau ~ normal(0, 0.20),
    delta_T_15 ~ normal(0, 0.35),
    bias_mm ~ normal(0, 0.5),
    sigma_seg_mm ~ normal(0, 2)
  )
  half_normal_constraints <- list(
    sigma_species_log_k = "lower=0",
    sigma_dish_log_k = "lower=0",
    sigma_species_log_tau = "lower=0",
    sigma_dish_log_tau = "lower=0",
    sigma_seg_mm = "lower=0"
  )
  max_treedepth <- 12L

  cat(sprintf(
    "Fitting %d observations from %d dishes in %d species using %d chains (%d iterations, %d warmup).\n",
    nrow(observations), nrow(dishes), length(species_levels), chains, iter, warmup
  ))
  fit <- ulam(
    model, data = model_data, chains = chains, cores = chains,
    iter = iter, warmup = warmup, seed = seed,
    control = list(adapt_delta = 0.99, max_treedepth = max_treedepth),
    constraints = half_normal_constraints,
    cmdstan = FALSE, log_lik = FALSE, messages = TRUE, coerce_int = FALSE
  )

  saveRDS(fit, file.path(output_dir, "ulam_fit.rds"), compress = "xz")
  writeLines(as.character(fit@model), file.path(output_dir, "ulam_model.stan"))
  posterior <- extract.samples(fit)
  saveRDS(posterior, file.path(output_dir, "posterior_draws.rds"), compress = "xz")

  if (inherits(fit@stanfit, "CmdStanMCMC")) {
    diagnostics <- fit@stanfit$summary()
  } else {
    draws_array <- posterior::as_draws_array(fit@stanfit)
    diagnostics <- posterior::summarise_draws(draws_array)
  }
  write.csv(diagnostics, file.path(output_dir, "mcmc_diagnostics.csv"), row.names = FALSE)
  sampler_params <- rstan::get_sampler_params(fit@stanfit, inc_warmup = FALSE)
  sampler_diagnostics <- do.call(rbind, lapply(seq_along(sampler_params), function(chain) {
    x <- sampler_params[[chain]]
    data.frame(
      chain = chain,
      sampling_iterations = nrow(x),
      divergent_transitions = sum(x[, "divergent__"] > 0),
      treedepth_limit_hits = sum(x[, "treedepth__"] >= max_treedepth),
      max_treedepth_observed = max(x[, "treedepth__"]),
      mean_accept_stat = mean(x[, "accept_stat__"]),
      stringsAsFactors = FALSE
    )
  }))
  write.csv(sampler_diagnostics,
    file.path(output_dir, "sampler_diagnostics.csv"), row.names = FALSE)

  as_draw_matrix <- function(name, width) {
    value <- posterior[[name]]
    if (is.null(value)) stop("Expected posterior variable not found: ", name, call. = FALSE)
    matrix(value, ncol = width)
  }
  n_draws <- length(posterior$mu_log_k)
  alpha <- as_draw_matrix("alpha", length(species_levels))
  gamma <- as_draw_matrix("gamma", length(species_levels))
  z_k <- as_draw_matrix("z_k", nrow(dishes))
  z_tau <- as_draw_matrix("z_tau", nrow(dishes))
  beta <- as.numeric(posterior$beta_T_15)
  delta <- as.numeric(posterior$delta_T_15)
  sigma_k <- as.numeric(posterior$sigma_dish_log_k)
  sigma_tau <- as.numeric(posterior$sigma_dish_log_tau)

  k_dish <- exp(alpha[, model_data$species_dish, drop = FALSE] +
    beta * rep(model_data$is_15C_dish, each = n_draws) + z_k * sigma_k)
  tau_dish <- exp(gamma[, model_data$species_dish, drop = FALSE] +
    delta * rep(model_data$is_15C_dish, each = n_draws) + z_tau * sigma_tau)

  summarize <- function(draws, truth = NA_real_, parameter, units) {
    q <- quantile(draws, c(0.025, 0.5, 0.975), names = FALSE)
    data.frame(
      parameter = parameter, truth = truth, posterior_mean = mean(draws),
      posterior_median = q[2], lower_95 = q[1], upper_95 = q[3],
      truth_in_95 = if (is.na(truth)) NA else truth >= q[1] && truth <= q[3],
      units = units, stringsAsFactors = FALSE
    )
  }

  fixed_rows <- list()
  for (s in seq_along(species_levels)) {
    species <- species_levels[s]
    truth_k <- exp(known$value[match(paste0("alpha_log_k[", species, "]"), known$parameter)])
    truth_tau <- exp(known$value[match(paste0("gamma_log_tau[", species, "]"), known$parameter)])
    fixed_rows[[length(fixed_rows) + 1L]] <- summarize(exp(alpha[, s]), truth_k,
      paste0("k_25_mm_day[", species, "]"), "mm/day")
    fixed_rows[[length(fixed_rows) + 1L]] <- summarize(exp(gamma[, s]), truth_tau,
      paste0("tau_25_days[", species, "]"), "days")
  }
  known_value <- function(name) known$value[match(name, known$parameter)]
  fixed_rows <- c(fixed_rows, list(
    summarize(exp(beta), exp(known_value("beta_T_15")), "rate_ratio_15_vs_25", "ratio"),
    summarize(exp(delta), exp(known_value("delta_T_15")), "lag_ratio_15_vs_25", "ratio"),
    summarize(sigma_k, known_value("sigma_log_k"), "sigma_dish_log_k", "log-rate SD"),
    summarize(sigma_tau, known_value("sigma_log_tau"), "sigma_dish_log_tau", "log-lag SD"),
    summarize(posterior$bias_mm, known_value("bias_mm"), "bias_mm", "mm"),
    summarize(posterior$sigma_seg_mm, known_value("sigma_seg_mm"), "sigma_seg_mm", "mm")
  ))
  # The generator fixes species baselines directly, so it has no generating
  # population-level means or between-species SDs to recover.
  for (name in c("mu_log_k", "sigma_species_log_k", "mu_log_tau", "sigma_species_log_tau")) {
    fixed_rows[[length(fixed_rows) + 1L]] <- summarize(posterior[[name]], NA_real_, name,
      "hyperparameter; no generating truth")
  }
  fixed_summary <- do.call(rbind, fixed_rows)
  write.csv(fixed_summary, file.path(output_dir, "fixed_parameter_recovery.csv"), row.names = FALSE)

  dish_rows <- lapply(seq_len(nrow(dishes)), function(d) {
    k_truth <- dishes$k_mm_day[d]
    tau_truth <- dishes$tau_days[d]
    k_q <- quantile(k_dish[, d], c(0.025, 0.5, 0.975), names = FALSE)
    tau_q <- quantile(tau_dish[, d], c(0.025, 0.5, 0.975), names = FALSE)
    data.frame(
      dish_id = dishes$dish_id[d], species = dishes$species[d],
      temperature_C = dishes$temperature_C[d],
      k_truth_mm_day = k_truth, k_median_mm_day = k_q[2],
      k_lower_95 = k_q[1], k_upper_95 = k_q[3], k_truth_in_95 = k_truth >= k_q[1] && k_truth <= k_q[3],
      tau_truth_days = tau_truth, tau_median_days = tau_q[2],
      tau_lower_95 = tau_q[1], tau_upper_95 = tau_q[3],
      tau_truth_in_95 = tau_truth >= tau_q[1] && tau_truth <= tau_q[3],
      stringsAsFactors = FALSE
    )
  })
  dish_summary <- do.call(rbind, dish_rows)
  write.csv(dish_summary, file.path(output_dir, "dish_parameter_recovery.csv"), row.names = FALSE)

  # Posterior expected radius by observation; summarize at condition × day.
  pred <- matrix(NA_real_, nrow = n_draws, ncol = nrow(observations))
  for (i in seq_len(nrow(observations))) {
    d <- dish_index[i]
    pred[, i] <- pmin(max_radius,
      k_dish[, d] * pmax(0, observations$time_days[i] - tau_dish[, d])) + posterior$bias_mm
  }
  point_predictions <- data.frame(
    dish_id = observations$dish_id,
    species = observations$species,
    temperature_C = observations$temperature_C,
    time_days = observations$time_days,
    R_obs_mm = observations$R_obs_mm,
    R_true_mm = observations$R_true_mm,
    pred_median_mm = apply(pred, 2, median),
    pred_lower_95_mm = apply(pred, 2, quantile, probs = 0.025),
    pred_upper_95_mm = apply(pred, 2, quantile, probs = 0.975),
    stringsAsFactors = FALSE
  )
  write.csv(point_predictions, file.path(output_dir, "observation_predictions.csv"), row.names = FALSE)

  cell_keys <- unique(observations[c("species", "temperature_C", "time_days")])
  cell_predictions <- lapply(seq_len(nrow(cell_keys)), function(j) {
    idx <- which(observations$species == cell_keys$species[j] &
      observations$temperature_C == cell_keys$temperature_C[j] &
      observations$time_days == cell_keys$time_days[j])
    pred_cell <- rowMeans(pred[, idx, drop = FALSE])
    q <- quantile(pred_cell, c(0.025, 0.5, 0.975), names = FALSE)
    data.frame(
      species = cell_keys$species[j], temperature_C = cell_keys$temperature_C[j],
      time_days = cell_keys$time_days[j], observed_mean_mm = mean(observations$R_obs_mm[idx]),
      predicted_mean_mm = q[2], lower_95_mm = q[1], upper_95_mm = q[3],
      stringsAsFactors = FALSE
    )
  })
  cell_predictions <- do.call(rbind, cell_predictions)
  write.csv(cell_predictions, file.path(output_dir, "condition_curve_recovery.csv"), row.names = FALSE)

  svg(file.path(output_dir, "condition_curve_recovery.svg"), width = 10, height = 7)
  old_par <- par(mfrow = c(2, 3), mar = c(4, 4, 2.5, 1))
  cells <- unique(cell_predictions[c("species", "temperature_C")])
  for (j in seq_len(nrow(cells))) {
    d <- cell_predictions[cell_predictions$species == cells$species[j] &
      cell_predictions$temperature_C == cells$temperature_C[j], ]
    plot(d$time_days, d$observed_mean_mm, pch = 16, col = "#555555",
      xlab = "Simulation day", ylab = "Radius (mm)", ylim = c(0, max_radius + 3),
      main = paste0(cells$species[j], " at ", cells$temperature_C[j], " C"))
    polygon(c(d$time_days, rev(d$time_days)), c(d$lower_95_mm, rev(d$upper_95_mm)),
      col = grDevices::adjustcolor("#277DA1", alpha.f = 0.20), border = NA)
    lines(d$time_days, d$predicted_mean_mm, col = "#277DA1", lwd = 2)
    points(d$time_days, d$observed_mean_mm, pch = 16, col = "#555555")
    abline(h = max_radius, lty = 2, col = "#C44536")
    grid(col = "grey88")
  }
  plot.new()
  legend("center", legend = c("Observed cell mean", "Posterior mean prediction", "95% credible band", "45 mm cap"),
    pch = c(16, NA, 15, NA), lty = c(NA, 1, NA, 2), lwd = c(NA, 2, NA, 1),
    col = c("#555555", "#277DA1", grDevices::adjustcolor("#277DA1", alpha.f = 0.20), "#C44536"),
    bty = "n")
  par(old_par)
  dev.off()

  rmse <- sqrt(mean((point_predictions$pred_median_mm - point_predictions$R_true_mm)^2))
  coverage <- mean(point_predictions$R_true_mm >= point_predictions$pred_lower_95_mm &
    point_predictions$R_true_mm <= point_predictions$pred_upper_95_mm)
  diagnostic_summary <- data.frame(
    n_observations = nrow(observations), n_dishes = nrow(dishes),
    latent_radius_rmse_mm = rmse, latent_truth_95_coverage = coverage,
    k_dish_truth_95_coverage = mean(dish_summary$k_truth_in_95),
    tau_dish_truth_95_coverage = mean(dish_summary$tau_truth_in_95),
    max_rhat = max(diagnostics$rhat, na.rm = TRUE),
    n_parameters_rhat_over_1_01 = sum(diagnostics$rhat > 1.01, na.rm = TRUE),
    min_ess_bulk = min(diagnostics$ess_bulk, na.rm = TRUE),
    min_ess_tail = min(diagnostics$ess_tail, na.rm = TRUE),
    divergent_transitions = sum(sampler_diagnostics$divergent_transitions),
    treedepth_limit_hits = sum(sampler_diagnostics$treedepth_limit_hits),
    stringsAsFactors = FALSE
  )
  write.csv(diagnostic_summary, file.path(output_dir, "recovery_diagnostics.csv"), row.names = FALSE)
  cat(sprintf(
    "Saved fit and recovery summaries to %s. Dish truth coverage: k %.1f%%; tau %.1f%%.\n",
    output_dir, 100 * diagnostic_summary$k_dish_truth_95_coverage,
    100 * diagnostic_summary$tau_dish_truth_95_coverage
  ))
  invisible(list(fit = fit, diagnostics = diagnostic_summary))
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  input_dir <- if (length(args) >= 1L) args[[1L]] else "data/derived/dish_growth_simulation"
  output_dir <- if (length(args) >= 2L) args[[2L]] else "data/derived/growth_parameter_recovery"
  if (length(args) > 2L) {
    stop("Usage: Rscript R/fit_growth.R [input_dir [output_dir]]", call. = FALSE)
  }
  fit_synthetic_growth(input_dir, output_dir)
}
