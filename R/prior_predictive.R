#!/usr/bin/env Rscript
# Prior predictive simulation for the latent lag/extension/rim model in PLAN §4C.
#
# Run from the repository root:
#   nix develop --command Rscript R/prior_predictive.R
#   nix develop --command Rscript R/prior_predictive.R data/derived/growth_prior 20261011 10000
#
# This checks the implications of candidate priors. It does not fit data or
# treat MyceliumSeg as a time series.

source("R/sim.R")

prior_specification <- function() {
  data.frame(
    parameter = c(
      "mu_log_k", "sigma_species_log_k", "sigma_dish_log_k", "beta_T_15",
      "mu_log_tau", "sigma_species_log_tau", "sigma_dish_log_tau", "delta_T_15",
      "bias_mm", "sigma_seg_mm"
    ),
    distribution = c(
      "Normal", "HalfNormal", "HalfNormal", "Normal",
      "Normal", "HalfNormal", "HalfNormal", "Normal",
      "Normal", "HalfNormal"
    ),
    parameters = c(
      "mean=log(3), sd=0.55", "scale=0.40", "scale=0.20", "mean=0, sd=0.35",
      "mean=log(3), sd=0.45", "scale=0.35", "scale=0.20", "mean=0, sd=0.35",
      "mean=0 mm, sd=0.5 mm", "scale=2 mm"
    ),
    units = c(
      "log(mm/day)", "log-rate SD", "log-rate SD", "log rate ratio",
      "log(days)", "log-lag SD", "log-lag SD", "log lag ratio",
      "mm", "mm"
    ),
    rationale = c(
      "Population-level median radial rate of 3 mm/day; broad across isolates and media.",
      "Partial pooling of species-specific rates.",
      "Dish-to-dish rate variation.",
      "Allows either sign; 95% prior rate-ratio interval is about 0.50 to 1.99.",
      "Positive lags on the log scale; population median at 3 days.",
      "Partial pooling of species-specific lags.",
      "Dish-to-dish lag variation.",
      "Allows either sign; 95% prior lag-ratio interval is about 0.50 to 1.99.",
      "Weakly regularizing measurement-bias prior around zero.",
      "Broad constant-mm error scale; scale chosen to include the 3.35 mm equivalent-radius SD implied by 7.45% relative SD at the 45 mm rim."
    ),
    stringsAsFactors = FALSE
  )
}

simulate_prior_predictive <- function(
  n_draws = 10000L,
  time_days = seq(0, 60, by = 1),
  design = default_growth_design(),
  max_radius_mm = 45,
  seed = 20261011L
) {
  if (length(n_draws) != 1L || is.na(n_draws) || n_draws < 1 ||
      n_draws != as.integer(n_draws)) {
    stop("n_draws must be one positive integer.", call. = FALSE)
  }
  if (!is.numeric(time_days) || length(time_days) < 2L ||
      any(!is.finite(time_days)) || any(time_days < 0) ||
      anyDuplicated(time_days) || is.unsorted(time_days, strictly = TRUE)) {
    stop("time_days must be finite, nonnegative, strictly increasing, and contain at least two times.",
         call. = FALSE)
  }
  if (!is.numeric(max_radius_mm) || length(max_radius_mm) != 1L ||
      !is.finite(max_radius_mm) || max_radius_mm <= 0) {
    stop("max_radius_mm must be one finite positive number.", call. = FALSE)
  }
  if (!is.data.frame(design) ||
      !all(c("species", "temperature_C") %in% names(design)) ||
      nrow(design) < 1L || anyNA(design[c("species", "temperature_C")])) {
    stop("design must be a non-empty data frame with species and temperature_C columns.",
         call. = FALSE)
  }
  design <- design[c("species", "temperature_C")]
  design$species <- as.character(design$species)
  species_names <- unique(design$species)
  validate_simulation_inputs(
    n_per_condition = 1L,
    times_days = c(0, 1),
    design = design,
    species_k_25_mm_day = setNames(rep(1, length(species_names)), species_names),
    species_tau_25_days = setNames(rep(1, length(species_names)), species_names),
    beta_T_15 = 0, delta_T_15 = 0, sigma_log_k = 0,
    sigma_log_tau = 0, bias_mm = 0, sigma_seg_mm = 0,
    max_radius_mm = max_radius_mm, seed = seed
  )

  n_draws <- as.integer(n_draws)
  time_days <- as.numeric(time_days)
  design <- unique(design[c("species", "temperature_C")])
  design$species <- as.character(design$species)
  design <- design[order(design$species, design$temperature_C), , drop = FALSE]
  rownames(design) <- NULL
  species <- sort(unique(design$species))
  species_index <- match(design$species, species)
  is_15C <- as.numeric(design$temperature_C == 15)

  with_local_seed(seed, {
    # Hierarchical priors on the log-rate and log-lag parameters.
    mu_log_k <- rnorm(n_draws, mean = log(3), sd = 0.55)
    sigma_species_log_k <- abs(rnorm(n_draws, mean = 0, sd = 0.40))
    sigma_dish_log_k <- abs(rnorm(n_draws, mean = 0, sd = 0.20))
    alpha <- matrix(mu_log_k, nrow = n_draws, ncol = length(species)) +
      matrix(rnorm(n_draws * length(species)), nrow = n_draws) * sigma_species_log_k
    beta_T_15 <- rnorm(n_draws, mean = 0, sd = 0.35)

    mu_log_tau <- rnorm(n_draws, mean = log(3), sd = 0.45)
    sigma_species_log_tau <- abs(rnorm(n_draws, mean = 0, sd = 0.35))
    sigma_dish_log_tau <- abs(rnorm(n_draws, mean = 0, sd = 0.20))
    gamma <- matrix(mu_log_tau, nrow = n_draws, ncol = length(species)) +
      matrix(rnorm(n_draws * length(species)), nrow = n_draws) * sigma_species_log_tau
    delta_T_15 <- rnorm(n_draws, mean = 0, sd = 0.35)

    bias_mm <- rnorm(n_draws, mean = 0, sd = 0.5)
    sigma_seg_mm <- abs(rnorm(n_draws, mean = 0, sd = 2))

    z_k <- matrix(rnorm(n_draws * nrow(design)), nrow = n_draws)
    z_tau <- matrix(rnorm(n_draws * nrow(design)), nrow = n_draws)
    k <- exp(alpha[, species_index, drop = FALSE] +
               beta_T_15 * rep(is_15C, each = n_draws) +
               z_k * sigma_dish_log_k)
    tau <- exp(gamma[, species_index, drop = FALSE] +
                 delta_T_15 * rep(is_15C, each = n_draws) +
                 z_tau * sigma_dish_log_tau)
    wall_hit_days <- tau + max_radius_mm / k

    condition_draws <- do.call(rbind, lapply(seq_len(nrow(design)), function(j) {
      data.frame(
        draw_id = seq_len(n_draws),
        species = design$species[j],
        temperature_C = design$temperature_C[j],
        k_mm_day = k[, j],
        tau_days = tau[, j],
        time_to_wall_days = wall_hit_days[, j],
        stringsAsFactors = FALSE
      )
    }))

    fixed_draws <- data.frame(
      draw_id = seq_len(n_draws),
      mu_log_k = mu_log_k,
      sigma_species_log_k = sigma_species_log_k,
      sigma_dish_log_k = sigma_dish_log_k,
      beta_T_15 = beta_T_15,
      mu_log_tau = mu_log_tau,
      sigma_species_log_tau = sigma_species_log_tau,
      sigma_dish_log_tau = sigma_dish_log_tau,
      delta_T_15 = delta_T_15,
      bias_mm = bias_mm,
      sigma_seg_mm = sigma_seg_mm
    )

    curve_rows <- vector("list", nrow(design))
    summary_rows <- vector("list", nrow(design))
    observed_outside <- numeric(nrow(design))
    for (j in seq_len(nrow(design))) {
      r_true <- vapply(time_days, function(day) {
        pmin(max_radius_mm, k[, j] * pmax(0, day - tau[, j]))
      }, numeric(n_draws))
      r_obs <- r_true + bias_mm +
        matrix(rnorm(n_draws * length(time_days)), nrow = n_draws) * sigma_seg_mm
      observed_outside[j] <- mean(r_obs < 0 | r_obs > max_radius_mm)

      quantiles <- t(apply(r_true, 2, quantile,
                           probs = c(0.05, 0.50, 0.95), names = FALSE))
      curve_rows[[j]] <- data.frame(
        species = design$species[j],
        temperature_C = design$temperature_C[j],
        time_days = time_days,
        R_true_q05_mm = quantiles[, 1],
        R_true_q50_mm = quantiles[, 2],
        R_true_q95_mm = quantiles[, 3],
        stringsAsFactors = FALSE
      )

      cd <- condition_draws[condition_draws$species == design$species[j] &
                              condition_draws$temperature_C == design$temperature_C[j], ]
      summary_rows[[j]] <- data.frame(
        species = design$species[j],
        temperature_C = design$temperature_C[j],
        k_q05_mm_day = unname(quantile(cd$k_mm_day, 0.05)),
        k_q50_mm_day = unname(quantile(cd$k_mm_day, 0.50)),
        k_q95_mm_day = unname(quantile(cd$k_mm_day, 0.95)),
        tau_q05_days = unname(quantile(cd$tau_days, 0.05)),
        tau_q50_days = unname(quantile(cd$tau_days, 0.50)),
        tau_q95_days = unname(quantile(cd$tau_days, 0.95)),
        wall_q05_days = unname(quantile(cd$time_to_wall_days, 0.05)),
        wall_q50_days = unname(quantile(cd$time_to_wall_days, 0.50)),
        wall_q95_days = unname(quantile(cd$time_to_wall_days, 0.95)),
        p_wall_by_7_days = mean(cd$time_to_wall_days <= 7),
        p_wall_by_14_days = mean(cd$time_to_wall_days <= 14),
        p_wall_by_30_days = mean(cd$time_to_wall_days <= 30),
        p_wall_by_60_days = mean(cd$time_to_wall_days <= 60),
        observed_outside_physical_fraction = observed_outside[j],
        true_radius_min_mm = min(r_true),
        true_radius_max_mm = max(r_true),
        stringsAsFactors = FALSE
      )
    }

    list(
      design = design,
      time_days = time_days,
      max_radius_mm = max_radius_mm,
      config = data.frame(
        parameter = c("seed", "n_draws", "time_start_days", "time_end_days",
                      "time_step_days", "max_radius_mm", "n_conditions"),
        value = c(as.character(seed), as.character(n_draws),
                  as.character(min(time_days)), as.character(max(time_days)),
                  as.character(if (length(unique(diff(time_days))) == 1L) diff(time_days)[1L] else NA),
                  as.character(max_radius_mm), as.character(nrow(design))),
        stringsAsFactors = FALSE
      ),
      prior_specification = prior_specification(),
      fixed_draws = fixed_draws,
      condition_draws = condition_draws,
      trajectory_quantiles = do.call(rbind, curve_rows),
      condition_summary = do.call(rbind, summary_rows)
    )
  })
}

write_prior_predictive <- function(result, output_dir) {
  required <- c("config", "prior_specification", "fixed_draws", "condition_draws",
                "trajectory_quantiles", "condition_summary")
  if (!is.list(result) || !all(required %in% names(result))) {
    stop("result must be the list returned by simulate_prior_predictive().", call. = FALSE)
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(result$config, file.path(output_dir, "prior_predictive_config.csv"),
            row.names = FALSE)
  write.csv(result$prior_specification,
            file.path(output_dir, "prior_specification.csv"), row.names = FALSE)
  write.csv(result$fixed_draws,
            file.path(output_dir, "fixed_prior_draws.csv"), row.names = FALSE)
  write.csv(result$condition_draws,
            file.path(output_dir, "condition_prior_draws.csv"), row.names = FALSE)
  write.csv(result$trajectory_quantiles,
            file.path(output_dir, "trajectory_quantiles.csv"), row.names = FALSE)
  write.csv(result$condition_summary,
            file.path(output_dir, "condition_summary.csv"), row.names = FALSE)

  svg(file.path(output_dir, "prior_predictive_trajectories.svg"), width = 10, height = 7)
  old_par <- par(mfrow = c(2, 3), mar = c(4, 4, 2.5, 1))
  on.exit({par(old_par); dev.off()}, add = TRUE)
  for (i in seq_len(nrow(result$design))) {
    row <- result$design[i, ]
    d <- result$trajectory_quantiles[
      result$trajectory_quantiles$species == row$species &
        result$trajectory_quantiles$temperature_C == row$temperature_C, ]
    plot(d$time_days, d$R_true_q50_mm, type = "n",
         xlim = range(result$time_days), ylim = c(0, result$max_radius_mm),
         xlab = "Incubation day", ylab = "Latent radius (mm)",
         main = paste0(row$species, " at ", row$temperature_C, " C"))
    polygon(c(d$time_days, rev(d$time_days)),
            c(d$R_true_q05_mm, rev(d$R_true_q95_mm)),
            col = grDevices::adjustcolor("#277DA1", alpha.f = 0.20), border = NA)
    lines(d$time_days, d$R_true_q50_mm, col = "#277DA1", lwd = 2)
    abline(h = result$max_radius_mm, lty = 2, col = "#C44536")
    grid(col = "grey88")
  }
  plot.new()
  legend("center", legend = c("Median prior curve", "90% prior interval", "45 mm wall"),
         lty = c(1, NA, 2), lwd = c(2, NA, 1),
         pch = c(NA, 15, NA), pt.cex = 2,
         col = c("#277DA1", grDevices::adjustcolor("#277DA1", alpha.f = 0.20), "#C44536"),
         bty = "n")
  invisible(output_dir)
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  output_dir <- if (length(args) >= 1L) args[[1L]] else
    "data/derived/growth_prior_predictive"
  seed <- if (length(args) >= 2L) as.numeric(args[[2L]]) else 20261011L
  n_draws <- if (length(args) >= 3L) as.numeric(args[[3L]]) else 10000L
  if (length(args) > 3L || is.na(seed) || !is.finite(seed) ||
      seed != as.integer(seed) || is.na(n_draws) || !is.finite(n_draws) ||
      n_draws != as.integer(n_draws)) {
    stop("Usage: Rscript R/prior_predictive.R [output_dir [integer_seed [integer_draws]]]",
         call. = FALSE)
  }

  result <- simulate_prior_predictive(n_draws = as.integer(n_draws), seed = as.integer(seed))
  write_prior_predictive(result, output_dir)
  cat(sprintf("Wrote prior predictive summaries for %d draws across %d condition cells to %s.\n",
              n_draws, nrow(result$design), output_dir))
}
