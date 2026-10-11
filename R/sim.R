#!/usr/bin/env Rscript
# Simulate a replicated dish-growth experiment with known parameters.
#
# This is a generative-data prototype for PLAN.md §4C and Chainlink #38.
# It does not infer time, dish identity, or growth parameters from MyceliumSeg.
# The default rates, lags, treatment effects, and replication are illustrative
# truths chosen for parameter-recovery work in the downstream #39 model.

# Run from the repository root:
#   nix develop --command Rscript R/sim.R
#   nix develop --command Rscript R/sim.R data/derived/dish_growth_simulation 20261010

default_growth_design <- function() {
  data.frame(
    species = c("GL", "GL", "GS", "PO", "TS"),
    temperature_C = c(15, 25, 25, 25, 25),
    stringsAsFactors = FALSE
  )
}

# Evaluate an expression under a reproducible local RNG seed, then restore the
# caller's RNG kind and state so sourcing/calling the simulator is side-effect
# free with respect to random-number generation.
with_local_seed <- function(seed, expr) {
  old_kind <- RNGkind()
  had_old_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_old_seed) {
    old_seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  }

  on.exit({
    do.call(RNGkind, as.list(old_kind))
    if (had_old_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)

  RNGkind("Mersenne-Twister", "Inversion", "Rejection")
  set.seed(seed)
  force(expr)
}

validate_simulation_inputs <- function(
  n_per_condition,
  times_days,
  design,
  species_k_25_mm_day,
  species_tau_25_days,
  beta_T_15,
  delta_T_15,
  sigma_log_k,
  sigma_log_tau,
  bias_mm,
  sigma_seg_mm,
  max_radius_mm,
  seed
) {
  if (length(n_per_condition) != 1L || is.na(n_per_condition) ||
      n_per_condition < 1 || n_per_condition != as.integer(n_per_condition)) {
    stop("n_per_condition must be one positive integer.", call. = FALSE)
  }
  if (!is.numeric(times_days) || length(times_days) < 2L ||
      any(!is.finite(times_days)) || any(times_days < 0) ||
      anyDuplicated(times_days) || is.unsorted(times_days, strictly = TRUE)) {
    stop("times_days must be finite, nonnegative, strictly increasing, and contain at least two times.",
         call. = FALSE)
  }
  if (!is.data.frame(design) || !all(c("species", "temperature_C") %in% names(design)) ||
      nrow(design) < 1L || anyNA(design[c("species", "temperature_C")])) {
    stop("design must be a non-empty data frame with species and temperature_C columns.",
         call. = FALSE)
  }
  design <- design[c("species", "temperature_C")]
  if (anyDuplicated(design)) {
    stop("design must have one row per unique species-temperature condition.", call. = FALSE)
  }
  if (!is.numeric(design$temperature_C) ||
      any(!design$temperature_C %in% c(15, 25))) {
    stop("This simulation codes the temperature contrast for 15 C versus the 25 C baseline.",
         call. = FALSE)
  }
  if (!is.numeric(species_k_25_mm_day) || is.null(names(species_k_25_mm_day)) ||
      any(!is.finite(species_k_25_mm_day)) || any(species_k_25_mm_day <= 0) ||
      anyDuplicated(names(species_k_25_mm_day))) {
    stop("species_k_25_mm_day must be a positive, named numeric vector.", call. = FALSE)
  }
  if (!is.numeric(species_tau_25_days) || is.null(names(species_tau_25_days)) ||
      any(!is.finite(species_tau_25_days)) || any(species_tau_25_days <= 0) ||
      anyDuplicated(names(species_tau_25_days))) {
    stop("species_tau_25_days must be a positive, named numeric vector.", call. = FALSE)
  }
  if (any(!design$species %in% names(species_k_25_mm_day)) ||
      any(!design$species %in% names(species_tau_25_days))) {
    stop("Each design species must have a 25 C rate and lag in the named parameter vectors.",
         call. = FALSE)
  }
  scalar_values <- c(beta_T_15, delta_T_15, sigma_log_k, sigma_log_tau,
                     bias_mm, sigma_seg_mm, max_radius_mm)
  if (any(!is.finite(scalar_values)) || sigma_log_k < 0 || sigma_log_tau < 0 ||
      sigma_seg_mm < 0 || max_radius_mm <= 0) {
    stop("Effects and scales must be finite; random-effect/noise SDs nonnegative and radius positive.",
         call. = FALSE)
  }
  if (length(seed) != 1L || is.na(seed) || !is.finite(seed) || seed < 0 ||
      seed > .Machine$integer.max || seed != as.integer(seed)) {
    stop("seed must be one nonnegative integer within R's integer range.", call. = FALSE)
  }
  invisible(TRUE)
}

simulate_dishes <- function(
  n_per_condition = 12L,
  times_days = seq(0, 30, by = 2),
  design = default_growth_design(),
  species_k_25_mm_day = c(GL = 1.65, GS = 1.80, PO = 1.25, TS = 1.45),
  species_tau_25_days = c(GL = 3.0, GS = 2.5, PO = 4.0, TS = 3.5),
  beta_T_15 = log(0.70),
  delta_T_15 = log(1.20),
  sigma_log_k = 0.12,
  sigma_log_tau = 0.15,
  bias_mm = 0,
  sigma_seg_mm = 0.5,
  max_radius_mm = 45,
  seed = 20261010L
) {
  validate_simulation_inputs(
    n_per_condition, times_days, design, species_k_25_mm_day,
    species_tau_25_days, beta_T_15, delta_T_15, sigma_log_k,
    sigma_log_tau, bias_mm, sigma_seg_mm, max_radius_mm, seed
  )

  design <- unique(design[c("species", "temperature_C")])
  design$species <- as.character(design$species)
  design <- design[order(design$species, design$temperature_C), , drop = FALSE]
  rownames(design) <- NULL
  times_days <- as.numeric(times_days)
  n_per_condition <- as.integer(n_per_condition)
  seed <- as.integer(seed)

  with_local_seed(seed, {
    cells <- do.call(rbind, lapply(seq_len(nrow(design)), function(i) {
      data.frame(
        species = rep(design$species[i], n_per_condition),
        temperature_C = rep(design$temperature_C[i], n_per_condition),
        replicate = seq_len(n_per_condition),
        stringsAsFactors = FALSE
      )
    }))
    rownames(cells) <- NULL
    cells$dish_id <- sprintf("D%03d", seq_len(nrow(cells)))

    is_15C <- as.numeric(cells$temperature_C == 15)
    z_log_k <- rnorm(nrow(cells))
    z_log_tau <- rnorm(nrow(cells))
    alpha <- log(species_k_25_mm_day[cells$species])
    gamma <- log(species_tau_25_days[cells$species])

    dishes <- data.frame(
      dish_id = cells$dish_id,
      species = cells$species,
      temperature_C = cells$temperature_C,
      replicate = cells$replicate,
      z_log_k = z_log_k,
      z_log_tau = z_log_tau,
      k_mm_day = exp(alpha + beta_T_15 * is_15C + z_log_k * sigma_log_k),
      tau_days = exp(gamma + delta_T_15 * is_15C + z_log_tau * sigma_log_tau),
      stringsAsFactors = FALSE
    )

    observations <- do.call(rbind, lapply(seq_len(nrow(dishes)), function(i) {
      dish <- dishes[i, , drop = FALSE]
      r_true <- pmin(
        max_radius_mm,
        dish$k_mm_day * pmax(0, times_days - dish$tau_days)
      )
      measurement_error <- rnorm(length(times_days), mean = 0, sd = sigma_seg_mm)
      r_obs <- r_true + bias_mm + measurement_error
      data.frame(
        dish_id = dish$dish_id,
        species = dish$species,
        temperature_C = dish$temperature_C,
        replicate = dish$replicate,
        time_days = times_days,
        R_true_mm = as.numeric(r_true),
        measurement_error_mm = measurement_error,
        R_obs_mm = as.numeric(r_obs),
        observed_outside_physical_range = (r_obs < 0 | r_obs > max_radius_mm),
        stringsAsFactors = FALSE
      )
    }))
    rownames(observations) <- NULL

    fixed_parameters <- data.frame(
      parameter = c(
        paste0("alpha_log_k[", names(species_k_25_mm_day), "]"),
        paste0("gamma_log_tau[", names(species_tau_25_days), "]"),
        "beta_T_15", "delta_T_15", "sigma_log_k", "sigma_log_tau",
        "bias_mm", "sigma_seg_mm", "max_radius_mm"
      ),
      value = c(
        log(species_k_25_mm_day), log(species_tau_25_days), beta_T_15,
        delta_T_15, sigma_log_k, sigma_log_tau, bias_mm, sigma_seg_mm,
        max_radius_mm
      ),
      stringsAsFactors = FALSE
    )
    fixed_parameters$units <- c(
      rep("log(mm/day)", length(species_k_25_mm_day)),
      rep("log(days)", length(species_tau_25_days)),
      "log rate ratio", "log lag ratio", "SD on log rate", "SD on log lag",
      "mm", "mm", "mm"
    )

    config <- data.frame(
      parameter = c("seed", "n_per_condition", "n_conditions", "n_dishes",
                    "n_timepoints_per_dish", "time_points_days", "design"),
      value = c(
        as.character(seed), as.character(n_per_condition),
        as.character(nrow(design)), as.character(nrow(dishes)),
        as.character(length(times_days)), paste(times_days, collapse = ","),
        paste(paste0(design$species, "@", design$temperature_C, "C"), collapse = ";")
      ),
      stringsAsFactors = FALSE
    )

    list(
      dishes = dishes,
      observations = observations,
      fixed_parameters = fixed_parameters,
      design = design,
      config = config
    )
  })
}

write_simulation <- function(simulation, output_dir) {
  required <- c("dishes", "observations", "fixed_parameters", "design", "config")
  if (!is.list(simulation) || !all(required %in% names(simulation))) {
    stop("simulation must be the list returned by simulate_dishes().", call. = FALSE)
  }
  if (length(output_dir) != 1L || is.na(output_dir) || !nzchar(output_dir)) {
    stop("output_dir must be one non-empty path.", call. = FALSE)
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(simulation$dishes, file.path(output_dir, "dish_parameters.csv"),
            row.names = FALSE, na = "")
  write.csv(simulation$observations, file.path(output_dir, "observations.csv"),
            row.names = FALSE, na = "")
  write.csv(simulation$fixed_parameters, file.path(output_dir, "known_parameters.csv"),
            row.names = FALSE, na = "")
  write.csv(simulation$design, file.path(output_dir, "condition_design.csv"),
            row.names = FALSE, na = "")
  write.csv(simulation$config, file.path(output_dir, "simulation_config.csv"),
            row.names = FALSE, na = "")
  invisible(output_dir)
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  output_dir <- if (length(args) >= 1L) args[[1L]] else
    "data/derived/dish_growth_simulation"
  seed <- if (length(args) >= 2L) as.numeric(args[[2L]]) else 20261010L
  if (length(args) > 2L || is.na(seed) || !is.finite(seed) || seed != as.integer(seed)) {
    stop("Usage: Rscript R/sim.R [output_dir [integer_seed]]", call. = FALSE)
  }

  simulation <- simulate_dishes(seed = as.integer(seed))
  write_simulation(simulation, output_dir)
  cat(sprintf(
    "Wrote %d dishes and %d observations to %s (seed %d).\n",
    nrow(simulation$dishes), nrow(simulation$observations), output_dir, seed
  ))
}
