# Known-parameter dish-growth simulation (#38)

`R/sim.R` generates a reproducible synthetic dish experiment for the parameter-
recovery model in `PLAN.md` §4C. Its default output is
`data/derived/dish_growth_simulation/`. These rows are simulated; they are not
measurements or inferred growth parameters from MyceliumSeg. MyceliumSeg does
not provide the dish identities or capture times needed for such a fit.

## Generative model

For dish `d` at day `t`, the latent colony radius is

```text
R_true[d,t] = min(45 mm, k[d] * max(0, t - tau[d]))
log k[d]    = alpha[species[d]] + beta_T_15 * I(temperature[d] = 15 C)
               + z_k[d] * sigma_log_k
log tau[d]  = gamma[species[d]] + delta_T_15 * I(temperature[d] = 15 C)
               + z_tau[d] * sigma_log_tau
R_obs[d,t]  ~ Normal(R_true[d,t] + bias_mm, sigma_seg_mm)
```

The dish effects `z_k` and `z_tau` are independent standard Normal draws. The
saved `dish_parameters.csv` contains the realized effects and the resulting true
`k` and `tau` for every dish. The 45 mm cap is the dish-wall radius in the plan.
The observation model is deliberately uncensored and unclipped, as written in
§4C; the observed-radius flag identifies noisy values outside `[0, 45]` for
diagnostics. `R_true_mm` remains physically bounded.

## Default synthetic scenario

| Quantity | Default | Interpretation |
|---|---:|---|
| Dishes per condition | 12 | Independent simulated dishes in each design cell |
| Conditions | GL at 15/25 C; GS, PO, TS at 25 C | Hypothetical replicated design; GL supplies the within-species temperature contrast |
| Observation days | 0, 2, …, 30 | 16 repeated observations per dish |
| 25 C rates, `k` | GL 1.65; GS 1.80; PO 1.25; TS 1.45 mm/day | Illustrative fixed species baselines |
| 25 C lags, `tau` | GL 3.0; GS 2.5; PO 4.0; TS 3.5 days | Illustrative fixed species baselines |
| 15 C rate ratio | `exp(beta_T_15) = 0.70` | Illustrative, known simulation effect |
| 15 C lag ratio | `exp(delta_T_15) = 1.20` | Illustrative, known simulation effect |
| Between-dish SDs | 0.12 for log-rate; 0.15 for log-lag | Random-effect scales |
| Segmentation error | bias 0 mm; SD 0.5 mm | Illustrative Normal observation error |
| Seed | 20261010 | Fixed default; local RNG state is restored after simulation |

The defaults intentionally represent a recovery exercise, not estimates from a
paper or evidence about fungal temperature response. The configurable design
and parameter arguments let #39 use other known truths for stress checks.

## Files and commands

From the repository root, generate the defaults with:

```sh
nix develop --command Rscript R/sim.R
```

An optional first argument selects the output directory; the second sets the
integer seed:

```sh
nix develop --command Rscript R/sim.R data/derived/dish_growth_simulation 1234
```

The generated directory contains:

- `observations.csv`: one row per dish and day, with latent and observed radius,
  the realized measurement error, design labels, and an out-of-range flag.
- `dish_parameters.csv`: dish-level true rates, lags, and standardized random
  effects. This file is for recovery checks; it would not be supplied as model
  input in a real-data analysis.
- `known_parameters.csv`: fixed generating values on the log scale used by the
  model, random-effect scales, observation error, and radius cap.
- `condition_design.csv`: species-temperature cells used for the simulation.
- `simulation_config.csv`: seed, replication, observation days, and design summary.

`simulate_dishes()` and `write_simulation()` can also be sourced from another R
script. The simulator uses base R only. The downstream model in #39 should fit
`observations.csv`, compare its posterior to `known_parameters.csv` and
`dish_parameters.csv`, and report parameter-recovery diagnostics separately
from any empirical analysis.
