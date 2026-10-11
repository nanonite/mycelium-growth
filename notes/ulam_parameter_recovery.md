# Synthetic growth parameter recovery (#40)

**Status:** completed on the #38 synthetic dataset. This is a model-recovery
exercise only; none of the fitted parameters are estimates from MyceliumSeg.

## Fit

`R/fit_growth.R` fits all 960 observations from 60 simulated dishes (16 time
points per dish) using the priors from #39. It uses the same lag → linear radial
extension → 45 mm wall cap as the #38 generator, with a Normal observation
model. The run used four chains, 1,000 warmup and 1,000 sampling iterations per
chain, seed `20261012`, `adapt_delta = 0.99`, and maximum tree depth 12.

The simulator specifies four species baselines directly rather than generating
them from a species population distribution. Therefore, the species rate/lag
baselines, temperature ratios, dish-level SDs, bias and observation SD have
known truths; the population means and between-species SDs do not. The latter
hyperparameters are reported but not scored as recovered truths.

`rethinking` 2.21 emits legacy Stan array syntax. The pinned CmdStan 2.35 parser
rejects that syntax, so this fit uses the compatible RStan 2.32 backend. The
runner temporarily sets Rethinking's backend switch and a local Makevars entry
for `/usr/bin/g++`, then restores the Rethinking and compiler environment on
exit. The compiled model and fit are saved so the recovery summaries can be
reviewed without recompilation.

## Recovery results

All 14 truth-bearing fixed parameters have 95% posterior intervals containing
their generating values. The individual dish-level intervals contain the true
rate for 57/60 dishes (95.0%) and the true lag for 56/60 dishes (93.3%). Median
posterior predictions recover latent radii with 0.179 mm RMSE; their 95%
intervals cover 95.3% of the 960 simulated latent radii.

| Parameter | Generating truth | Posterior median | 95% interval |
|---|---:|---:|---:|
| GL rate at 25 C (mm/day) | 1.650 | 1.719 | 1.610–1.838 |
| GS rate at 25 C (mm/day) | 1.800 | 1.731 | 1.623–1.851 |
| PO rate at 25 C (mm/day) | 1.250 | 1.239 | 1.161–1.327 |
| TS rate at 25 C (mm/day) | 1.450 | 1.400 | 1.307–1.496 |
| GL lag at 25 C (days) | 3.000 | 3.007 | 2.757–3.289 |
| GS lag at 25 C (days) | 2.500 | 2.423 | 2.214–2.663 |
| PO lag at 25 C (days) | 4.000 | 3.799 | 3.464–4.161 |
| TS lag at 25 C (days) | 3.500 | 3.812 | 3.485–4.135 |
| 15 C / 25 C rate ratio | 0.700 | 0.661 | 0.604–0.728 |
| 15 C / 25 C lag ratio | 1.200 | 1.155 | 1.016–1.302 |
| Dish SD, log rate | 0.120 | 0.119 | 0.099–0.147 |
| Dish SD, log lag | 0.150 | 0.139 | 0.111–0.177 |
| Observation bias (mm) | 0.000 | -0.051 | -0.130–0.031 |
| Observation SD (mm) | 0.500 | 0.519 | 0.495–0.546 |

## Sampling diagnostics and limits

There were no divergent transitions. Minimum bulk ESS was 507 and minimum tail
ESS was 780. Maximum R-hat was 1.0112, with four parameters slightly above the
conventional 1.01 target; one post-warmup transition in chain 3 reached the
configured tree-depth limit of 12. This is adequate evidence of recovery for
the synthetic workflow, with that small convergence caveat recorded rather than
hidden.

The fit deliberately matches the uncensored Normal generator. It does not
resolve the bounded-measurement-likelihood refinement tracked by #69 for any
future real radius data. No dish, time, species or growth claims are inferred
from MyceliumSeg.

## Artifacts

- `R/fit_growth.R`: reproducible fit and recovery-summary workflow.
- `data/derived/growth_parameter_recovery/ulam_fit.rds`: fitted `ulam` object.
- `data/derived/growth_parameter_recovery/posterior_draws.rds`: posterior draws.
- `data/derived/growth_parameter_recovery/ulam_model.stan`: generated Stan model.
- `data/derived/growth_parameter_recovery/fixed_parameter_recovery.csv` and
  `dish_parameter_recovery.csv`: fixed- and dish-level truth comparisons.
- `data/derived/growth_parameter_recovery/mcmc_diagnostics.csv` and
  `sampler_diagnostics.csv`: parameter and transition diagnostics.
- `data/derived/growth_parameter_recovery/condition_curve_recovery.svg`: cell
  means and posterior recovery curves.

Run from the repository root with `nix develop --command Rscript R/fit_growth.R`.
