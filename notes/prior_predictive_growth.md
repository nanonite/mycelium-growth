# Prior predictive checks for the radial growth model (#39)

**Status:** candidate prior set implemented and simulated; use it for the
synthetic parameter-recovery step (#40), then revisit it for a specific
time-resolved dataset. It is not an empirical model fit. MyceliumSeg still has
no linked dish identities or capture times, so these trajectories are not
attributed to its images.

## Model and priors

The check follows the linear-after-lag curve with a 45 mm wall cap in
`PLAN.md` §4C. Species intercepts are partially pooled, and treatment effects
remain centered at zero so the prior does not force slower growth at 15 °C:

```text
mu_log_k                 ~ Normal(log(3), 0.55)
sigma_species_log_k      ~ HalfNormal(0.40)
alpha[species]           ~ Normal(mu_log_k, sigma_species_log_k)
sigma_dish_log_k         ~ HalfNormal(0.20)
beta_T_15                ~ Normal(0, 0.35)

mu_log_tau               ~ Normal(log(3), 0.45)
sigma_species_log_tau    ~ HalfNormal(0.35)
gamma[species]           ~ Normal(mu_log_tau, sigma_species_log_tau)
sigma_dish_log_tau       ~ HalfNormal(0.20)
delta_T_15               ~ Normal(0, 0.35)

bias_mm                 ~ Normal(0, 0.5)
sigma_seg_mm            ~ HalfNormal(2)
```

`k` and `tau` are log-linked, so they remain positive. The 15 °C rate and lag
ratios each have a 95% prior interval of about 0.50–1.99. The measurement scale
is intentionally broad: #31 reported a 7.448% relative SD for equivalent-radius
measurement, corresponding to about 3.35 mm at the 45 mm rim if that relative
error is carried to the largest radius. This constant-mm scale is a working
approximation, not a new calibration for other datasets.

The center and scale for radial rates are deliberately broad across fungal
isolates and media. Published examples report about 10–11.6 mm/day for fast
*Ganoderma* and *Pleurotus* isolates on standard malt agar, with plates reaching
the edge by days 6–7; a separate study on lignocellulosic substrates reports
3.34 mm/day for *G. lucidum*, 5.40 for *P. ostreatus*, and up to 5.27 for
*Trametes versicolor*. These are context-specific examples, not values
transported directly into this model. Rates depend on isolate, medium, and
incubation conditions ([Scientific Reports study](https://www.nature.com/articles/s41598-023-48203-4),
[lignocellulosic-substrate study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11057518/)).

## Run and outputs

From the repository root:

```sh
nix develop --command Rscript R/prior_predictive.R
```

The default run uses 10,000 prior draws, seed `20261011`, the five species /
temperature cells from #38, and a daily grid from day 0 through day 60. Outputs
are in `data/derived/growth_prior_predictive/`:

- `prior_predictive_trajectories.svg`: median latent curve and 90% prior interval
  for each condition, with the 45 mm wall shown.
- `condition_summary.csv`: rate, lag, wall-contact-time quantiles and the
  fraction of curves that have reached the wall by days 7, 14, 30 and 60.
- `trajectory_quantiles.csv`: daily 5th, 50th and 95th percentiles for latent
  radius.
- `condition_prior_draws.csv` and `fixed_prior_draws.csv`: reproducible draws.
- `prior_specification.csv` and `prior_predictive_config.csv`: priors and run
  settings.

## Findings and interpretation

All latent radii remain in `[0, 45]` mm because the generative curve applies the
physical wall cap. Across the five condition cells, the 5th–50th–95th percentiles
for `k` range from about 0.85–3.05–10.95 mm/day (GL at 15 °C) to
0.94–3.02–9.61 mm/day (the 25 °C cells). The corresponding lag percentiles are
about 0.97–3.02–9.52 days for GL at 15 °C and roughly 1.11–3.02–8.28 days for
the other cells.

The time to a 45 mm radius, `tau + 45/k`, has a median near 18.5–18.8 days.
Its 5th–95th percentile range is about 7.0–57.2 days across conditions. Between
about 3.5% and 4.9% of curves reach the wall by day 7, 30–32% by day 14,
77–80% by day 30, and 96–97% by day 60. Thus the prior spans rapid plate fill
within about a week through slower growth over several weeks. It is suitable as
a broad starting point for #40, not a calibrated forecast for any particular
fungus or substrate. The #38 generating rates and lags all lie within the
condition-specific central 90% prior intervals.

The Normal observation model is unbounded even though a measured equivalent
radius must lie between zero and the dish wall. With the broad measurement-scale
prior, about 35–36% of prior-predictive noisy observations fall outside the
physical interval, concentrated near the lag phase and at wall saturation.
Those draws are not latent-radius violations and are not clipped. Parameter
recovery in #40 can use the exact unbounded Normal generator from #38; before
fitting measured radii from a real sequence, the observation likelihood needs a
bounded or censored treatment and dataset-specific measurement calibration.
The likelihood comparison and the provisional #31 error-scale audit are recorded
in [`notes/bounded_radius_likelihood.md`](bounded_radius_likelihood.md). The
MyceliumSeg calibration is not transferable to a different image pipeline, and
no currently analyzed time-resolved dataset supplies a validated equivalent-radius
measurement. This refinement therefore does not block the exact synthetic
recovery check in #40 or authorize a real measured-radius fit.

This model is only eligible for a biological temporal interpretation when the
target dataset has linked specimens, real capture times, and a defensible mapping
from image measurements to colony radius.
