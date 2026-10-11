# Bounded observation likelihood for measured radius (#69)

**Status:** candidate-likelihood comparison and MyceliumSeg calibration audit
complete. No real time-resolved equivalent-radius dataset is currently eligible
for a fit, so the final likelihood and error scale must be checked against the
future target's measurement process before fitting it.

## Scope and support

The synthetic radial model has latent radius `R_true` in `[0, 45]` mm. Its
additive Normal observation model can generate values outside that interval;
the #39 prior predictive run found 35–36% outside the physical support. Those
values must not be clipped and then scored with the uncensored Normal density.
The #40 fit remains valid as a recovery check because it deliberately matches
the exact unbounded-Normal #38 generator.

## Candidate likelihoods

| Model | Observation process and likelihood | When it is appropriate | Main risk here |
|---|---|---|---|
| Censored Normal | Let `Y* ~ Normal(mu, sigma)` and record `Y = min(45, max(0, Y*))`. Interior values use the Normal density; `Y=0` uses `Phi((0-mu)/sigma)` and `Y=45` uses `1-Phi((45-mu)/sigma)`. | The measurement pipeline physically clips or saturates readings at the endpoints. | It predicts point masses at 0 and 45. With `mu` at either endpoint, about half the readings are placed exactly there. Use only if that endpoint pile-up matches the actual measurement process. |
| Truncated Normal | Condition `Y* ~ Normal(mu, sigma)` on `0 < Y* < 45`; subtract `log(Phi((45-mu)/sigma)-Phi((0-mu)/sigma))` from the interior log density. | Values outside the interval are excluded from the analyzed sample, and the selection rule is documented. | It has no endpoint mass and shifts the conditional mean near a boundary. It is not equivalent to clipping. No such exclusion rule is established for the future target. |
| Logit-normal / scaled Beta | Model an interior fraction `p=Y/45` on `(0,1)`, for example `logit(p) ~ Normal(eta, sigma_logit)` or `p ~ Beta(mu_p*phi, (1-mu_p)*phi)`. | Measurements are intrinsically bounded and error changes with the radius level. | Exact zero/full-dish values need an endpoint or censoring component. The transform/precision must be calibrated; a constant transformed-scale error is not implied by #31. |
| Lognormal with wall treatment | For positive interior radii, model `log(Y) ~ Normal(log(R_true)+b, sigma_log)` and explicitly model any full-dish cap. | Relative error is more stable than absolute millimetre error. | Lognormal alone has no 45-mm upper bound and cannot represent exact zero. It needs an explicit lower-bound state and a justified wall treatment. |

### Decision rule

There is no single likelihood to impose before the future measurement process is
known. Use a censored model only when the stored radius is actually saturated
at a physical endpoint. Use a truncated model only when an explicit QC/selection
rule drops out-of-range observations. For positive interior mask-derived radii,
compare a logit-normal (or scaled-Beta) likelihood with a positive log-scale
model plus an explicit wall treatment; decide endpoint handling from how exact
zero and full-dish values are produced. Keep posterior predictive checks and
held-out paired-mask error by radius range in the selection criteria.

Stan likelihood reminders for an additive Normal candidate:

```stan
// Censoring: observed y is the raw normal value clamped at 0 and 45.
if (y == 0)       target += normal_lcdf(0 | mu, sigma);
else if (y == 45) target += normal_lccdf(45 | mu, sigma);
else              target += normal_lpdf(y | mu, sigma);

// Truncation: use only for a sample conditioned on being interior.
target += normal_lpdf(y | mu, sigma)
          - log_diff_exp(normal_lcdf(45 | mu, sigma),
                         normal_lcdf(0 | mu, sigma));
```

For an interior logit-normal likelihood expressed as a density on `y`, include
the change-of-variables term:

```stan
real p = y / 45;
target += normal_lpdf(logit(p) | eta, sigma_logit)
          - log(45) - log(p) - log1m(p);
```

This expression does not cover exact endpoints; they need a separate
measurement-state likelihood or a resolution-based interval model.

## Calibration evidence and limits

The reproducible audit is `scripts/bounded_radius_likelihood.py`; it reads the
deduplicated, calibrated, held-out radius pairs from `data/triage/measurement_error.csv`
and writes `data/derived/bounded_radius_likelihood/segmentation_error_summary.csv`.
The population is 88 MyceliumSeg held-out masks, measured with the #28 model at
the #31 900-pixel working width.

| Error scale | Estimate on those 88 pairs |
|---|---:|
| Radius error bias, `predicted - reference` | −0.022 mm |
| Radius error SD, absolute scale | 0.766 mm |
| Relative radius error SD | 7.448% |
| `log(predicted/reference)` SD | 0.0849 |
| Reference-radius range | 2.88–35.24 mm |

The scale is not constant across the four equal-count reference-radius groups:
absolute error SDs are 0.810, 0.581, 0.370 and 1.099 mm; relative error SDs
are 14.20%, 3.40%, 1.36% and 3.51%. These are small groups (`n=22` each), so
they indicate that a single error scale needs checking rather than establishing
a stable variance function. None of the 88 references are near 0 or 45 mm, so
they do not identify endpoint behavior.

These values describe the #28 segmenter on the MyceliumSeg held-out masks. They
are provisional sensitivity scales (`0.77 mm` additive; `0.085` log-ratio), not
calibration for another image pipeline. The time-resolved datasets currently
used in the project do not provide this same validated equivalent-radius
measurement: Fricker/Boddy supplies network outcomes, while the *Podospora*
analysis uses ridge-occupancy proxies. MyceliumSeg itself has no linked dish
identity or capture chronology. Consequently, this issue does not authorize a
real-radius time-series fit or a temporal claim from MyceliumSeg.

## Reproduce and handoff

```sh
python scripts/bounded_radius_likelihood.py
```

Before any future measured-radius fit: identify its image-to-radius measurement
pipeline, obtain paired reference masks or equivalent measurement calibration
over the target radius range, inspect zero/wall observations, estimate any
radius-dependent error, and compare posterior predictive behavior for the
eligible likelihoods above. Keep the #38/#40 synthetic Normal workflow as its
own parameter-recovery benchmark.
