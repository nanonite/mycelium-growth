# MyceliumSeg modeling index

This index connects the converted biology references to the measurements that the
MyceliumSeg release can plausibly support. It is a planning document, not evidence
that the current images identify a mechanism. A claim is treated as identified only
when the image resolution, metadata, timing, and study design support it.

## 1. Reference corpus

The complete conversion inventory is in [`conversion-report.md`](conversion-report.md).
These references are most relevant to the growth-model question:

| Reference | Modeling relevance |
|---|---|
| [`Fungal biology Deacon.md`](Fungal%20biology%20Deacon.md) | Hyphal tip growth, repeated branching, mycelial networks, nutrient uptake, and the distinction between hyphal structure and nuclear status. See the converted pages corresponding to Fungal Biology pp. 12–14 and 48–67. |
| [`SHA256E-s4506228...md`](SHA256E-s4506228--2efb366faf8a39061319977d13fe1d3c3d377396ef3d336e6b9835133ae537e9.md) | Direct treatment of fungal morphology, image analysis, growth measurement, and empirical, mechanistic, deterministic, and stochastic growth models. The key discussion is in converted pages 28–33. |
| [`SHA256E-s7299972...md`](SHA256E-s7299972--85474b8c88b864cf57285b803f3ba2080648aa18a14a6a3f4df19c7abea23ae1.lc.md) | Hyphal tip growth and developmental control. It describes continuous tip growth and septation, environmental control of developmental form, and high-resolution pulsatile elongation. See converted pages 16–17, 42–48. |
| [`SHA256E-s10157412...md`](SHA256E-s10157412--19960d26f74d06c79370420afbd477ad269e14e2b320fe393ef58c6aeb74fac9.md) | Mycorrhizal and fungal biology background; useful for biological context and candidate mechanisms, not a direct measurement protocol for MyceliumSeg. |
| [`SHA256E-s24451673...md`](SHA256E-s24451673--f697d9d9badc97d8387b8280ac5a66359374198d6e449e68466a65a9080cdbd8.md) | Fungal genomics and development context; useful for later composition with molecular or developmental datasets. |

The pathogenic microbiology, bioinformatics, biochemistry, and mycoremediation
references provide supporting context. They should not be treated as if they were
measurements from the MyceliumSeg experiment.

## 2. Biological priors that are useful, with limits

### Tip growth and branching

The references describe filamentous fungi as hyphae that extend at their apices,
with branches forming behind the tips and the resulting network called a mycelium.
The Mycoremediation reference also describes image analysis based on hyphal length,
branching frequency, colony dimensions, projected area, perimeter, and texture.
These are reasonable candidate observables or latent variables for a future model.

The same reference defines a hyphal growth unit as total hyphal length divided by
the number of tips, while noting that not every visible tip is necessarily viable.
That distinction matters here: a protrusion in a low-resolution mask can be an image
boundary event without being a living hyphal tip. The project should therefore use
“tip-like boundary event” until resolution and persistence checks validate the
biological interpretation.

### Pulsatile microscopic growth is a separate scale

The fungal-development reference reports that high-magnification video can reveal
pulsatile elongation when measured at one-to-five-second intervals. That observation
is valuable as a biological prior, but it cannot be imported into MyceliumSeg as a
claim about individual hyphae. The current images do not resolve nuclei, vesicles,
septa, or reliably track individual tips, and their capture interval is not known.

### Environmental controls are confounders without metadata

The references identify medium composition, carbon and nitrogen supply, oxygen,
pH, temperature, water availability, and physical culture conditions as influences
on fungal growth or morphology. Those mechanisms motivate future interventions and
covariates. They do not make a temperature or medium effect identifiable in
MyceliumSeg, where the released files do not provide the required dish tracks,
capture times, or an independent separation of species and medium.

## 3. What the current data can identify

The following table uses the project triage results together with the reference
priors. “Observable” means measurable from the image or a validated mask; it does
not mean that the quantity is a direct biological mechanism.

| Target | Required evidence | MyceliumSeg status | Defensible near-term statement |
|---|---|---|---|
| Colony mask area and area fraction | Segmentation mask and dish region | Available after segmentation; area is more stable than fine boundary features | Describe colony occupancy and its uncertainty |
| Equivalent radius `sqrt(A / pi)` | Area plus dish-scale calibration | Available with the per-condition calibration already recorded in `data/triage/` | Describe an area-derived size index, not a hyphal-front speed |
| Perimeter, roughness, compactness, radial anisotropy | Reliable boundary and measurement-error calibration | Candidate features; boundary IoU is weaker than area accuracy | Compare observable boundary shape with uncertainty |
| Frame-to-frame change | Correct image order and repeated observations | Not established for the released files | Use only where sequence provenance is demonstrated; otherwise analyze cross-sectional morphology |
| Tip-like boundary events | Adequate resolution, stable masks, persistence, and false-event controls | Exploratory only | Report event candidates, not biological tip formation |
| Hyphal length, branch count, hyphal growth unit, nuclei, septa | Resolving and tracking individual hyphae or a validated mapping | Not identified at the current image resolution | Requires a higher-resolution or microscopy dataset |
| Growth rate in mm/day or a lag time | Dish identity, capture interval, ordered observations, scale | Not identified | Use frame units only when frame order is known; do not convert to time without an interval |
| Temperature effect | Controlled temperature contrast, dish identity, time, replication | GL has 15 °C and 25 °C labels, but the triage gate is still unresolved | Treat as associational condition contrast until metadata and design are established |
| Medium effect | Independent variation of medium apart from species | Species and medium are aliased in the release | Do not estimate a medium effect from this dataset |

## 4. Near-term estimands

The primary MyceliumSeg estimand should be an observable change, for example:

```text
Delta log(area_fraction) per ordered frame
```

or the corresponding fractional change in area, equivalent radius, or a validated
shape feature. The unit is per frame unless a capture interval is recovered. If
dish identity and frame order cannot be recovered, the estimand changes to a
cross-sectional distribution of morphology by the known condition labels. That is
an associational summary, not a growth curve.

Every reported feature should include:

1. the image and mask provenance;
2. the unit and calibration used;
3. segmentation and feature-measurement bias and uncertainty estimated on labeled
   masks;
4. the split rule, with duplicate images and related images kept in the same split;
5. uncertainty intervals and posterior predictive checks where a model is fitted.

## 5. Model composition for this dataset

The useful composition is deliberately layered:

1. **Observation layer:** image → predicted mask, with segmentation uncertainty.
2. **Geometry layer:** mask → area, equivalent radius, perimeter, roughness, and
   other shape summaries, with a measurement-error model.
3. **Cross-sectional layer:** morphology distributions by species or known condition,
   using partial pooling and avoiding the aliased medium effect.
4. **Frame-index layer:** for any validated ordered subset, model changes per frame
   with missing-frame, duplicate, and process-error checks.
5. **Event layer:** add tip-like boundary events only after testing persistence,
   resolution dependence, and false-event rates.
6. **Mechanistic layer:** reserve tip, branching, nutrient, oxygen, and temperature
   parameters for datasets that supply the observations needed to identify them.

The reference literature supports comparing empirical, mechanistic, deterministic,
and stochastic model families. It does not justify selecting a mechanistic model
from the current images alone. A first generative model should therefore simulate
the observable feature and its measurement error, recover known parameters on
simulated data, and make predictive claims on held-out images or sequences.

## 6. Recommended statistical workflow

1. Pre-register the feature, unit, inclusion rule, and biological interpretation.
2. Fit the segmentation and geometry measurement model before fitting dynamics.
3. Simulate area or radius trajectories with measurement noise, duplicates, missing
   frames, and false boundary events.
4. Compare simple descriptive growth forms against saturation or state-space forms
   using predictive checks rather than relying on a mechanistic label.
5. Hold out entire dishes or provenance blocks whenever those units are available;
   frame-level random splits would leak information.
6. Treat any biological interpretation as a separate validation step requiring
   higher-resolution images, known timing, or an independent dataset.

This gives MyceliumSeg a useful role in a composition of models: it can contribute
segmentation, image-scale morphology, and measurement-error calibration even if the
time-resolved mechanistic growth component must come from another dataset.

## 7. Immediate follow-up

- Complete the formal Gate C decision in `notes/gate.md` and issue #26.
- Run inference on selected unlabeled shards only after recording provenance and
  duplicate handling.
- Validate area and radius first; treat boundary-derived features as secondary.
- Use the time-resolved Fricker/Boddy or Podospora datasets to test dynamic and
  intervention models if MyceliumSeg metadata cannot be recovered.

The converted source files preserve page boundaries but not figures, equation layout,
or text that exists only inside an image. Those limitations are recorded in the
per-file front matter and [`conversion-report.md`](conversion-report.md).
