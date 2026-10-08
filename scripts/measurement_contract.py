#!/usr/bin/env python3
"""The measurement contract: estimands, units and reporting guards (Chainlink #60).

PLAN.md Phase 2 asks for the estimand specification *before* anything is
inferred, so this module is that specification in a form a test can check.  The
prose half lives in ``notes/estimands.md``; the two must agree, and
``tests/test_measurement_contract.py`` is what keeps them agreeing.

What the contract pins down:

* **Population.**  The finite set of released MyceliumSeg images (20,176 JPGs,
  567 labeled masks), described rather than sampled: Gate C established that
  neither dish identity nor capture time exists, so nothing here generalises
  beyond the release and nothing here is a rate per unit time.
* **Estimands.**  :data:`IMAGE_ESTIMANDS` is the primary image-level morphology
  contract -- area fraction, log area, equivalent radius and the calibrated
  shape features, stratified by species and known condition.  :data:`FRAME_ESTIMANDS`
  holds the single conditional frame-level estimand,
  ``delta_log_area_per_frame``, which is defined only for sequences whose order
  has been demonstrated.
* **Inclusion rule, duplicate handling, units, calibration and uncertainty
  target**, each recorded with the number it is checked against in
  :data:`CONTRACT`.

The guards are the point of the module:

* :func:`frame_delta_log_area` refuses to difference two areas unless frame
  order is demonstrated -- an unordered pair of areas carries no direction.
* :func:`to_physical_rate` refuses to turn a per-frame change into a change per
  unit time unless a capture interval is supplied -- a frame index is not a
  duration (``notes/gate.md`` section 1).
* :func:`log_area` refuses a log of area with no reference unit:
  log(mm^2) and log(px^2) are different quantities.

Everything here is numbers-in / numbers-out with no file I/O, so the tests can
drive it on synthetic inputs (plus the one committed CSV the duplicate rule is
counted against).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

# --- vocabulary ---------------------------------------------------------------

LEVEL_IMAGE = "image"
LEVEL_FRAME = "frame"
LEVELS = (LEVEL_IMAGE, LEVEL_FRAME)

STATUS_PRIMARY = "primary"
STATUS_SECONDARY = "secondary"
STATUS_CONDITIONAL = "conditional"
STATUSES = (STATUS_PRIMARY, STATUS_SECONDARY, STATUS_CONDITIONAL)

PRECONDITION_NONE = "none"
PRECONDITION_FRAME_ORDER = "frame_order"
PRECONDITIONS = (PRECONDITION_NONE, PRECONDITION_FRAME_ORDER)

# A log of a dimensioned quantity needs its reference unit stated: the contract
# allows millimetres^2 (the calibrated quantity) and pixels^2 (scale-free, for
# within-resolution comparisons only), and nothing else.
LOG_UNITS = ("mm^2", "px^2")

# Time units are exactly what a per-frame change may *not* be dressed up as
# without a capture interval, so they are named here to be checked against.
TIME_UNITS = ("minute", "hour", "day")
TIME_WORDS = TIME_UNITS + (
    "second",
    "week",
    "rate_per_time",
)

# Kind of feature, with the uncertainty target that applies to it.
KIND_SIZE = "size"
KIND_BOUNDARY = "boundary"
KINDS = (KIND_SIZE, KIND_BOUNDARY)

REQUIRED_CONTRACT_KEYS = (
    "population",
    "inclusion_rule",
    "duplicate_handling",
    "units",
    "calibration",
    "uncertainty_target",
    "prohibited",
)

REQUIRED_ESTIMAND_FIELDS = (
    "name",
    "level",
    "definition",
    "unit",
    "strata",
    "status",
    "precondition",
)


class ContractViolation(ValueError):
    """An estimand was computed in a way the contract forbids."""


# --- estimand registry --------------------------------------------------------


@dataclass(frozen=True)
class Estimand:
    """One named quantity: what it is, in what unit, over which strata."""

    name: str
    level: str  # LEVEL_IMAGE | LEVEL_FRAME
    definition: str  # formula or verbal definition, unique and checkable
    unit: str  # the unit the number is reported in
    strata: tuple[str, ...]  # the strata the estimand is reported by
    status: str  # STATUS_PRIMARY | STATUS_SECONDARY | STATUS_CONDITIONAL
    precondition: str  # PRECONDITION_NONE | PRECONDITION_FRAME_ORDER
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level,
            "definition": self.definition,
            "unit": self.unit,
            "strata": list(self.strata),
            "status": self.status,
            "precondition": self.precondition,
            "notes": self.notes,
        }


# The primary image-level estimands.  "By species and known condition" is the
# strata field: species (GL/GS/PO/TS), the medium and temperature the release
# actually labels (known_labels.csv), and the condition block, which is a
# nuisance stratum because the px->mm calibration hangs off it.
IMAGE_ESTIMANDS: tuple[Estimand, ...] = (
    Estimand(
        name="area_fraction",
        level=LEVEL_IMAGE,
        definition=(
            "A_colony / A_dish, both measured in the same frame (mask px over the "
            "fitted 90-mm dish disc px) -- the ratio is identical in mm^2 and px^2"
        ),
        unit="1",
        strata=("species", "condition", "condition_block"),
        status=STATUS_PRIMARY,
        precondition=PRECONDITION_NONE,
        notes=(
            "dimensionless and calibration-free (one frame, one scale), but it needs "
            "a readable dish disc: off_scale images have no trustworthy denominator "
            "and are excluded from it"
        ),
    ),
    Estimand(
        name="log_area",
        level=LEVEL_IMAGE,
        definition="ln(A_colony / 1 mm^2) of the calibrated colony mask area",
        unit="ln(mm^2)",
        strata=("species", "condition", "condition_block"),
        status=STATUS_PRIMARY,
        precondition=PRECONDITION_NONE,
        notes=(
            "right-skewed and multiplicative: a length-scale error is squared into "
            "area, and in logs it is additive; reference unit is mandatory, and "
            "off_scale images report ln(px^2) at the 900-px width instead, flagged"
        ),
    ),
    Estimand(
        name="equivalent_radius",
        level=LEVEL_IMAGE,
        definition="r_eq = sqrt(A_colony / pi)",
        unit="mm",
        strata=("species", "condition", "condition_block"),
        status=STATUS_PRIMARY,
        precondition=PRECONDITION_NONE,
        notes="derived from calibrated area, so its error is area error halved in log space",
    ),
    Estimand(
        name="perimeter",
        level=LEVEL_IMAGE,
        definition="P of the colony mask boundary, measured on the 900-px working image",
        unit="mm",
        strata=("species", "condition", "condition_block"),
        status=STATUS_PRIMARY,
        precondition=PRECONDITION_NONE,
        notes="calibrated shape feature; boundary-derived, so it carries the boundary error budget",
    ),
    Estimand(
        name="circularity",
        level=LEVEL_IMAGE,
        definition="4 * pi * A_colony / P^2",
        unit="1",
        strata=("species", "condition", "condition_block"),
        status=STATUS_PRIMARY,
        precondition=PRECONDITION_NONE,
        notes="dimensionless but resolution-locked: P changes with working width, so fix width = 900 px",
    ),
    Estimand(
        name="edge_roughness",
        level=LEVEL_IMAGE,
        definition="P / (2 * pi * r_eq) -- perimeter excess over a circle of equal area",
        unit="1",
        strata=("species", "condition", "condition_block"),
        status=STATUS_PRIMARY,
        precondition=PRECONDITION_NONE,
        notes="1.0 = circle; resolution-locked like circularity (working width = 900 px)",
    ),
    Estimand(
        name="solidity",
        level=LEVEL_IMAGE,
        definition="A_colony / A_convex_hull of the colony mask",
        unit="1",
        strata=("species", "condition", "condition_block"),
        status=STATUS_PRIMARY,
        precondition=PRECONDITION_NONE,
        notes="dimensionless shape feature; hull is stable against pixel-scale noise",
    ),
    Estimand(
        name="mean_intensity",
        level=LEVEL_IMAGE,
        definition="mean 8-bit grey level inside the colony mask",
        unit="grey level",
        strata=("condition_block",),
        status=STATUS_SECONDARY,
        precondition=PRECONDITION_NONE,
        notes="density proxy only: dark incubation and fixed exposure are per-block, not cross-block",
    ),
)


# The one conditional frame-level estimand.  It is defined as a change per
# *frame*, never per unit time, and only once frame order has been demonstrated
# independently of the file numbering (#22 says the numbering is shuffled).
FRAME_ESTIMANDS: tuple[Estimand, ...] = (
    Estimand(
        name="delta_log_area_per_frame",
        level=LEVEL_FRAME,
        definition=(
            "ln(A_frame+1 / 1 mm^2) - ln(A_frame / 1 mm^2) for consecutive frames "
            "of one validated sequence, in demonstrated frame order"
        ),
        unit="log units per frame",
        strata=("species", "condition", "sequence"),
        status=STATUS_CONDITIONAL,
        precondition=PRECONDITION_FRAME_ORDER,
        notes=(
            "the denominator is one frame step, not a duration: report per frame "
            "until a capture interval exists (to_physical_rate guards the conversion)"
        ),
    ),
)

ALL_ESTIMANDS: tuple[Estimand, ...] = IMAGE_ESTIMANDS + FRAME_ESTIMANDS
ESTIMANDS_BY_NAME: dict[str, Estimand] = {e.name: e for e in ALL_ESTIMANDS}

# The unit each estimand is reported in, derived from the registry so the two
# can never drift apart.
UNITS: dict[str, str] = {e.name: e.unit for e in ALL_ESTIMANDS}


# --- contract -----------------------------------------------------------------

CONTRACT: dict[str, Any] = {
    "spec": "measurement contract for observable growth estimands",
    "issue": 60,
    "dataset": (
        "MyceliumSeg (Yuan et al. 2025, doi:10.1038/s41597-025-06265-1): "
        "20,176 JPGs in 11 archives, 567 labeled masks, 90-mm dishes"
    ),
    "gate": "C -- no dish identity and no capture time (notes/gate.md, #26)",
    "population": {
        "definition": (
            "every released MyceliumSeg image that survives the inclusion rule; "
            "a finite, fully enumerable set -- not a sample of dishes"
        ),
        "n_images": 20176,
        "n_labeled_masks": 567,
        "n_unlabeled": 19609,
        "n_condition_known": 18487,  # species/medium labelled per image (known_labels.csv)
        "n_condition_unknown": 1689,  # flat unlabeled-GS_PO_TS shard, species/medium blank
        "sampling_frame": (
            "none -- the release is not a random sample of plates; whether an "
            "edge-morphology selection rule was applied is undocumented (#35)"
        ),
        "generalization": (
            "descriptive of the release only; associations by species and known "
            "condition, never a species-population or causal claim"
        ),
    },
    "inclusion_rule": {
        "requires": [
            (
                "the image is in the release and has a segmentation to measure "
                "(labeled mask or inference output)"
            ),
            "it is the representative of its byte-identical duplicate group",
            (
                "condition labels come from known_labels.csv (folder + paper Table 1) "
                "for any stratum that needs them; the 20 mislabelled images are "
                "corrected from the md5 map or dropped"
            ),
            (
                "anything that reads the dish disc or the px->mm scale -- "
                "area_fraction, log_area in mm^2, equivalent_radius, perimeter -- "
                "additionally requires image_scale.csv fit_status == 'ok' for that "
                "image; off_scale images keep the flag and stay in the ratio shape "
                "features, mean_intensity and log_area in px^2"
            ),
            "boundary-derived shape features are measured at the 900-px working width",
        ],
        "excludes": [
            "duplicate copies after the representative (71 of 20,176 images)",
            (
                "images with no readable condition label, from condition-stratified "
                "estimands only (1,689 images stay in block-level and unstratified "
                "ones)"
            ),
            (
                "off_scale images (560, 2.8%) from the dish-disc and mm-calibrated "
                "estimands (area_fraction, log_area ln(mm^2), equivalent_radius, "
                "perimeter) -- their fitted dish disc is the thing that failed"
            ),
        ],
        "reporting": (
            "publish N included and N excluded per estimand and per stratum next to "
            "every summary; an exclusion that changes a headline number is a finding"
        ),
    },
    "duplicate_handling": {
        "criterion": "byte-identical MD5, recorded per image in data/triage/dish_tracks.csv",
        "n_duplicate_groups": 62,
        "n_duplicate_images": 133,
        "n_after_deduplication": 20105,
        "rule": (
            "one representative per hash -- the lowest (archive, id) -- kept for "
            "estimation, uncertainty and train/test splitting; the rest are counted "
            "and dropped, never split across a split or a CI"
        ),
        "sensitivity": (
            "re-run headline numbers with every copy included to show they do not move"
        ),
        "frame_rule": (
            "de-duplicate before differencing: two identical copies of one frame "
            "would otherwise report delta_log_area_per_frame = 0 as 'no growth'"
        ),
        "labels": (
            "PO == MYG and PDA == ten labeled-GL images: 20 folder-derived labels "
            "are wrong, so condition comes from the md5 map, not the folder name"
        ),
    },
    "units": UNITS,
    "calibration": {
        "anchor": "the 90-mm dish rim (DISH_DIAMETER_MM), fit per image and grouped per setup",
        "tables": ["data/triage/calibration.csv", "data/triage/image_scale.csv"],
        "constants_px_per_mm_at_900": {
            "GL": 6.0444,
            "GS": 4.4482,
            "GS_PO_TS_flat_setup1": 6.6222,
            "GS_PO_TS_flat_setup2": 4.4444,
            "MYG_PO": 6.6345,
            "PDA": 6.0419,
            "TEMP15": 6.0444,
            "TS": 6.6507,
        },
        "resolution_rule": (
            "constants are for 900-px working images; px/mm scales linearly with "
            "frame width (rescale_px_per_mm), and area scales with its square -- "
            "one global constant is off by 36% on the GS block, i.e. every area "
            "46% too small"
        ),
        "uncertainty_from_scale": (
            "carry px_per_mm_p25..p75 into every mm^2 quantity; the spread is "
            "0.0-1.7% in length, so 0.0-3.4% in area"
        ),
        "flagged": "560 images (2.8%) are fit_status == 'off_scale' and take their block's prior",
    },
    "uncertainty_target": {
        "confidence_level": 0.95,
        "estimation_unit": "the deduplicated image (duplicate groups are one observation)",
        "targets_pct": {
            KIND_SIZE: {"relative_bias": 3.0, "relative_sd": 10.0},
            KIND_BOUNDARY: {"relative_bias": 5.0, "relative_sd": 20.0},
        },
        "frame_detection_sigma": 2.0,
        "frame_rule": (
            "report delta_log_area_per_frame only when |delta| exceeds "
            "frame_detection_sigma x the frame-to-frame repeatability SD"
        ),
        "on_failure": (
            "publish the number with the target marked unmet; never drop it, never "
            "retune the threshold until it passes"
        ),
        "label": (
            "intervals describe the release (associational, within-release), not a "
            "sample of dishes or a population of plates"
        ),
    },
    "prohibited": [
        (
            "any per-time rate (mm/day, log/day, events/hour) from per-frame change "
            "without a capture interval"
        ),
        (
            "calling a MyceliumSeg sequence a growth curve, or inferring elapsed time "
            "from file numbering or zip mtimes"
        ),
        (
            "causal temperature or medium effects (Gate C: medium is aliased with "
            "species; 10 images at 15 C; no dish identity, no replication)"
        ),
        (
            "stratifying on growth stage or sclerotium colour (no per-image labels; "
            "stage is downstream of growth)"
        ),
        "boundary events reported as biological hyphal tips",
        "a log of area without its reference unit",
        (
            "mixing px^2 and mm^2 areas in one summary, or mixing working widths in "
            "one boundary-feature summary"
        ),
    ],
}


# --- selection-bias record (Chainlink #35) -------------------------------------

# The paper's Methods / Data collection section states: "We collected 20,176
# mycelium images with distinctive edge morphology" (Yuan et al. 2025,
# doi:10.1038/s41597-025-06265-1).  That sentence *describes the collected
# release*; it does not document an image-selection inclusion rule based on edge
# morphology.  The same section describes process-level, periodic fixed-position
# acquisition ("we periodically placed mycelial petri dishes at a fixed position
# for image acquisition") but ships no per-image dish, time, order or stage link.
#
# So neither half of assumption #35 -- selection is independent of latent growth
# features and frame/time -- can be settled from the source or the release, and
# this record keeps both halves below "supported":
#
#   * independence from OBSERVED morphology is UNDOCUMENTED -- the source names
#     no inclusion rule, no candidate/included/excluded image frame, and no
#     in-release evidence establishes that selection read the measured
#     morphology.  The release cannot be shown to over-represent the estimands'
#     own features, and the direction of any morphology bias is unknown.
#   * independence from LATENT time/frame is UNTESTABLE under Gate C -- dish
#     identity, capture timestamps and ordered sequences are all unavailable
#     (notes/gate.md section 1).  Periodic fixed-position acquisition is
#     process-level context, not per-image linkage.
SELECTION_STATUSES = ("undocumented", "untestable")

SELECTION_BIAS: dict[str, Any] = {
    "issue": 35,
    "assumption": (
        "image selection for distinctive edge morphology is independent of "
        "latent growth features and frame/time"
    ),
    "release_description": (
        "We collected 20,176 mycelium images with distinctive edge morphology"
    ),
    # The paper documents no inclusion rule; the field stays None so no
    # selection mechanism can be asserted by accident.
    "documented_inclusion_rule": None,
    "acquisition_process": (
        "we periodically placed mycelial petri dishes at a fixed position for "
        "image acquisition (process-level context; no per-image dish/time link)"
    ),
    "source": (
        "Yuan et al. 2025, Methods/Data collection, "
        "doi:10.1038/s41597-025-06265-1; in-repo: PLAN.md section 0, "
        "notes/estimands.md section 1, notes/gate.md section 1"
    ),
    "claims": {
        "independent_of_observed_morphology": "undocumented",
        "independent_of_latent_time_frame": "untestable",
    },
    "supported": (
        "cross-sectional morphology distributions describe the selected release "
        "only: associational, within-release, by species and known condition -- "
        "never a plate population, a treatment effect or a growth curve"
    ),
    "gate_c_prohibition": (
        "no causal temperature or medium effect and no physical growth-rate "
        "claim is licensed by any status above (notes/gate.md)"
    ),
    "required_to_test": (
        "the documented inclusion rule; the candidate / included / excluded "
        "image frame; and linked per-image dish/time or growth-stage metadata"
    ),
}


def selection_claim(claim: str) -> str:
    """Return the #35 status of one half of the selection assumption.

    The two halves are fixed: morphology-based selection is *undocumented* (the
    paper describes the release, not an inclusion rule) and dependence on latent
    time/frame is *untestable* under Gate C.  A claim the record does not carry
    is a :class:`ContractViolation` -- there is no code path through which
    selection independence may be asserted.
    """
    try:
        return SELECTION_BIAS["claims"][claim]
    except KeyError:
        raise ContractViolation(
            f"unknown selection claim {claim!r}; the #35 record carries exactly "
            f"{sorted(SELECTION_BIAS['claims'])}"
        ) from None


def validate_selection_record() -> list[str]:
    """Return every way the #35 selection-bias record breaks its own rules.

    Empty list means the record is internally consistent: exactly the two
    halves, each with a status the evidence supports (never "supported" or the
    retracted "falsified"), no invented inclusion rule, the release description
    quoted, periodic acquisition recorded as process-level context, what a
    future test needs stated, the primary paper cited, and the supported scope
    bounded to the release.
    """
    problems: list[str] = []
    claims = SELECTION_BIAS.get("claims") or {}
    if set(claims) != {
        "independent_of_observed_morphology",
        "independent_of_latent_time_frame",
    }:
        problems.append(
            "selection record must carry exactly the two #35 halves: "
            "independent_of_observed_morphology, independent_of_latent_time_frame"
        )
    for claim, status in claims.items():
        if status not in SELECTION_STATUSES:
            problems.append(
                f"selection claim {claim!r} has status {status!r}; the evidence "
                f"supports only {list(SELECTION_STATUSES)}"
            )
    if SELECTION_BIAS.get("documented_inclusion_rule") is not None:
        problems.append(
            "selection record must not invent an inclusion rule: the source "
            "documents none, so documented_inclusion_rule must stay None"
        )
    if "distinctive edge morphology" not in (
        SELECTION_BIAS.get("release_description") or ""
    ):
        problems.append(
            "selection record must quote the release description that names "
            "distinctive edge morphology"
        )
    acquisition = SELECTION_BIAS.get("acquisition_process") or ""
    if "periodically" not in acquisition and "fixed position" not in acquisition:
        problems.append(
            "selection record must record periodic fixed-position acquisition "
            "as process-level context"
        )
    required = SELECTION_BIAS.get("required_to_test") or ""
    for token in ("inclusion rule", "candidate", "dish/time"):
        if token not in required:
            problems.append(
                f"selection record must state what a future test needs ({token!r})"
            )
    if "10.1038/s41597-025-06265-1" not in (SELECTION_BIAS.get("source") or ""):
        problems.append("selection record must cite the primary paper DOI")
    if not SELECTION_BIAS.get("supported"):
        problems.append("selection record must state the supported scope")
    return problems


# --- validation ---------------------------------------------------------------


def validate_contract() -> list[str]:
    """Return every way :data:`CONTRACT` and the registries break their own rules.

    Empty list means the contract is internally consistent: all sections
    present, every estimand fully and coherently specified, no time unit hiding
    in a per-frame quantity, the prohibition on unconverted frame change
    actually recorded, and the #35 selection-bias record carrying exactly the
    two halves with the statuses the evidence supports (undocumented morphology
    selection, untestable time/frame dependence).
    """
    problems: list[str] = validate_selection_record()

    for key in REQUIRED_CONTRACT_KEYS:
        if key not in CONTRACT or not CONTRACT[key]:
            problems.append(f"CONTRACT is missing section {key!r}")

    names = [e.name for e in ALL_ESTIMANDS]
    if len(names) != len(set(names)):
        problems.append("duplicate estimand names: " + ", ".join(sorted(names)))

    for estimand in ALL_ESTIMANDS:
        for field in REQUIRED_ESTIMAND_FIELDS:
            value = getattr(estimand, field)
            if not value or value == ():
                problems.append(f"{estimand.name}: field {field!r} is empty")
        if estimand.level not in LEVELS:
            problems.append(f"{estimand.name}: unknown level {estimand.level!r}")
        if estimand.status not in STATUSES:
            problems.append(f"{estimand.name}: unknown status {estimand.status!r}")
        if estimand.precondition not in PRECONDITIONS:
            problems.append(
                f"{estimand.name}: unknown precondition {estimand.precondition!r}"
            )
        if not estimand.name.islower() or not estimand.name.isidentifier():
            problems.append(f"{estimand.name}: name must be lower_snake_case")
        if estimand.level == LEVEL_FRAME and estimand.precondition == PRECONDITION_NONE:
            problems.append(
                f"{estimand.name}: a frame-level estimand must carry a precondition"
            )
        if estimand.status == STATUS_CONDITIONAL and (
            estimand.precondition == PRECONDITION_NONE
        ):
            problems.append(f"{estimand.name}: conditional means preconditioned")
        lowered = estimand.unit.lower()
        for word in TIME_WORDS:
            if word in lowered:
                problems.append(
                    f"{estimand.name}: unit {estimand.unit!r} names a time unit; "
                    "per-frame quantities may not be rates"
                )

    if CONTRACT.get("units") != UNITS:
        problems.append("CONTRACT['units'] has drifted from the estimand registry")

    target = CONTRACT.get("uncertainty_target") or {}
    level = target.get("confidence_level")
    if not isinstance(level, float) or not 0.0 < level < 1.0:
        problems.append("uncertainty_target.confidence_level must be in (0, 1)")
    targets = target.get("targets_pct") or {}
    for kind in KINDS:
        if kind not in targets:
            problems.append(f"uncertainty_target has no limit for kind {kind!r}")

    if not any(
        "capture interval" in str(entry) for entry in CONTRACT.get("prohibited", [])
    ):
        problems.append("prohibited must name the capture-interval rule")

    if not IMAGE_ESTIMANDS or not FRAME_ESTIMANDS:
        problems.append("both image-level and frame-level estimands must be defined")

    return problems


# --- estimand computation -----------------------------------------------------


def log_area(area: float, *, reference_unit: str = "mm^2") -> float:
    """``ln(A / reference_unit)`` -- a log of area with its unit stated.

    The reference unit is mandatory because log(mm^2) and log(px^2) differ by
    twice the log of the scale: dropping it silently moves every log area by
    2 * ln(px/mm) *and* mixes resolutions.
    """
    if reference_unit not in LOG_UNITS:
        raise ContractViolation(
            f"reference unit must be one of {LOG_UNITS}, got {reference_unit!r}"
        )
    if not isinstance(area, (int, float)) or not math.isfinite(area) or area <= 0:
        raise ContractViolation(f"area must be a positive finite number, got {area!r}")
    return math.log(float(area))


def frame_delta_log_area(
    area_before: float,
    area_after: float,
    *,
    frame_order_demonstrated: bool,
    reference_unit: str = "mm^2",
) -> float:
    """``ln(A_after) - ln(A_before)`` per frame -- conditional on demonstrated order.

    File numbering is not frame order (#22: forward steps ~0.5), so the caller
    has to assert that order was demonstrated by independent evidence.  Without
    that assertion the two areas are an unordered pair and "before/after" means
    nothing, which is why an undemonstrated order raises instead of returning a
    number.
    """
    if not frame_order_demonstrated:
        raise ContractViolation(
            "frame order is not demonstrated: delta_log_area_per_frame is a "
            "conditional estimand (#60) and needs evidence independent of the "
            "file numbering before two frames may be differenced"
        )
    return log_area(area_after, reference_unit=reference_unit) - log_area(
        area_before, reference_unit=reference_unit
    )


def to_physical_rate(
    delta_log_per_frame: float,
    capture_interval: float | None,
    *,
    time_unit: str = "hour",
) -> float:
    """Convert a per-frame change to a change per unit time -- or refuse to.

    The denominator has to be a measured capture interval.  None, non-positive
    or non-finite intervals raise: MyceliumSeg has no capture interval at all
    (Gate C), so its ``delta_log_area_per_frame`` stays per frame.
    """
    if time_unit not in TIME_UNITS:
        raise ContractViolation(
            f"time_unit must be one of {TIME_UNITS}, got {time_unit!r}"
        )
    if capture_interval is None:
        raise ContractViolation(
            "no capture interval: a frame index is not a duration, so a per-frame "
            "change may not be reported per unit time (notes/gate.md, #60)"
        )
    if not isinstance(capture_interval, (int, float)):
        raise ContractViolation(
            f"capture interval must be a number of {time_unit}s, got {capture_interval!r}"
        )
    if not math.isfinite(capture_interval) or capture_interval <= 0:
        raise ContractViolation(
            f"capture interval must be positive and finite, got {capture_interval!r}"
        )
    if not isinstance(delta_log_per_frame, (int, float)):
        raise ContractViolation(
            f"delta_log_per_frame must be a number, got {delta_log_per_frame!r}"
        )
    return float(delta_log_per_frame) / float(capture_interval)


def meets_uncertainty_target(
    kind: str,
    *,
    relative_bias_pct: float,
    relative_sd_pct: float,
) -> bool:
    """Whether measured error clears the target for ``kind`` (size / boundary).

    Both the bias *and* the SD have to clear their limits: a small spread around
    a badly biased segmenter is still a wrong area.
    """
    targets = CONTRACT["uncertainty_target"]["targets_pct"]
    if kind not in targets:
        raise ValueError(
            f"unknown estimand kind {kind!r}, expected one of {sorted(targets)}"
        )
    limits = targets[kind]
    return (
        abs(relative_bias_pct) <= limits["relative_bias"]
        and abs(relative_sd_pct) <= limits["relative_sd"]
    )


def min_reportable_delta(repeatability_sd_log: float) -> float:
    """Smallest ``|delta_log_area_per_frame|`` worth reporting: 2 x repeatability SD.

    Below that, a frame-to-frame change is indistinguishable from the
    measurement error of two independent masks (``notes/estimands.md`` section 8).
    """
    if not isinstance(repeatability_sd_log, (int, float)) or not math.isfinite(
        repeatability_sd_log
    ):
        raise ContractViolation(
            f"repeatability_sd_log must be a finite number, got {repeatability_sd_log!r}"
        )
    if repeatability_sd_log < 0:
        raise ContractViolation(
            f"a standard deviation cannot be negative, got {repeatability_sd_log!r}"
        )
    return float(CONTRACT["uncertainty_target"]["frame_detection_sigma"]) * float(
        repeatability_sd_log
    )


def deduplicate(
    rows: Iterable[Mapping[str, Any]],
    *,
    hash_key: str = "md5",
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split rows into ``(representatives, duplicates)`` by byte-identical hash.

    The representative of a group is the row with the lowest ``(archive, id)``
    -- deterministic, so the same input always de-duplicates the same way --
    and both outputs keep the input order.  Rows without a hash are a data
    error, not a duplicate: silently treating them as unique would let an
    unhashed copy of a frame differenced against itself.
    """
    rows = list(rows)  # one pass: the function must not consume a generator twice
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for index, row in enumerate(rows):
        digest = row.get(hash_key)
        if not isinstance(digest, str) or not digest:
            raise ValueError(f"row {index} has no {hash_key!r} to de-duplicate on")
        groups.setdefault(digest, []).append(row)

    keep: set[int] = set()
    for group in groups.values():
        if len(group) == 1:
            keep.add(id(group[0]))
            continue
        # Rank the group; ties fall back to input order, which min() keeps.
        ranked = sorted(
            group,
            key=lambda row: (
                str(row.get("archive", "")),
                str(row.get("id", "")),
            ),
        )
        keep.add(id(ranked[0]))

    kept: list[Mapping[str, Any]] = []
    dropped: list[Mapping[str, Any]] = []
    for row in rows:
        (kept if id(row) in keep else dropped).append(row)
    return kept, dropped
