#!/usr/bin/env Rscript
# ===========================================================================
# R/dag.R — Causal DAGs for the MyceliumSeg observable-dynamics project
# ===========================================================================
# Chainlink #33. PLAN.md Phase 3.
#
# This script defines two families of DAGs and checks adjustment sets with
# dagitty (0.3.4, from the repository Nix R environment):
#
# 1. dag_myceliumseg — the MyceliumSeg cross-sectional DAG under Gate C
#    (notes/gate.md). Dish identity (D) and capture time (t) are NOT
#    recoverable from the release, so they are latent. The physical growth
#    rate (k) requires a time axis and is latent. The identifiable estimand
#    is the cross-sectional association E[Y | S, M, T] — associational, not
#    causal. Causal temperature/medium effects and the growth rate are NOT
#    identifiable.
#
# 2. dag_intervention / dag_intervention_alloc — future causal intervention
#    DAGs for a dataset with identifiable dish, time, treatment allocation
#    and replication (Fricker/Boddy, Podospora, or a controlled rig —
#    notes/gate.md §5, notes/provenance.md §5). With D and t observed, the
#    causal effect T -> k becomes identifiable.
#
# Evidence base: PLAN.md Phase 3, notes/gate.md (Gate C), notes/estimands.md
# (#60), notes/provenance.md (#61), notes/measurement_error.md (#31).
#
# Run: nix develop --command Rscript R/dag.R
# ===========================================================================

library(dagitty)

# --- helpers ----------------------------------------------------------------

# Print a paragraph (single string, may contain \n).
para <- function(text) {
  cat(text, "\n", sep = "")
}

# Print an adjustment-set query with a clear verdict.
#   dag       : a dagitty graph
#   exposure  : exposure node name
#   outcome   : outcome node name
#   label     : human-readable description of the query
# Returns the dagitty.sets result invisibly.
check_adjustment <- function(dag, exposure, outcome, label) {
  cat("\n---", label, "---\n")
  cat("Query: adjustmentSets(dag, exposure = \"", exposure,
      "\", outcome = \"", outcome, "\")\n", sep = "")
  sets <- adjustmentSets(dag, exposure = exposure, outcome = outcome)
  if (length(sets) == 0) {
    cat("Result: EMPTY (length 0) — no adjustment set identifies this effect.\n")
  } else {
    cat("Result:\n")
    print(sets)
  }
  invisible(sets)
}

# Node list from a dagitty graph (edges() returns columns v = from, w = to).
graph_nodes <- function(dag) {
  e <- edges(dag)
  sort(unique(c(e$v, e$w)))
}

# ===========================================================================
# 1. MyceliumSeg cross-sectional DAG (Gate C)
# ===========================================================================
# The graph below is the PLAN.md Phase 3 draft made explicit, with every edge
# tied to the committed evidence. NOTE: dagitty graph strings do not support
# '#' comments, so all annotation lives in R comments here.
#
#   S -> M        species-medium aliasing by design (GL/GS on MYG, PO/TS on
#                 PDA) — PLAN.md §0, notes/gate.md §3, notes/estimands.md §3
#   S -> Z        species effect on latent colony morphology
#   M -> Z        medium effect on latent morphology (aliased with S)
#   S -> T        design: temperature varied only within GL (15 vs 25 °C)
#   T -> Z        temperature effect on latent morphology
#   T -> k        temperature effect on physical growth rate — the causal
#                 estimand exp(beta_T) of PLAN.md Phase 4C
#   D -> T        dish determines its incubation temperature (assignment)
#   D -> Z        dish varying effect on morphology (PLAN.md Phase 3 draft)
#   D -> k        dish varying growth rate (PLAN.md Phase 4C: log k[d])
#   t -> Z        capture time effect on morphology — latent under Gate C
#   k -> Z        growth rate -> morphology at capture
#   k -> G        growth rate -> growth stage (stage is downstream of growth)
#   Z -> Y        true morphology -> observed feature Y
#   I -> Y        imaging/segmentation error (notes/measurement_error.md #31)
#   Z -> Sel      selection on "distinctive edge morphology" reads the true
#                 morphology (PLAN.md §0) — so selection is NOT independent of
#                 observed morphology: that half of assumption #35 is falsified
#                 by construction (notes/assumptions.md §5)
#   I -> Sel      ... and the imaged morphology, so selection is a collider
#                 on Z -> Sel <- I (PLAN.md Phase 3 draft: "possible collider")
#   (t,k) -> Sel  dependence of selection on latent time/frame is UNTESTABLE
#                 under Gate C — no dish identity, no timestamps, no ordered
#                 sequences (notes/gate.md §1) — and no temporal independence
#                 may be asserted (assumption #35, notes/assumptions.md §5)
#   Sel -> In     inclusion in the release / analysis
#   Dup -> C_obs  duplicate/mislabeled file -> observed condition label
#                 (133 byte-identical images, 20 wrong folder labels —
#                 notes/gate.md §1, notes/provenance.md §4.1)
#   S/M/T -> C_obs  true condition -> observed label
#
# Latent (unobserved) nodes: D, t, k, G, Z, I, Sel.
# Observed nodes: S, M, T, Y, In, Dup, C_obs.

dag_myceliumseg <- dagitty('dag {
  D [latent]
  t [latent]
  k [latent]
  G [latent]
  Z [latent]
  I [latent]
  Sel [latent]
  S -> M
  S -> Z
  M -> Z
  S -> T
  T -> Z
  T -> k
  D -> T
  D -> Z
  D -> k
  t -> Z
  k -> Z
  k -> G
  Z -> Y
  I -> Y
  Z -> Sel
  I -> Sel
  Sel -> In
  Dup -> C_obs
  S -> C_obs
  M -> C_obs
  T -> C_obs
}')

cat("=================================================================\n")
cat("1. MyceliumSeg cross-sectional DAG (Gate C — notes/gate.md)\n")
cat("=================================================================\n")
cat("Nodes (", length(graph_nodes(dag_myceliumseg)), "): ",
    paste(graph_nodes(dag_myceliumseg), collapse = ", "), "\n", sep = "")
cat("Latent (unobserved): D, t, k, G, Z, I, Sel\n")
cat("Observed: S, M, T, Y, In, Dup, C_obs\n")
cat("Edges (", nrow(edges(dag_myceliumseg)), ")\n", sep = "")

# ===========================================================================
# 2. Adjustment-set checks — MyceliumSeg (Gate C)
# ===========================================================================

cat("\n-----------------------------------------------------------------")
cat("\n2a. Causal temperature effect on observed morphology: T -> Y")
cat("\n-----------------------------------------------------------------")
check_adjustment(dag_myceliumseg, "T", "Y",
  "T -> Y  (causal temperature effect on observed morphology)")
para("Why not identified: the back-door path T <- D -> Z -> Y runs through the
latent dish D. Gate C (notes/gate.md §1) established that dish identity
is not recoverable, so D cannot be conditioned on. The path T <- S -> Z
is blocked by {S}, but the D-path is unblockable.")

cat("\n-----------------------------------------------------------------")
cat("\n2b. Causal medium effect on observed morphology: M -> Y")
cat("\n-----------------------------------------------------------------")
check_adjustment(dag_myceliumseg, "M", "Y",
  "M -> Y  (causal medium effect on observed morphology)")
para("Why not identified: the only formal adjustment set is {S}, but it is
VACUOUS. M is a deterministic function of S (S -> M, design aliasing):
each species is on exactly one medium (GL/GS on MYG, PO/TS on PDA), so
M does not vary within levels of S and the adjusted contrast has no data
support. The medium effect is aliased with the species effect by design.
notes/gate.md §3: 'Not identified at all — do not estimate.'")

cat("\n-----------------------------------------------------------------")
cat("\n2c. Causal temperature effect on physical growth rate: T -> k")
cat("\n-----------------------------------------------------------------")
check_adjustment(dag_myceliumseg, "T", "k",
  "T -> k  (causal temperature effect on physical growth rate)")
para("Why not identified: (a) k is latent — the growth rate requires a time
axis, and Gate C says no capture interval exists (notes/gate.md §1:
0/20,176 EXIF, zip mtimes are export sessions, numbering is shuffled).
(b) The back-door path T <- D -> k runs through the latent dish D and is
unblockable. notes/gate.md §3: 'Not identified — do not fit.'")

cat("\n-----------------------------------------------------------------")
cat("\n2d. Growth-stage stratification (PLAN.md Phase 3 implication)")
cat("\n-----------------------------------------------------------------")
para("G is a descendant of k (k -> G) and hence of T (T -> k -> G): growth
stage is downstream of growth. Conditioning on a descendant of the
mediator/outcome biases the total effect of T. G is also latent — no
per-image stage labels ship (#24, notes/gate.md §1) — so it cannot be
adjusted for anyway. PLAN.md Phase 3: 'Stratifying by growth stage can
bias a treatment effect because stage is downstream of growth.'")

cat("\n-----------------------------------------------------------------")
cat("\n2e. The identifiable estimand: cross-sectional association")
cat("\n-----------------------------------------------------------------")
para("E[Y | S, M, T] — the observed conditional distribution of morphology
given condition — is identifiable from the release: it is the observed
data distribution and needs no causal identification. It is
ASSOCIATIONAL, not causal: it describes this release, not a treatment
effect or a growth curve (notes/estimands.md §3, notes/gate.md §4).
This is the only estimand MyceliumSeg supports under Gate C.")

# ===========================================================================
# 3. Future causal intervention DAGs
# ===========================================================================
# In a future dataset with identifiable dish, time, treatment allocation and
# replication (notes/gate.md §5, notes/provenance.md §5), D and t are
# observed and the causal effect T -> k becomes identifiable. Two variants:
#
#   dag_intervention       — randomized treatment (T exogenous), as in the
#                            Fricker/Boddy grazer series (notes/provenance.md
#                            §4.2: treatments randomized)
#   dag_intervention_alloc — treatment allocated by dish (D -> T) with D
#                            observed, as in a controlled rig

dag_intervention <- dagitty('dag {
  k [latent]
  G [latent]
  Z [latent]
  I [latent]
  Sel [latent]
  S -> M
  S -> Z
  M -> Z
  T -> Z
  T -> k
  D -> k
  t -> Z
  k -> Z
  k -> G
  Z -> Y
  I -> Y
  Z -> Sel
  I -> Sel
  Sel -> In
  Dup -> C_obs
  S -> C_obs
  M -> C_obs
  T -> C_obs
}')

dag_intervention_alloc <- dagitty('dag {
  k [latent]
  G [latent]
  Z [latent]
  I [latent]
  Sel [latent]
  S -> M
  S -> Z
  M -> Z
  D -> T
  T -> Z
  T -> k
  D -> k
  t -> Z
  k -> Z
  k -> G
  Z -> Y
  I -> Y
  Z -> Sel
  I -> Sel
  Sel -> In
  Dup -> C_obs
  S -> C_obs
  M -> C_obs
  T -> C_obs
}')

cat("\n=================================================================")
cat("\n3. Future causal intervention DAGs (notes/gate.md §5)")
cat("\n=================================================================")
para("Dish identity D and capture time t are observed; treatment is either
randomized (dag_intervention) or allocated by dish with D observed
(dag_intervention_alloc). k remains latent but is now estimable from
the time axis.")

cat("\n-----------------------------------------------------------------")
cat("\n3a. Randomized treatment, D observed: T -> k")
cat("\n-----------------------------------------------------------------")
check_adjustment(dag_intervention, "T", "k",
  "T -> k  (future: randomized treatment, D observed)")
para("Identified: T is randomized (exogenous), so there are no back-door
paths; the empty set (no adjustment) is valid. This is the
Fricker/Boddy case (notes/provenance.md §4.2).")

cat("\n-----------------------------------------------------------------")
cat("\n3b. Treatment allocated by dish, D observed: T -> k")
cat("\n-----------------------------------------------------------------")
check_adjustment(dag_intervention_alloc, "T", "k",
  "T -> k  (future: T allocated by dish, D observed)")
para("Identified: the back-door path T <- D -> k is blocked by D, which is
observed in the future dataset. This is the controlled-rig case
(notes/gate.md §5).")

# ===========================================================================
# 4. Conclusion
# ===========================================================================

cat("\n=================================================================")
cat("\nConclusion")
cat("\n=================================================================")
para("Under Gate C, NO adjustment set makes the following identifiable from
MyceliumSeg:

  * causal temperature effect on observed morphology  T -> Y   (empty)
  * causal medium effect on observed morphology       M -> Y   (only {S}, vacuous)
  * causal temperature effect on physical growth rate T -> k   (empty)

The identifiable estimand is the cross-sectional association E[Y | S, M, T],
which is associational, not causal. Causal intervention effects require a
future dataset with dish identity, capture time, treatment allocation and
replication (Fricker/Boddy, Podospora, or a controlled rig).

Represented in the MyceliumSeg graph:
  * species-medium aliasing   S -> M (deterministic; M = f(S))
  * missing dish/time identity  D, t latent (Gate C)
  * selection for distinctive edge morphology  Z -> Sel <- I (collider;
    #35: falsified for observed morphology, untestable for latent time)
  * imaging/segmentation error  I -> Y (notes/measurement_error.md #31)
  * duplicate/mislabeled files  Dup -> C_obs (133 duplicates, 20 wrong labels)
")
