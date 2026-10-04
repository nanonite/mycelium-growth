#!/usr/bin/env bash
# Seed chainlink epics/issues from PLAN.md. Run once from the workspace root.
set -euo pipefail
cd "$(dirname "$0")/.."

epic() { chainlink issue create -q -l epic "$@"; }
sub()  { chainlink issue subissue -q "$@"; }
blk()  { chainlink issue block "$1" "$2" >/dev/null; }   # $1 blocked by $2

# ── E0 Setup ─────────────────────────────────────────────────────────
E0=$(epic -p high "E0: Setup" -d "Tooling and disk. PLAN.md §Tooling.")
DISK=$(sub "$E0" -p high -l setup "Check free disk (~80 GB: MyceliumSeg 56.5 + Fricker 13.4 + Podospora/other)" -d "Decide data/raw location; MyceliumSeg unlabeled shards only after gate.")
LAYOUT=$(sub "$E0" -p high -l setup "Create project layout + data manifest" -d "data/raw/<dataset>/SOURCE.md (URL, DOI, licence, date, md5), data/manifest.csv, scripts/fetch.sh with md5 verify + resume (curl -C -). Consider git init (data/raw gitignored).")
R_ENV=$(sub "$E0" -l setup "Install R stack: rethinking, cmdstanr + install_cmdstan(), dagitty, posterior")
PY_ENV=$(sub "$E0" -l setup "Install Python stack: MMSegmentation (MyceliumSeg-benchmark), exiftool, scikit-image")

# ── E1 Data acquisition (one dataset per issue, in order) ────────────
E1=$(epic -p high "E1: Data acquisition" -d "Download each dataset one by one; verify md5; record SOURCE.md + licence.")
D1=$(sub "$E1" -p high -l data "Download MyceliumSeg labeled-MYG_PDA_TEMP + labeled-GS_PO_TS (235 MB)" -d "Zenodo 15224240. CC BY 4.0.")
D2=$(sub "$E1" -p high -l data "Download MyceliumSeg labeled-GL (1.5 GB)" -d "457 train / 50 test.")
D3=$(sub "$E1" -p high -l data "Download MyceliumSeg probe shard unlabeled-GL2 (6.2 GB)" -d "Used for metadata triage before committing to the full 56.5 GB.")
D4=$(sub "$E1" -l data "Download remaining MyceliumSeg unlabeled shards (GL1, GL3-7, GS_PO_TS)" -d "Only after gate (E2) says data is usable.")
D5=$(sub "$E1" -p high -l data "Download Fricker/Boddy grazer series (Zenodo 14931953, 10.2 GB)" -d "Hf/Pi/Pv/Rb zips; CC BY 4.0. Also clone github.com/aguilart/FungalGrazers. Fallback anchor.")
D6=$(sub "$E1" -l data "Download Fricker network dataset (Zenodo 5725751, 3.2 GB)" -d "Check whether any series are time-resolved or snapshots only.")
D7=$(sub "$E1" -l data "Download Podospora movies (Figshare 22666138, 9897056, 10626968)" -d "Verify licence + size + format on each item before use.")
D8=$(sub "$E1" -p low -l data "Download Aureobasidium time-lapse (Zenodo 14001310)")
D9=$(sub "$E1" -p low -l data "Download Physarum scanner set (Zenodo 22868368) as publishing template")
for d in $D1 $D2 $D3 $D4 $D5 $D6 $D7 $D8 $D9; do blk "$d" "$LAYOUT"; blk "$d" "$DISK"; done

# ── E2 Phase 1 triage / gate ─────────────────────────────────────────
E2=$(epic -p critical "E2: MyceliumSeg metadata triage (go/no-go gate)" -d "PLAN.md Phase 1. Output: data/metadata.csv + gate decision A/B/C.")
A1=$(sub "$E2" -p critical -l assumption "ASSUMPTION: MyceliumSeg dish ID + timestamp are recoverable" -d "If falsified → gate C: cross-sectional model only, or switch anchor to Fricker grazer (D5). Use 'chainlink cascade' to see downstream impact.")
T1=$(sub "$E2" -l triage "EXIF scan (DateTimeOriginal etc.) on labeled + probe shard")
T2=$(sub "$E2" -l triage "Check whether sequential file numbering follows time within dish (thumbnail strips)")
T3=$(sub "$E2" -l triage "Recover dish tracks (inoculum position, lid marks, embedding clustering)")
T4=$(sub "$E2" -l triage "Check whether per-image stage / sclerotium colour labels ship in the release")
T5=$(sub "$E2" -l triage "Calibrate px → mm using 90-mm dish rim")
GATE=$(sub "$E2" -p critical -l decision "Record gate decision (A/B/C) in notes/gate.md")
for t in $T1 $T2 $T3 $T4 $T5; do blk "$t" "$D1"; blk "$t" "$D2"; blk "$t" "$D3"; blk "$GATE" "$t"; done
blk "$A1" "$GATE"; blk "$D4" "$GATE"

# ── E3 Measurement pipeline ──────────────────────────────────────────
E3=$(epic "E3: Measurement pipeline" -d "PLAN.md Phase 2. Masks → area, radius, roughness, density + measurement-error model.")
M1=$(sub "$E3" -l measurement "Reproduce SegFormer baseline (MyceliumSeg-benchmark)")
M2=$(sub "$E3" -l measurement "Inference on unlabeled images → masks")
M3=$(sub "$E3" -l measurement "Shape metrics (area, r_eq, perimeter, roughness, density) → data/measurements.csv")
M4=$(sub "$E3" -p high -l measurement "Measurement-error model: bias + SD of area/radius on test set")
blk "$M1" "$D2"; blk "$M1" "$PY_ENV"; blk "$M2" "$M1"; blk "$M2" "$D4"; blk "$M3" "$M2"; blk "$M3" "$T5"; blk "$M4" "$M1"

# ── E4 Estimand & DAG ────────────────────────────────────────────────
E4=$(epic "E4: Estimand & DAG" -d "PLAN.md Phase 3. Primary estimand: effect of 15 vs 25C on G. lucidum radial extension rate.")
G1=$(sub "$E4" -l causal "Write DAG in R/dag.R; check adjustmentSets for T→k")
G2=$(sub "$E4" -l causal "Count G. lucidum 15C dishes/images (power for primary estimand)")
A3=$(sub "$E4" -l assumption "ASSUMPTION: 'distinctive edge morphology' image selection is not time-dependent" -d "If falsified, growth-curve shape is biased; need selection model.")
blk "$G1" "$R_ENV"; blk "$G2" "$GATE"; blk "$A3" "$T2"

# ── E5 Generative growth model ───────────────────────────────────────
E5=$(epic "E5: Generative growth model" -d "PLAN.md Phase 4. Lag → linear radial extension → saturation at 45 mm; varying effects per dish.")
A2=$(sub "$E5" -p high -l assumption "ASSUMPTION: growth form lag → constant extension → saturation (Trinci) is invariant across datasets" -d "Core transport hypothesis. If falsified, revisit transport ladder.")
S1=$(sub "$E5" -l model "R/sim.R: simulate dishes from known parameters")
S2=$(sub "$E5" -l model "Prior predictive checks (0–45 mm, realistic days)")
S3=$(sub "$E5" -l model "Parameter recovery: fit ulam to simulated data")
S4=$(sub "$E5" -p high -l model "Fit real MyceliumSeg data; diagnostics + posterior predictive per dish")
S5=$(sub "$E5" -l model "Compare alternatives (logistic, Gompertz, ODE) with PSIS/WAIC")
blk "$S1" "$G1"; blk "$S2" "$S1"; blk "$S3" "$S2"
blk "$S4" "$S3"; blk "$S4" "$M3"; blk "$S4" "$M4"; blk "$S4" "$A1"; blk "$S4" "$G2"
blk "$S5" "$S4"; blk "$A2" "$S5"

# ── E6 Transport ─────────────────────────────────────────────────────
E6=$(epic "E6: Transport to other datasets" -d "PLAN.md Phase 5 ladder. Score by out-of-sample predictive error using pooled priors.")
X1=$(sub "$E6" -l transport "Hold out GS/PO/TS within MyceliumSeg; predict with species partial pooling")
X2=$(sub "$E6" -l transport "Fricker grazer: refit growth form + estimate randomized grazer effect")
X3=$(sub "$E6" -l transport "Fricker network: species-level variation prior (if time-resolved)")
X4=$(sub "$E6" -l transport "Podospora: growth form at hyphal scale + multiplicative nutrient (M2 vs M0) effect")
X5=$(sub "$E6" -p low -l transport "MDPI 110-species set (if access granted)")
blk "$X1" "$S4"; blk "$X2" "$S4"; blk "$X2" "$D5"; blk "$X3" "$S4"; blk "$X3" "$D6"
blk "$X4" "$S4"; blk "$X4" "$D7"; blk "$X5" "$S4"
for x in $X1 $X2 $X3 $X4 $X5; do blk "$x" "$A2"; done

# ── E7 Own rig experiment ────────────────────────────────────────────
E7=$(epic -p low "E7: Own-rig intervention experiment" -d "Break species×medium confound with RPi timelapse (archive/mycelium-workspace). Publish on Zenodo.")
R1=$(sub "$E7" -l rig "Design: species × medium (e.g. oyster on PDA and MYG), randomized positions, temp logging")
R2=$(sub "$E7" -l rig "Rig calibration: fixed lighting, interval, px→mm")
R3=$(sub "$E7" -l rig "Pre-register predicted rate ratio from E5")
R4=$(sub "$E7" -l rig "Run experiment + publish (OME-Zarr, CC BY 4.0, Physarum template)")
blk "$R3" "$S4"; blk "$R4" "$R1"; blk "$R4" "$R2"; blk "$R4" "$R3"

# Assumption links (drive `chainlink cascade` / `falsify`)
asm() { chainlink issue relate "$1" "$2" -t assumption >/dev/null; }
for i in $S4 $G2 $S5 $X1 $R3; do asm "$A1" "$i"; done
for i in $X1 $X2 $X3 $X4 $X5 $R3 $R4; do asm "$A2" "$i"; done
for i in $S4 $S5; do asm "$A3" "$i"; done

echo "Seeded. Epics: $E0 $E1 $E2 $E3 $E4 $E5 $E6 $E7"
