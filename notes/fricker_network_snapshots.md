# Fricker fungal network snapshots: capability assessment (#46)

## Decision

This dataset supports **cross-sectional, study-specific summaries and weak
species-level priors for the authors' network traits**. It supplies no
within-colony change, growth rate, chronology, or intervention effect. Its
network features are not outcomes of the MyceliumSeg image-to-mask model, so
they do not provide a prior for segmentation, area fraction, or the existing
coverage-morphology model.

Do not pool all taxa into one generic fungal growth or morphology prior. The
study combines two distinct observation regimes that align with the taxonomic
groups: microscopic Ascomycota/Zygomycetous hyphae on water agar at 20 °C,
imaged after roughly 10–14 hours at 0.79 µm/pixel; and macroscopic
Basidiomycota cords from beech blocks in compressed soil at 18 °C, imaged
after 8–12 days at 80–155 µm/pixel. The paper analyzes the two scales
separately as well as jointly, using author-defined normalized traits in the
joint comparison. Any species prior is therefore conditional on those image,
substrate, temperature, sampling, segmentation, and graph-construction
protocols. The cultivation and imaging regimes are nested within the taxonomic
groups, so their effects cannot be separated by this dataset.

## Sample audit

The Zenodo 5725751 release staged here contains 33 JPEGs and 33 matching
parameter files, representing 11 coded taxa with three samples each. The
published paper describes 12 taxa and three replicates per strain. The
author-published `all_data.RDS` at the pinned repository commit contains 36
real-network rows and 12 species, but those rows are not a balanced three-per-
species table:

- nine taxa have three rows;
- *Umbelopsis isabellina* has one;
- *Phallus impudicus* and *Resinicium bicolor* have four each.

The row-level identifiers also do not exactly reconcile. The published table
has four IDs absent from the Zenodo parameter archive (`C35(4)_t30`,
`M(5)_t13_c`, `Pi 18C day 8 004`, and `Rb_ctrl1_d8_5`), while the archive has
`M(5)_t13` without an exact table match. That last pair may be a suffix/name
change, but the files do not establish the correspondence. Treat the paper's
balanced-replication statement as its design description, not as verified
balance in the released analysis table.

The scaled table has 36 rows, with 9 Ascomycota, 16 Zygomycetous, and 11
Basidiomycota records. The author RDS has four non-finite `Hyphal_tip_access`
values and `Mycelia_Vol_MST` is NaN in all 36 rows; that field must not be used
as an observed trait. The non-finite values do not affect the available
normalized candidate fields `Mycelia_length_MST`, `Root_eff_l_scaled`,
`Reff_tip_l_scaled`, and `Geff_MST`.

## Modeling scope

The original study reports species-associated network-trait structure and
separate ordinations for the microscopic and macroscopic groups. This supports
descriptive comparisons among those sampled strains under the study's own
measurement pipeline. With only three samples for most taxa, an unmatched
archive/table, and scale and cultivation regime aligned with phylum, it does
not justify a broadly transportable hierarchical prior for unmeasured species
or another image pipeline. At most, use its author-defined normalized network
traits as weak, protocol-specific prior information when modeling comparable
network measurements, with a sensitivity analysis that excludes the unmatched
rows.

There is no model refinement to carry back into the current MyceliumSeg or
time-series work. If a future project uses this dataset quantitatively, first
reconcile the Zenodo images/parameters with the 36-row author table and verify
which IDs belong to the intended replicate set.

## Reproduction

Run from the workspace root in the pinned Nix R environment:

```sh
nix develop --quiet --command Rscript scripts/fricker_network_snapshots.R
```

The script fetches the two author RDS files from a pinned repository commit,
checks their SHA-256 digests, compares IDs and counts with the Zenodo archives,
and writes summaries under `data/derived/fricker_network_snapshots/`.
The exact commit and both expected digests are recorded in
`scripts/fricker_network_snapshots.R`.

## Sources

- Aguilar-Trigueros et al. (2022), [Network traits predict ecological
  strategies in fungi](https://doi.org/10.1038/s43705-021-00085-1).
- [Zenodo dataset 5725751](https://zenodo.org/records/5725751), CC BY 4.0.
- Authors' [analysis and data-assembly repository](https://github.com/aguilart/Fungal_Networks), pinned to the commit and digests recorded in `scripts/fricker_network_snapshots.R`.
