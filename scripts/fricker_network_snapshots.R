#!/usr/bin/env Rscript

# Audit the Zenodo snapshot bundle against the author-published analysis table.
# The author RDS files are downloaded to a temporary directory at a pinned
# repository commit; they are not copied into the project.

root <- normalizePath(getwd(), mustWork = TRUE)
out_dir <- file.path(root, "data", "derived", "fricker_network_snapshots")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

repo_commit <- "df7f55a4fa7fbd76877cf9cbbbf9e4052b682b89"
repo_base <- paste0(
  "https://raw.githubusercontent.com/aguilart/Fungal_Networks/",
  repo_commit,
  "/ISMEJ_C_output/"
)
source_files <- c("all_data.RDS", "all_data_scaled.RDS")
source_sha256 <- c(
  all_data = "ce3e4a6a5c03cd726924f0704f26f5a3688240b89e1b67f22f6fe93c3c87c7a3",
  all_data_scaled = "4d5df3815ab60e891d381e9eabb243bd2b0396bfa067ed4e46b0f17790d844fd"
)

sha256 <- function(path) {
  result <- system2("sha256sum", shQuote(path), stdout = TRUE, stderr = TRUE)
  if (length(result) != 1L || grepl("Error", result[[1L]], fixed = TRUE)) {
    stop("Could not calculate SHA-256 for ", path)
  }
  sub(" .*", "", result[[1L]])
}

read_pinned_rds <- function(filename, temp_dir) {
  path <- file.path(temp_dir, filename)
  utils::download.file(
    paste0(repo_base, filename), path, mode = "wb", quiet = TRUE
  )
  key <- sub("\\.RDS$", "", filename)
  observed <- sha256(path)
  expected <- unname(source_sha256[[key]])
  if (!identical(observed, expected)) {
    stop("Source checksum mismatch for ", filename, ": ", observed)
  }
  readRDS(path)
}

main <- function() {
  temp_dir <- tempfile("fricker-network-source-")
  dir.create(temp_dir)
  on.exit(unlink(temp_dir, recursive = TRUE), add = TRUE)

  all_data <- read_pinned_rds("all_data.RDS", temp_dir)
  scaled <- read_pinned_rds("all_data_scaled.RDS", temp_dir)
  real <- all_data[all_data$Network == "Real", , drop = FALSE]

  if (nrow(real) != nrow(scaled)) {
    stop("The published real-network and scaled tables have different row counts")
  }
  if (!setequal(real$name_col, scaled$name_col)) {
    stop("The published real-network and scaled tables have different sample IDs")
  }
  if (anyDuplicated(real$name_col) || anyDuplicated(scaled$name_col)) {
    stop("Duplicate sample IDs found in published analysis table")
  }

  parameter_zip <- file.path(root, "data", "raw", "fricker_networks", "parameters.zip")
  image_zips <- file.path(
    root,
    "data", "raw", "fricker_networks",
    c("ascomycota.zip", "basidiomycota.zip", "Zygomycetous1.zip",
      "Zygomycetous2.zip", "Zygomycetous3.zip")
  )
  if (!file.exists(parameter_zip) || !all(file.exists(image_zips))) {
    stop("Expected Fricker network Zenodo ZIP archives are missing")
  }

  parameter_paths <- utils::unzip(parameter_zip, list = TRUE)$Name
  parameter_ids <- sub(
    "_param\\.mat$", "", basename(parameter_paths[!grepl("/$", parameter_paths)])
  )
  image_paths <- unlist(lapply(image_zips, function(path) {
    utils::unzip(path, list = TRUE)$Name
  }), use.names = FALSE)
  image_ids <- sub("\\.jpg$", "", basename(image_paths[!grepl("/$", image_paths)]))

  if (!setequal(parameter_ids, image_ids)) {
    stop("Zenodo image and parameter archives do not contain the same sample IDs")
  }

  sample_code <- function(ids) {
    sub("^((DF[0-9]+|C[0-9]+|FOX|M|Pi|Pv|Rb)).*$", "\\1", ids)
  }
  raw_taxon_counts <- as.data.frame(table(sample_code(parameter_ids)))
  names(raw_taxon_counts) <- c("sample_code", "n")
  raw_taxon_counts$n <- as.integer(raw_taxon_counts$n)
  utils::write.csv(
    raw_taxon_counts,
    file.path(out_dir, "zenodo_taxon_counts.csv"),
    row.names = FALSE
  )

  species_counts <- aggregate(
    list(n = scaled$name_col),
    by = list(phylum = scaled$phylum, species = scaled$Species),
    FUN = length
  )
  species_counts <- species_counts[order(species_counts$phylum, species_counts$species), ]
  utils::write.csv(
    species_counts,
    file.path(out_dir, "species_replication.csv"),
    row.names = FALSE
  )

  reconciliation <- data.frame(
    source = c(
      "Zenodo image archives",
      "Zenodo parameter archive",
      "Author analysis table (Network = Real)",
      "Author analysis comparison rows (Network = Resistance_MST)"
    ),
    n_samples = c(
      length(image_ids),
      length(parameter_ids),
      nrow(real),
      sum(all_data$Network == "Resistance_MST")
    ),
    n_taxa = c(
      length(unique(sample_code(image_ids))),
      length(unique(sample_code(parameter_ids))),
      length(unique(scaled$Species)),
      NA_integer_
    ),
    stringsAsFactors = FALSE
  )
  reconciliation$only_in_author_analysis <- c(
    NA_character_, NA_character_,
    paste(sort(setdiff(real$name_col, parameter_ids)), collapse = "; "),
    NA_character_
  )
  reconciliation$only_in_zenodo_archive <- c(
    NA_character_, NA_character_,
    paste(sort(setdiff(parameter_ids, real$name_col)), collapse = "; "),
    NA_character_
  )
  utils::write.csv(
    reconciliation,
    file.path(out_dir, "source_reconciliation.csv"),
    row.names = FALSE,
    na = ""
  )

  numeric_columns <- names(scaled)[vapply(scaled, is.numeric, logical(1))]
  finite_counts <- data.frame(
    feature = numeric_columns,
    n_nonfinite = vapply(
      scaled[numeric_columns], function(x) sum(!is.finite(x)), integer(1)
    ),
    stringsAsFactors = FALSE
  )
  utils::write.csv(
    finite_counts,
    file.path(out_dir, "feature_nonfinite_counts.csv"),
    row.names = FALSE
  )

  metrics <- intersect(
    c("Mycelia_length_MST", "Root_eff_l_scaled", "Reff_tip_l_scaled", "Geff_MST"),
    names(scaled)
  )
  metric_summary <- do.call(rbind, lapply(metrics, function(metric) {
    groups <- split(scaled[[metric]], scaled$phylum)
    do.call(rbind, lapply(names(groups), function(group) {
      values <- groups[[group]]
      values <- values[is.finite(values)]
      data.frame(
        metric = metric,
        phylum = group,
        n = length(values),
        median = if (length(values)) stats::median(values) else NA_real_,
        min = if (length(values)) min(values) else NA_real_,
        max = if (length(values)) max(values) else NA_real_
      )
    }))
  }))
  utils::write.csv(
    metric_summary,
    file.path(out_dir, "author_scaled_metric_ranges.csv"),
    row.names = FALSE,
    na = ""
  )

  summary <- c(
    sprintf("pinned_author_repo_commit=%s", repo_commit),
    sprintf("zenodo_images=%d", length(image_ids)),
    sprintf("zenodo_parameter_files=%d", length(parameter_ids)),
    sprintf("author_real_rows=%d", nrow(real)),
    sprintf("author_real_taxa=%d", length(unique(scaled$Species))),
    sprintf("author_only_ids=%s", paste(sort(setdiff(real$name_col, parameter_ids)), collapse = ";")),
    sprintf("archive_only_ids=%s", paste(sort(setdiff(parameter_ids, real$name_col)), collapse = ";")),
    sprintf("all_nan_Mycelia_Vol_MST=%s", all(is.nan(scaled$Mycelia_Vol_MST)))
  )
  writeLines(summary, file.path(out_dir, "audit_summary.txt"))
  cat(paste(summary, collapse = "\n"), "\n")
}

main()
