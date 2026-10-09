# Data

This directory is where datasets live locally. Its contents (other than this file) are not
committed; `.gitignore` excludes everything else here.

## MLVS-PT

The primary measured dataset used throughout Chapters 3, 5, 6, and 7. It is an unpublished
project dataset with no DOI. **Available from the author on reasonable request.** Once obtained,
place it at `data/MLVS-PT.parquet`.

## SPS-UK

A public dataset archived on Zenodo, used as the second dataset throughout the same chapters.
DOI: `10.5281/zenodo.5500457`. The Zenodo record holds six raw CSVs (weather, PV, and demand,
each split train/test), not a single assembled file.
`twiga_thesis.datasets.fetch_sps_uk_raw()` downloads these six CSVs automatically into
`data/SPS-UK_raw/`. **Open gap:** there is no verified, reproducible script in this repository
that merges those raw CSVs into the single net-load parquet (`data/SPS-UK.parquet`) that
`load_dataset` expects; the parquet used in the thesis experiments was assembled once, outside of
any committed pipeline. Reconstructing that merge step is noted as follow-up work, not assumed
to be solved by this function.

## DSS-EU, DSS-UK, DSS-US, MLVS-PT-3PHASE (Chapter 7 only)

Chapter 7's multi-target experiments additionally use several low-voltage power-flow records.
`DSS-EU` and `DSS-UK` are built from scenario files in a separate, unpublished repository
(`PhD-Thesis-Anthony`); `DSS-US` is an OpenDSS simulation produced in the companion
`ml-energy-disaggregation` simulator repository. None of these are redistributed here.

Scripts that need them call `twiga_thesis.paths.external_data_dir()`, which resolves the
`EXTERNAL_DATA_DIR` environment variable (the root of a local `ml-energy-disaggregation` checkout)
and raises a clear error if it is not set. See `chapters/ch7_multitarget/README.md` and
`chapters/ch7_multitarget/FPQR_GATING_README.md` for the exact steps to obtain or regenerate each
record.

## Synthetic data

For chapters or experiments that don't require a specific measured dataset, `twiga.core.data.
generate_synthetic_netload` (shipped in the `twiga` package itself) produces a configurable
synthetic net-load series with no external dependency at all.
