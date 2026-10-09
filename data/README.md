# Data

This directory is where datasets live locally. Its contents (other than this file) are not
committed; `.gitignore` excludes everything else here.

## MLVS-PT

The primary measured dataset used throughout Chapters 3, 5, 6, and 7. It is an unpublished
project dataset with no DOI. **Available from the author on reasonable request.** Once obtained,
place it at `data/MLVS-PT.parquet`.

## SPS-UK

A public dataset archived on Zenodo (Chapter 2, Table 2.1), used as the second dataset throughout
Chapters 3, 5, 6, and 7. DOI: `10.5281/zenodo.5500457`. The thesis calls it "SPS-UK"; `twiga`'s own
data loader (`twiga/utils/data_loading.py`) uses the internal dataset key `"MLVS-UK"` and expects
it at `data/MLVS-UK.parquet` -- the naming mismatch between the thesis text and the code is worth
knowing about if you're searching for it.

The Zenodo record holds six raw CSVs (weather at hourly resolution across six stations, PV and
demand at 30-minute resolution, each split train/test), not a single assembled file:

```python
from twiga_thesis.datasets import fetch_sps_uk_raw, build_sps_uk_parquet

raw_dir = fetch_sps_uk_raw()  # downloads the 6 CSVs to data/SPS-UK_raw/
build_sps_uk_parquet(raw_dir)  # writes data/MLVS-UK.parquet
```

`build_sps_uk_parquet` follows Chapter 2's own description of the merge (net load = demand minus
PV generation, both in MW; weather linearly interpolated from 60 to 30 minutes), verified
end-to-end against the real Zenodo files (47,039 rows spanning the documented Nov 2017-Jul 2020
range, 36 rows with a small PV/demand gap). **One assumption is not specified in the thesis text
and could not be verified against the original parquet** (which could not be located in this
project for comparison): the six weather stations' temperature and irradiance readings are
averaged into single `Temperature`/`Ghi` columns, since `load_dataset` expects exactly one of
each. Treat the resulting dataset as a verified-plausible reconstruction, not a byte-for-byte match
to what produced the thesis's reported numbers, until checked against the original file.

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
