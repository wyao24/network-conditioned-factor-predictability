# Data directory guide

## What data types are included here
The `data/` directory stores **compact tabular artifacts** used for interpretation and reporting, including:

- A manifest-style inventory (`manifest.csv`) with per-date status/row counts for underlying daily source files.
- Long-format pooled regression outputs for benchmark, split timing (CC/CO/OC), synchronization comparisons, and matched synchronization summaries.
- Priority-level summaries for OOS forecasting and portfolio evaluation.

These files are primarily analysis-ready summaries rather than raw market-history dumps.

## What is absent from version control
The repository intentionally does **not** include large local-only data and generated trees (as documented in repository-level notes), including directories such as:
- `analysis_outputs/`
- `crsp_outputs/`
- `crsp_outputs_final/`
- `parquet/`
- large `.pkl` / `.parquet` artifacts

As a result, this folder should be read as a compact results-and-intermediate layer, not a complete raw-data package.

## Reconstruction requirements
To reconstruct the full pipeline from raw inputs, you will need:
1. Access to the underlying daily stock/ETF input files referenced by the manifest and notebook workflow.
2. The integrated notebook execution path that builds core return panels, synchronization measures, HAR features, and model-specific design matrices.
3. The local output structure expected by the priority 4-6 workflow (including larger untracked daily portfolio-return exports).

In short: tracked files are sufficient for **summary-level replication checks** and figure/table regeneration, but not by themselves for full end-to-end raw-data reprocessing.
