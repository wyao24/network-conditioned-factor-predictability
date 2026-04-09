# Network-Conditioned Factor Predictability

## Overview
This repository documents an integrated Priority 4–6 analysis layer for a MATH 279 project, framed as a progression from in-sample model evidence to cross-ETF comparison, out-of-sample (OOS) forecasting, and portfolio-value evaluation. The notebook guide describes this flow as: data -> in-sample evidence -> cross-ETF comparison -> OOS forecasting -> portfolio value.

## Research Question
The guide explicitly frames three linked questions:
- Priority 4: Which ETFs show stronger state dependence, and does matched synchronization sharpen economic alignment?
- Priority 5: Do synchronization-conditioned ETF predictors add OOS ranking power over HAR-only baselines?
- Priority 6: Does predictive ranking translate into economically meaningful long-short spreads after trading frictions?

## Methodology
Based on the notebook guide and retained summary outputs:
- Priority 4 post-processes saved regression outputs into total-effect/timing-decomposition summaries and heatmap-ready tables.
- Priority 5 uses an expanding-window OOS forecast loop that builds HAR predictors, forms design matrices, updates cumulative moment objects, refits periodically, and generates stock-level predictions.
- Priority 6 converts forecasts into long-short decile portfolios, computes turnover relative to prior weights, and reports both gross and transaction-cost-adjusted (net) results.

## Repository Structure
This README is intentionally limited to evidenced artifacts from the specified files:
- `reports/priority4_6_notebook_guide.md`: workflow narrative, block-by-block notebook integration notes, and expected Priority 4–6 outputs.
- `data/priority5_oos_forecast_summary.csv`: ETF/model-level OOS forecast metrics (`nobs`, `rmse`, `daily_ic_mean`, `daily_ic_std`).
- `data/priority6_portfolio_summary.csv`: ETF/model-level portfolio performance metrics (gross/net returns, gross/net Sharpe, turnover, cumulative returns, days).

## Current Artifacts
- Priority 5 summary table: `data/priority5_oos_forecast_summary.csv`.
- Priority 6 summary table: `data/priority6_portfolio_summary.csv`.
- Priority 4–6 integration and interpretation guide: `reports/priority4_6_notebook_guide.md`.

## Notebook Roles And Expected Execution Sequence

- **Primary notebook (chosen):** `notebooks/MATH279Project_priority4_6_integrated.ipynb`
- **Supporting notebook:** `notebooks/MATH279Project.ipynb`
- **Role of `src/run_alphamark_benchmark.py`:** script entry point for running the AlphaMark benchmark workflow and exporting the in-sample benchmark/split/matched-sync regression artifacts consumed by the integrated notebook.

Expected sequence:
1. Run in-sample regressions (benchmark/split/matched-sync estimation outputs).
2. Run Priority 4 summary generation (cross-ETF timing and matched-sync summaries/heatmaps).
3. Run Priority 5 out-of-sample forecasting.
4. Run Priority 6 portfolio construction and performance summaries.

Unresolved prerequisites (from `reports/priority4_6_notebook_guide.md`) that should be confirmed before treating the workflow as fully reproducible:
- TODO: Confirm the upstream cells create `ret_cc`, `ret_co`, `ret_oc`.
- TODO: Confirm the upstream cells create `vol`.
- TODO: Confirm the upstream cells create `etf_ret_cc`.
- TODO: Confirm the upstream cells create `stock_meta`.
- TODO: Confirm `make_har_panel` is defined before the priorities 4–6 section.
- TODO: Confirm saved CSV outputs from benchmark/split/matched-sync sections exist before running the integrated priority 4–6 blocks.

TODO (non-speculative):
- TODO: Add a cell-by-cell execution checklist (with cell IDs or section anchors) for the integrated notebook.
- TODO: Add the minimal environment specification (Python version + required packages) used to generate the two retained summary CSVs.
- TODO: Add an explicit mapping from notebook sections to each output file listed in the guide.

## Key Findings
From `data/priority5_oos_forecast_summary.csv`:
- Across all listed ETFs, `har_only` has the lowest RMSE among compared models in this summary table.
- Daily IC means are small in magnitude; most ETF/model combinations are positive, while all shown QQQ variants are slightly negative.

From `data/priority6_portfolio_summary.csv`:
- All listed ETF/model combinations have negative net Sharpe ratios (range approximately -2.58 to -1.32 in this file).
- Average turnover is high (roughly 1.07 to 2.13), and mean net returns are negative across listed strategies.
- Several ETF/model combinations show positive gross Sharpe, but these do not remain positive after transaction-cost adjustment in this summary.

## Limitations
Evidence-constrained limitations from the provided files:
- The guide is procedural/descriptive and does not provide a fully specified standalone run script for Priority 4–6.
- The retained CSVs are aggregate summaries; they do not include the underlying daily panel needed to independently recompute every intermediate step.
- The guide confirms dependency on earlier notebook objects, so Priority 4–6 cannot be validated as an isolated module from the available evidence alone.

## Next Steps
- Confirm and document authoritative execution order for the integrated notebook blocks.
- Publish reproducibility metadata (environment + deterministic run instructions) tied to Priority 5 and Priority 6 outputs.
- Extend reporting to include uncertainty intervals and robustness variants in the same summary format as the retained CSV artifacts.
