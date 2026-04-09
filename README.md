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

## Reproducibility / Running the Project
Known execution-order evidence:
- The guide states Priority 4–6 cells are inserted after Extension 2 and before legacy compatibility cells.
- The guide also states the section is not standalone and assumes earlier notebook cells already created core objects (for example `ret_cc`, `vol`, `etf_ret_cc`, `stock_meta`, `make_har_panel`, and prior regression CSVs).

Uncertainty (explicit):
- The exact end-to-end execution order across all notebook cells is not fully specified in the provided summary files.
- Environment/package requirements and exact command-line entry points for reproducing the same outputs are not fully enumerated in the provided evidence set.

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
