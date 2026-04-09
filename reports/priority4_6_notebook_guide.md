
# Priority 4–6 notebook integration guide

This guide explains how the new `priority_4_6_pipeline.py` logic was organized into the notebook and what each new block is doing.

## Where the new section was inserted

In the integrated notebook, the new section is inserted **immediately after Extension 2 (ETF-Matched Synchronization)** and **before** the legacy compatibility cells.

That placement is intentional:

1. your notebook first builds the core data objects,
2. then estimates the benchmark / split / matched-sync regressions,
3. then moves naturally into:
   - Priority 4 cross-ETF summaries,
   - Priority 5 out-of-sample forecasting,
   - Priority 6 portfolio construction.

So the empirical flow now reads like a paper:
**data -> in-sample evidence -> cross-ETF comparison -> OOS forecasting -> portfolio value**.

## New notebook blocks

### 1. Priorities 4–6 overview markdown
This is the framing cell. It tells the reader:
- why this section exists,
- how it relates to your completed work,
- what each priority adds.

Use this as the transition from “I have estimated the models” to “now I compare and evaluate them”.

### 2. Path and configuration cell
This cell defines:
- `CRSP_OUTPUTS`
- `PROJECT_ROOT`
- `ANALYSIS_ROOT`
- `PRIORITY_ROOT`
- `ALPHAMARK_INPUT_ROOT`

This keeps all new outputs in a dedicated folder tree instead of mixing them with the earlier raw regression exports.

### 3. Priority 4 helper and execution cells
These cells convert the existing saved regression outputs into:
- benchmark total-effect tables,
- timing decomposition tables,
- matched-sync overlap tables,
- paper-ready heatmaps.

Conceptually:

- the long regression files are good for estimation,
- but the heatmaps and pivoted summary tables are what you actually want for interpretation and writing.

### 4. Priority 5–6 model architecture block
This section introduces the forecasting layer.

Core objects:
- `StrategySpec`: stores the ETF, model name, active regressors, and whether the sync regime is matched or benchmark.
- `build_sync_measure`: reconstructs the synchronization indicator.
- `build_matched_universe_map`: maps each ETF to a more economically aligned stock universe.
- `build_design_matrix_for_day`: forms the daily cross-sectional design matrix.
- `compute_invvol_betsize`: creates a volatility-scaled portfolio sizing control.

### 5. Main OOS pipeline cell
This is the main engine.

It does five things:

1. builds HAR predictors for stocks and ETFs,
2. creates benchmark and matched synchronization regimes,
3. runs an expanding-window forecasting loop,
4. stores daily model forecasts,
5. converts forecasts into long-short decile portfolio returns.

This is the part that operationalizes your “does the signal survive out of sample?” question.

### 6. Summary / interpretation cells
These cells:
- display the forecast summary table,
- display the portfolio summary table,
- create paper-ready pivots for:
  - daily IC by ETF/model,
  - net Sharpe by ETF/model.

These are the tables you will likely cite in the final paper.

---

## How the code works, in plain language

## Priority 4
The notebook takes your saved regression outputs and rewrites them into more interpretable objects.

Example:
- the benchmark model estimates `etf_d` and `etf_d_high`.
- low-sync effect is just `etf_d`.
- high-sync effect is `etf_d + etf_d_high`.

That same logic is repeated for weekly and monthly horizons and for each ETF.

So Priority 4 is mainly a **post-processing and comparison layer**.

## Priority 5
For forecasting, the notebook moves one day at a time.

On each date:
- it looks at stock-level realized returns that day,
- uses lagged HAR and ETF variables to build the regressor matrix,
- updates cumulative `X'X` and `X'y`,
- periodically refits coefficients,
- then generates predicted returns for all eligible stocks.

This is an **expanding-window OOS setup**, which is much more appropriate for your paper than reusing full-sample estimates.

## Priority 6
Once a day’s predicted returns are available:
- stocks are ranked cross-sectionally,
- top decile goes long,
- bottom decile goes short,
- weights are equal within long and short legs,
- turnover is computed relative to yesterday’s weights,
- net return subtracts transaction costs.

So Priority 6 asks the economic question:
**does the synchronization-conditioned signal have trading value after costs?**

---

## Why this structure is good for your final paper

This organization cleanly separates:

### earlier notebook sections
- data engineering,
- return construction,
- synchronization measurement,
- in-sample pooled regressions.

### new notebook sections
- cross-ETF comparison,
- OOS validation,
- economic value.

That mirrors the likely paper structure very well:
1. Data and variable construction
2. Baseline in-sample evidence
3. Timing decomposition
4. Matched synchronization extension
5. Out-of-sample forecasting
6. Portfolio implications
7. Robustness and interpretation

---

## Recommended narrative in the paper

When writing up the new section:

### Priority 4
Frame this as:
- “Which ETFs show the strongest state dependence?”
- “Does matched synchronization sharpen the economic alignment story?”

### Priority 5
Frame this as:
- “Do the synchronization-conditioned ETF predictors add genuine out-of-sample ranking power over HAR alone?”

### Priority 6
Frame this as:
- “Does predictive ranking translate into economically meaningful long-short spreads after trading frictions?”

---

## Files produced by the integrated notebook

The new section writes outputs to `analysis_outputs/priority4_6/`, including:

- `priority4_benchmark_total_effects.csv`
- `priority4_timing_decomposition_summary.csv`
- `priority4_matched_sync_total_effects.csv`
- `priority4_matched_sync_overlap_metrics.csv`
- `priority5_oos_forecast_summary.csv`
- `priority6_portfolio_daily_returns.csv`
- `priority6_portfolio_summary.csv`
- `priority5_daily_ic_pivot.csv`
- `priority6_net_sharpe_pivot.csv`

and heatmaps:
- `priority4_benchmark_total_effects.png`
- `priority4_timing_decomposition.png`
- `priority4_matched_sync_overlap_metrics.png`

---

## One practical note

The integrated notebook assumes the earlier notebook cells have already created:
- `ret_cc`, `ret_co`, `ret_oc`
- `vol`
- `etf_ret_cc`
- `stock_meta`
- `make_har_panel`
- the saved CSV outputs from the benchmark / split / matched-sync sections

So this new section is meant to be run **after** your existing estimation blocks, not as a standalone notebook entry point.
