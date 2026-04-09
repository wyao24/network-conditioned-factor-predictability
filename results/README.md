# Results directory guide

This directory documents how result artifacts are organized and how they map to project priorities. Primary stored outputs currently live in `figures/` and `data/`.

## Priority mapping

### Priority 4: in-sample comparative evidence
Typical artifacts:
- benchmark total-effects summaries
- timing decomposition summaries
- matched synchronization overlap/effect summaries
- companion heatmaps

Stored examples are present in `figures/priority4_*` (PNG + CSV) and related long-form tables in `data/`.

### Priority 5: OOS forecast evaluation
Typical artifacts:
- forecast summary tables (e.g., RMSE and daily IC aggregates)
- slide-ready forecast and RMSE comparison plots/tables

Stored examples are present in `data/priority5_oos_forecast_summary.csv` and `figures/priority5_*` artifacts.

### Priority 6: portfolio-level evaluation
Typical artifacts:
- gross/net long-short performance summaries
- turnover and cumulative return fields
- slide-ready portfolio summary visuals/tables

Stored examples are present in `data/priority6_portfolio_summary.csv` and `figures/priority6_*` artifacts.

## How to use this mapping in writeups
- Use Priority 4 outputs to establish whether state-conditioned effects differ in-sample.
- Use Priority 5 outputs to evaluate whether ranking quality persists OOS.
- Use Priority 6 outputs to check whether predictive differences survive trading frictions at the portfolio level.

This sequencing supports a conservative evidence chain from statistical relation to economic relevance.
