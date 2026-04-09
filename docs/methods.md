# Methods summary

This document summarizes the methods implemented across the integrated notebook workflow and retained CSV/figure outputs.

## 1) HAR setup
The forecasting layer uses a HAR-style predictor construction for both stock-level and ETF-level return signals. In implementation terms, the workflow builds lagged features before running expanding-window prediction loops. The goal is to provide a strong autoregressive baseline and then test whether ETF-linked conditioning terms add incremental value.

## 2) ETF predictors and synchronization interactions
The model families in retained forecast summaries indicate three broad classes:
- `har_only`
- `etf_har`
- synchronization-conditioned variants such as `sync_etf_har` and, where available, `matched_sync_etf_har`

Synchronization interactions are implemented by allowing ETF-linked effects to vary by regime, rather than imposing a single unconditional coefficient profile.

## 3) CC / CO / OC decomposition
The in-sample design includes split timing components labeled CC, CO, and OC. In practical terms, this decomposition separates return windows so that total ETF-linked effects can be interpreted across distinct timing segments instead of only at the daily aggregate level.

Operationally in the tracked outputs, this appears as separate long-format regression result files for CC/CO/OC splits and related synchronization comparison tables.

## 4) Matched synchronization extension
Beyond a benchmark synchronization setup, the workflow includes a matched-universe extension intended to align synchronization measures with ETF-specific stock universes. The purpose is to test whether a more economically aligned synchronization map changes estimated effects or forecast behavior relative to benchmark synchronization.

## 5) OOS forecasting design
The OOS component is implemented as an expanding-window forecasting loop:
- build date-t design matrices from lagged predictors,
- estimate/refit on available history,
- produce forward predictions cross-sectionally,
- aggregate forecast diagnostics (e.g., RMSE and daily IC summaries).

This setup is intended to reduce in-sample overstatement by evaluating predictions only on data not used at each step of model fitting.

## 6) Portfolio evaluation
Forecasts are converted into long-short portfolios by cross-sectional ranking (top-decile minus bottom-decile convention in the project guide), with turnover tracking and cost-adjusted returns. Retained summaries report both gross and net metrics, including mean returns, Sharpe-style summaries, turnover, and cumulative performance fields.

## 7) Interpretation discipline
Given the repository state, method interpretation should remain conservative:
- treat outputs as evidence for comparative model behavior under the recorded pipeline,
- avoid strong structural or causal claims,
- emphasize consistency across priority layers (in-sample decomposition -> OOS forecast quality -> portfolio-level consequences).
