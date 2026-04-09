# Project overview

## Research framing
This repository studies whether stock-level return predictability changes with ETF-linked synchronization states and whether any such conditioning remains useful outside the estimation sample. The work is organized as an empirical progression from in-sample pooled regressions to out-of-sample (OOS) forecasting and then to long-short portfolio evaluation.

The framing in the existing project materials is consistent with a three-layer question:
1. **Statistical relation**: Do ETF-linked predictors load differently in lower- vs higher-synchronization states?
2. **Generalization**: Do those state-conditioned predictors improve OOS ranking metrics relative to simpler HAR baselines?
3. **Economic relevance**: Do forecast differences translate into gross and net long-short spread differences once turnover-based frictions are applied?

## Core research question
A concise statement of the question is:

> Does conditioning ETF predictor effects on synchronization regimes improve stock-level predictability in ways that are both out-of-sample robust and economically meaningful?

This is implemented in the repository through benchmark synchronization specifications, split timing views (CC/CO/OC), and a matched synchronization extension before OOS and portfolio checks.

## Claimed contribution (conservative)
Based on the tracked artifacts, the contribution is best stated as a **reproducible workflow contribution** rather than a final causal claim:

- A unified notebook workflow links baseline estimation, synchronization timing decomposition, matched synchronization checks, OOS forecasts, and portfolio summaries.
- Lightweight figure/table artifacts are versioned in `figures/` and `data/` to preserve interpretable outputs even when large raw inputs and intermediate trees are not in git.
- Priority-structured outputs (`priority4`, `priority5`, `priority6`) create a transparent mapping from statistical evidence to forecasting and then to economic evaluation.

## Scope boundaries
The repository currently supports interpretation of **reported model outputs**, not full from-scratch reconstruction from raw local storage, because large source datasets and some generated artifacts are explicitly excluded from version control.
