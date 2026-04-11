# Network-Conditioned Factor Predictability

This repository contains the research artifacts for **"ML in Finance - Network-Conditioned Factor Predictability (MATH 279)"**.

Primary paper artifacts:
- `reports/ML in Finance - Network-Conditioned Factor Predictability (MATH 279).pdf`
- `docs/ML in Finance - Network-Conditioned Factor Predictability (MATH 279).pptx`

## Paper focus

The paper asks whether ETF-linked signals predict stock returns in the cross section, and whether that predictability changes with market-wide synchronization (co-trading intensity).

Core questions:
1. **ETF predictability:** Do ETF returns predict stock returns cross-sectionally?
2. **State dependence:** Is predictability different on high-synchronization vs low-synchronization days?
3. **Timing/sign structure:** Does predictability appear as continuation or reversal, and through overnight vs intraday channels?

## Research design in the paper

### 1) Data and return construction
- Sample period is rebuilt from **2000–2023** using adjusted prices.
- Returns are decomposed into:
  - close-to-close,
  - close-to-open (overnight),
  - open-to-close (intraday).
- HAR-style aggregations are built for stock and ETF returns across horizons.

### 2) Synchronization measure
- Start from standardized stock-level abnormal volume shocks.
- Keep unusually large shocks.
- Aggregate squared shocks across stocks into a market-wide synchronization index.
- Define **high synchronization** as top-quantile days (top 20% in the paper slides).

### 3) Benchmark model logic
- Target variable: stock close-to-close return.
- Predictors: stock HAR controls + ETF HAR predictors.
- Key interaction: ETF predictors × high-synchronization indicator.
- Interpretation emphasizes sign/magnitude patterns and cross-ETF consistency.

## Main findings emphasized in the paper

From the paper/presentation narrative:
- ETF predictability is **state-dependent**.
- For major equity ETFs, high synchronization is associated with a short-horizon shift from weak continuation toward stronger reversal.
- Weekly/monthly horizons become relatively more positive in high-synchronization regimes.
- Timing decomposition indicates distinct overnight and intraday channels.
- TLT behavior is qualitatively different from equity ETF patterns.

## Repository guide (paper-oriented)

### Documents
- `reports/ML in Finance - Network-Conditioned Factor Predictability (MATH 279).pdf`: primary write-up.
- `docs/ML in Finance - Network-Conditioned Factor Predictability (MATH 279).pptx`: presentation version of the same research.
- `reports/Math_279_Research_Proposal.pdf`: proposal-stage framing.
- `reports/William_Yao_Math_279_Midterm_Progress_Report.pdf`: midterm progress report.

### Methods/context notes
- `docs/project-overview.md`: concise framing.
- `docs/methods.md`: implementation summary tied to notebook artifacts.

### Code/notebooks
- `notebooks/MATH279Project.ipynb`: main course notebook lineage.
- `notebooks/MATH279Project_priority4_6_integrated.ipynb`: integrated analysis notebook containing later-stage action-plan blocks.
- `src/run_alphamark_benchmark.py`: benchmark pipeline entry point used for in-sample exports.

### Data/results tables
- `data/`: compact long-format regression outputs and summary tables.
- `results/tables/` and `results/figures/`: export-ready tables/figures used in reporting.

## How to read this repository for the paper

If your goal is to understand the paper quickly:
1. Read the final paper PDF in `reports/`.
2. Use the slide deck in `docs/` for concise narrative and headline results.
3. Use `docs/methods.md` + notebooks for implementation details.
4. Use `data/` and `results/tables/` for table-level verification and figure regeneration.

## Reproducibility boundaries

This repository tracks compact analysis artifacts, but large local datasets and generated trees are intentionally excluded from version control. As a result:
- you can validate summary-level findings and regenerate many reported tables/figures,
- full from-scratch reconstruction requires local raw inputs and full notebook dependency chain.

---

If you are citing this project, treat the **paper PDF** as the canonical research narrative and use notebooks/tables as supporting implementation evidence.
