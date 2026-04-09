# Inventory

## Important Files Found

- `src/run_alphamark_benchmark.py`
- `notebooks/MATH279Project.ipynb`
- `notebooks/MATH279Project_priority4_6_integrated.ipynb`
- `reports/Math_279_Research_Proposal.pdf`
- `reports/William_Yao_Math_279_Midterm_Progress_Report.pdf`
- `reports/ML in Finance - Network-Conditioned Factor Predictability (MATH 279).pdf`
- `docs/ML in Finance - Network-Conditioned Factor Predictability (MATH 279).pptx`
- `results/figures/priority4_*`, `results/figures/priority5_*`, and `results/figures/priority6_*` slide-ready PNG outputs
- `results/tables/priority4_*`, `results/tables/priority5_*`, and `results/tables/priority6_*` tabular result CSV exports
- `data/manifest.csv` and lightweight non-result input data retained for the first upload

## Files Moved

- Top-level Python code moved to `src/`
- Top-level notebooks moved to `notebooks/`
- Proposal, progress report, final report PDF, and notebook guide moved to `reports/`
- Slide deck moved to `docs/`
- Lightweight PNG slide outputs moved from `analysis_outputs/priority4_6/` to `results/figures/`
- Companion result CSV exports moved from `analysis_outputs/priority4_6/` to `results/tables/`
- `data/` kept for lightweight non-result inputs and manifests
- `_external/` moved to `archive/external/_external/`
- `scribe_notes_work/` moved to `archive/supporting/scribe_notes_work/`
- `presentation_media_extract/` moved to `archive/supporting/presentation_media_extract/`

## Files Left Ambiguous

- `analysis_outputs/priority4_6/alphamark_manifest.json` remains local with ignored outputs and may or may not belong in tracked project metadata later.
- `analysis_outputs/priority4_6/priority6_portfolio_daily_returns.csv` remains local because it is materially larger than the other retained CSV outputs.
- Archived AlphaMark materials under `archive/external/_external/alphamark/` may be either useful supporting code or vendored third-party content, and should be reviewed before any deeper cleanup.
- Archived lecture and scribe-note materials under `archive/supporting/scribe_notes_work/` appear unrelated to the core repository story and should be reviewed before future publishing decisions.

## Possible Duplicates Or Overlap

- `archive/external/_external/alphamark/output/Quantile_Combined_Report.pdf`
- `analysis_outputs/alphamark_output_smoke_2023/Quantile_Combined_Report.pdf`
- `crsp_outputs/core_research/*`
- `crsp_outputs_final/core_research/*`
- `notebooks/MATH279Project.ipynb`
- `notebooks/MATH279Project_priority4_6_integrated.ipynb`
- `archive/supporting/presentation_media_extract/*` likely derives from the slide deck in `docs/`

## Large Local-Only Content Excluded From Git

- `analysis_outputs/`
- `crsp_outputs/`
- `crsp_outputs_final/`
- `parquet/`
- `.pkl` and `.parquet` artifacts generally

## Recommended Next Cleanup Steps

- Review archived material and decide whether it belongs in the main repository, a separate repository, or nowhere public.
- Add a reproducibility note describing required data sources, environment setup, and execution order.
- Confirm which notebook is the primary analysis notebook and whether one should be retired or renamed later.
- Review overlapping CRSP output folders and designate a single canonical output location.
- Maintain the convention: `results/tables/` for result exports and `data/` for non-result inputs/manifests.
