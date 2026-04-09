# Network-Conditioned Factor Predictability

This repository contains a first-pass organization of materials for a MATH 279 project on network-conditioned factor predictability in finance. The current upload preserves the main notebooks, benchmark script, reports, slide deck, lightweight figures, and small tabular outputs while keeping large local datasets and generated artifact trees out of version control.

This is an initial repository organization pass rather than a final cleaned research archive. The goal is to make the project uploadable and readable on GitHub without aggressively refactoring code, renaming files, or deleting uncertain material.

## Current Contents

- `src/`: project Python entry point used for the AlphaMark benchmark workflow
- `notebooks/`: project notebooks used for the course work and integrated analysis pass
- `reports/`: proposal, progress report, final project write-up, and notebook guide
- `docs/`: project slide deck
- `results/figures/`: lightweight presentation-ready figure artifacts (PNG)
- `results/tables/`: tabular result exports (CSV) used for slides and summaries
- `data/`: lightweight non-result data and input manifests only
- `archive/`: supporting or less central materials preserved without deleting them

## Notes

- Large local data and generated output directories remain in the working folder but are excluded from git in this first pass.
- Archive content is retained conservatively so existing work is not lost during cleanup.

## TODO

- Review archived materials and decide what should remain in the long-term repository.
- Add clearer documentation for the data pipeline, notebook workflow, and reproducibility steps.
- Decide whether selected large outputs should move to GitHub Releases, cloud storage, or data versioning.
- Clean up duplicate or overlapping outputs after confirming which versions are authoritative.
