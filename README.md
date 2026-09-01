# Analysis code for the Stack-NNLS asthma prediction manuscript

This repository contains the analysis scripts corresponding to the manuscript, together with a separate calibration analysis for the canonical multi-view Stack-NNLS model. The manuscript scripts are preserved byte-for-byte from the validated run used for the current figures and tables.

## Contents

- `scripts/stack_views_nnls.py` — development-cohort preprocessing, view-specific base learners, OOF predictions, and logit-space non-negative stacking.
- `scripts/external_validate_stack_views_nnls_delong.py` — independent external validation using the fixed development-cohort model procedure.
- `scripts/compare_smote_vs_no_smote.py` — sensitivity analysis comparing resampling strategies.
- `scripts/make_figure_collages.py` — figure-collage utility used by the analysis workflow.
- `scripts/calibration_analysis.py` — calibration-in-the-large, calibration slope, Brier score, ECE/MCE, bootstrap confidence intervals, and reliability plot for internal and external Stack-NNLS probabilities.
- `config/best_params.json` and `scripts/best_params.json` — model hyperparameters used by the manuscript pipeline.
- `results/calibration_results/` — aggregate calibration outputs generated from the canonical prediction artefacts.

No patient-level workbook, prediction cache, or other individual-level data are included in this repository. Keep the input workbooks outside version control and point the scripts to the directory containing the validated analysis artefacts.

## Reproduction

Install the pinned Python dependencies from `requirements.txt`. The manuscript scripts expect their input files (including `train_pruned_ratio_0p04.xlsx`, `External_2.0.xlsx`, and the relevant cache files) in the analysis directory. For the validated run, execute from that directory:

```text
python scripts/stack_views_nnls.py
python scripts/external_validate_stack_views_nnls_delong.py
python scripts/compare_smote_vs_no_smote.py
```

The calibration analysis can reuse the canonical internal OOF cache and external prediction workbook without refitting the primary model:

```text
python scripts/calibration_analysis.py --analysis-dir <validated-analysis-directory> --output-dir results/calibration_results --bootstrap-rounds 2000 --seed 42
```

Use `--write-predictions` only for a private local audit; it is intentionally disabled by default so that patient-level probabilities are not written to the repository output.

## Calibration result provenance

The checked-in calibration result was generated on 2026-09-01 from the canonical internal prediction cache and the external `proba_stack_nnls` prediction workbook associated with the manuscript run, using 2,000 stratified bootstrap resamples and seed 42. The primary model was not refit or recalibrated. See `results/calibration_results/calibration_report.md` for the metrics, confidence intervals, source files, and script hashes.

The reported calibration analysis is an additional model-performance assessment. It does not replace the manuscript's prespecified discrimination, threshold, decision-curve, or external-validation analyses.

