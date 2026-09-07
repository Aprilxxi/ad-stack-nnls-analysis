# Multi-view Stack-NNLS analysis code for pediatric asthma prediction

## Overview

This repository contains the analysis code and model configuration used for a multi-view pediatric asthma prediction study. The workflow combines view-specific base learners with a non-negative least-squares (NNLS) stacking layer operating on base-model logits.

This is intentionally a code-only repository. It contains analysis scripts, pinned Python dependencies, the model parameters used by the pipeline, and a header-only Excel template showing the expected input schema. The example workbook contains column headers only and no participant records.

## Repository contents

- `scripts/stack_views_nnls.py` — development-cohort preprocessing, view-specific base learners, five-fold base-model out-of-fold predictions, NNLS stacking, and development assessment.
- `scripts/external_validate_stack_views_nnls_delong.py` — independent external validation using models and decision thresholds derived from the development cohort.
- `scripts/compare_smote_vs_no_smote.py` — sensitivity analysis comparing the pipeline with and without oversampling.
- `scripts/calibration_analysis.py` — calibration-in-the-large, calibration slope, Brier score, ECE, MCE, bootstrap confidence intervals, and reliability curves for the raw Stack-NNLS probabilities.
- `scripts/nested_logistic_recalibration.py` — development-only nested selection and evaluation of logistic recalibration, followed by application of the development-fitted mapping to external predictions.
- `scripts/global_feature_importance_shap.py` — global feature-importance analysis using NNLS-weighted SHAP values on the logit scale.
- `scripts/tune_base_models.py` — optional Optuna-based hyperparameter tuning for the view-specific base learners.
- `scripts/statistical_analysis_table1.py` — descriptive comparison of the development and external cohorts using the primary cohort eligibility boundaries.
- `scripts/make_figure_collages.py` — utility for assembling locally generated figure panels.
- `scripts/best_params.json` — model parameters used by the analysis pipeline; performance scores and tuning logs are intentionally omitted.
- `examples/example_data_headers.xlsx` — header-only input template with no data rows.
- `requirements.txt` — pinned Python dependencies.


## Environment setup

From the repository root:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Activate the virtual environment using the command appropriate for your operating system before running the analyses.

## Preparing input data

Use `examples/example_data_headers.xlsx` only as a schema reference. It contains 49 model-feature headers, the binary outcome header `Asthma`, and the development-cohort screening header `病历记录数量`; it contains no observations and is not itself a runnable dataset.

Create local, access-controlled workbooks with matching column names and place them in the `scripts` directory using these filenames:

```text
scripts/train_pruned_ratio_0p04.xlsx
scripts/External_2.0.xlsx
```

The outcome column `Asthma` must be encoded as `0` or `1`. Column names are case-sensitive. The template uses the 17 merged binary allergen variables expected by the current setting `USE_ENGINEERED_ALLERGEN_FEATURES = False`. The raw allergen-source columns are needed only if that setting is changed and the feature-engineering mapping in `stack_views_nnls.py` is used.

The primary development script applies the following configured eligibility rules:

- age less than or equal to 8 years; and
- when `病历记录数量` is present, a value greater than 5.

The external-validation script applies the age limit of 8 years or younger and does not apply the record-count filter. The external workbook may therefore omit `病历记录数量`. Missing external predictors are aligned to the development feature schema and handled using preprocessing fitted only on development data.

Do not add real clinical workbooks to the repository.

## Running the analyses

Run all commands from the repository root. Generated caches, predictions, tables, and figures are local outputs and are excluded by `.gitignore`.

### 1. Development analysis

```bash
python scripts/stack_views_nnls.py
```

This script generates base-model out-of-fold predictions, pooled NNLS-layer out-of-fold predictions, local caches, metrics, and figures.

### 2. Independent external validation

```bash
python scripts/external_validate_stack_views_nnls_delong.py
```

This script refits each base learner using the complete development cohort, fits the final NNLS meta-model using development base-model out-of-fold logits, and applies the resulting fixed procedure to the external cohort. External outcomes are used only for evaluation.

### 3. Calibration assessment

Run this after the development and external-validation scripts:

```bash
python scripts/calibration_analysis.py \
  --analysis-dir scripts \
  --output-dir local_outputs/calibration \
  --bootstrap-rounds 2000 \
  --seed 42
```

### 4. Nested logistic recalibration

```bash
python scripts/nested_logistic_recalibration.py \
  --analysis-dir scripts \
  --output-dir local_outputs/nested_logistic_recalibration \
  --outer-folds 5 \
  --inner-folds 5 \
  --bootstrap-rounds 2000 \
  --seed 42
```

Do not use `--write-predictions` in a public or shared working copy.

## Additional analyses

Oversampling sensitivity analysis:

```bash
python scripts/compare_smote_vs_no_smote.py
```

Global feature importance:

```bash
mkdir local_outputs
python scripts/global_feature_importance_shap.py \
  --splits 5 \
  --seed 42 \
  --per-model \
  --out local_outputs/global_feature_importance.xlsx
```

Development-versus-external descriptive table:

```bash
python scripts/statistical_analysis_table1.py \
  --base-dir scripts \
  --output-dir local_outputs/table1
```

Optional hyperparameter retuning is not required to run the supplied manuscript configuration. To retune without overwriting it:

```bash
python scripts/tune_base_models.py \
  --models blood_xgb,blood_extratrees,allergen_cat,history_logreg \
  --trials 200 \
  --metric auc \
  --splits 5 \
  --seed 42 \
  --out local_outputs/best_params_retuned.json
```

Figure-panel assembly, after the required source figures have been generated locally:

```bash
python scripts/make_figure_collages.py --figures-dir scripts/figures
```

## Validation design and interpretation

Each base learner produces stratified five-fold out-of-fold probabilities. Preprocessing and oversampling for each base-model fold are fitted using that fold's training partition and then applied to its held-out partition.

For development assessment of the stacking layer, the fixed base-model out-of-fold probability matrix is transformed to clipped logits. An NNLS linear meta-model with non-negative coefficients and an intercept is then cross-fitted over five stratified folds. Predictions from the five held-out meta-layer folds are pooled for development evaluation.

This procedure cross-fits the NNLS meta-layer on an already constructed base-model out-of-fold matrix. It is not a fully nested outer validation of the complete workflow, including feature definition, hyperparameter selection, preprocessing, and base-learner development. Development estimates should therefore be described specifically as pooled five-fold NNLS-layer out-of-fold performance, rather than as outer cross-validation of the entire pipeline.

For independent external validation:

1. the base learners are refitted using all available development data;
2. a separate final NNLS meta-model is fitted using all development base-model out-of-fold logits;
3. preprocessing learned from the development cohort is applied to the external predictors;
4. the fixed model is applied once to the external cohort; and
5. external outcomes are used only to calculate validation metrics.

Classification thresholds applied externally are selected from development out-of-fold predictions and are not re-estimated in the external cohort.

The nested logistic recalibration analysis is a post-hoc probability-calibration analysis. Calibrator selection and fitting use development out-of-fold probabilities only. The final development-fitted mapping is then applied once to the locked external predictions.

## Local outputs and privacy

Several scripts generate caches, predictions, spreadsheets, and figures locally. Some primary scripts also generate patient-level prediction files as part of normal execution. These files may contain sensitive or derived participant-level information.

All generated outputs must remain in protected local storage. They are excluded from version control and must not be committed, uploaded, or shared through this repository.

## Code and data availability

The analysis code, pinned dependencies, model configuration, and a header-only input template are provided in this repository. No clinical data, patient-level predictions, fitted-model caches, tables, figures, or numerical study results are included.

Access to the underlying clinical data may be requested from the corresponding author and is subject to institutional approval, applicable ethics and privacy requirements, and an appropriate data-use agreement.
