# %%
"""
Compare model performance with vs without SMOTE for the stack NNLS pipeline.

This script reuses the existing preprocessing/model setup in:
- stack_views_nnls.py
- external_validate_stack_views_nnls_delong.py

Output:
- smote_vs_no_smote.xlsx
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, List, Tuple

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold

import external_validate_stack_views_nnls_delong as ev
import stack_views_nnls as sv


SCENARIO_WITH_SMOTE = "with_smote"
SCENARIO_NO_SMOTE = "without_smote"
OUTPUT_FILE = "smote_vs_no_smote.xlsx"
METRIC_COLS = ["auc", "auprc", "accuracy", "recall", "specificity", "f1"]


def _safe_clone(model):
    try:
        return clone(model)
    except Exception:
        return copy.deepcopy(model)


def _resample_identity(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    _categorical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    return X_train, y_train


@contextmanager
def _resampling_mode(use_smote: bool) -> Iterator[None]:
    """
    Temporarily switch sv.resample_training_data behavior.
    """
    original_fn: Callable[..., Tuple[pd.DataFrame, pd.Series]] = sv.resample_training_data
    if use_smote:
        yield
        return
    sv.resample_training_data = _resample_identity
    try:
        yield
    finally:
        sv.resample_training_data = original_fn


def _scenario_name(use_smote: bool) -> str:
    return SCENARIO_WITH_SMOTE if use_smote else SCENARIO_NO_SMOTE


def _to_metric_row(
    scenario: str,
    dataset: str,
    model: str,
    metrics: dict,
    threshold_j_train: float,
    threshold_f1_train: float,
    threshold_used: float,
) -> dict:
    return {
        "scenario": scenario,
        "dataset": dataset,
        "model": model,
        "auc": float(metrics["auc"]),
        "auprc": float(metrics["auprc"]),
        "accuracy": float(metrics["accuracy"]),
        "recall": float(metrics["recall"]),
        "specificity": float(metrics["specificity"]),
        "f1": float(metrics["f1"]),
        "threshold_used": float(threshold_used),
        "threshold_j_train": float(threshold_j_train),
        "threshold_f1_train": float(threshold_f1_train),
    }


def run_one_scenario(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_ext: pd.DataFrame,
    y_ext: pd.Series,
    use_smote: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run internal OOF + external validation once for one sampling scenario.
    """
    scenario = _scenario_name(use_smote)
    cat_cols = sv.DEFAULT_CATEGORICAL
    kf = StratifiedKFold(
        n_splits=sv.K_SPLITS, shuffle=True, random_state=sv.RANDOM_STATE
    )
    models = sv.build_models()

    internal_rows: List[dict] = []
    external_rows: List[dict] = []
    threshold_rows: List[dict] = []
    weight_rows: List[dict] = []

    base_names: List[str] = []
    base_oof_probs: List[np.ndarray] = []
    base_ext_probs: List[np.ndarray] = []

    with _resampling_mode(use_smote):
        for name, (model, view_cols, scale_numeric) in models.items():
            print(f"[{scenario}] OOF and external fit: {name}")

            oof_proba, _, _ = sv.oof_proba_for_model(
                base_model=model,
                X=X_train,
                y=y_train,
                view_cols=view_cols,
                categorical_cols=cat_cols,
                kf=kf,
                scale_numeric=scale_numeric,
                return_fold_curves=False,
            )
            oof_proba = np.asarray(oof_proba, dtype=float)
            thr_j, thr_f1, _ = sv._select_thresholds(y_train.values, oof_proba)
            point_internal = sv._evaluate_with_threshold(y_train.values, oof_proba)
            internal_rows.append(
                _to_metric_row(
                    scenario=scenario,
                    dataset="internal_cv",
                    model=name,
                    metrics=point_internal,
                    threshold_j_train=float(thr_j),
                    threshold_f1_train=float(thr_f1),
                    threshold_used=float(thr_j),
                )
            )

            X_tr_view, X_ext_view = sv.prepare_view_split(
                train_df=X_train,
                test_df=X_ext,
                view_cols=view_cols,
                categorical_cols=cat_cols,
                scale_numeric=scale_numeric,
            )
            X_tr_bal, y_tr_bal = sv.resample_training_data(X_tr_view, y_train, cat_cols)
            final_model = _safe_clone(model)
            final_model.fit(X_tr_bal, y_tr_bal)
            ext_proba = final_model.predict_proba(X_ext_view)[:, 1]
            ext_proba = np.asarray(ext_proba, dtype=float)
            point_external = ev.evaluate_with_fixed_threshold(
                y_true=y_ext.values,
                proba=ext_proba,
                threshold=float(thr_j),
            )
            external_rows.append(
                _to_metric_row(
                    scenario=scenario,
                    dataset="external",
                    model=name,
                    metrics=point_external,
                    threshold_j_train=float(thr_j),
                    threshold_f1_train=float(thr_f1),
                    threshold_used=float(point_external["threshold_used"]),
                )
            )

            threshold_rows.append(
                {
                    "scenario": scenario,
                    "model": name,
                    "threshold_j_train": float(thr_j),
                    "threshold_f1_train": float(thr_f1),
                }
            )
            base_names.append(name)
            base_oof_probs.append(oof_proba)
            base_ext_probs.append(ext_proba)

        base_oof_matrix = np.column_stack(base_oof_probs)
        meta = LinearRegression(positive=True, fit_intercept=True)
        meta.fit(sv._logit(base_oof_matrix), y_train.values)
        meta_oof_result = sv.nnls_oof_predictions(base_oof_matrix, y_train.values)
        stack_oof_proba = meta_oof_result["nnls_oof"]
        np.savez_compressed(sv.BASE_DIR / f"smote_{scenario}_predictions.npz",
            y=y_train.to_numpy(), base_prob_matrix=base_oof_matrix,
            base_ext_matrix=np.column_stack(base_ext_probs),
            nnls_oof=stack_oof_proba, fold_ids=meta_oof_result["meta_fold_ids"],
            counts=meta_oof_result["meta_oof_counts"],
            coef=meta.coef_, intercept=meta.intercept_,
            evaluation_version=sv.NNLS_EVALUATION_VERSION)
        stack_thr_j, stack_thr_f1, _ = sv._select_thresholds(y_train.values, stack_oof_proba)
        point_stack_internal = sv._evaluate_with_threshold(y_train.values, stack_oof_proba)
        internal_rows.append(
            _to_metric_row(
                scenario=scenario,
                dataset="internal_cv",
                model="stack_nnls",
                metrics=point_stack_internal,
                threshold_j_train=float(stack_thr_j),
                threshold_f1_train=float(stack_thr_f1),
                threshold_used=float(stack_thr_j),
            )
        )

        base_ext_matrix = np.column_stack(base_ext_probs)
        stack_ext_score = sv._logit(base_ext_matrix) @ meta.coef_ + meta.intercept_
        stack_ext_proba = sv._sigmoid(stack_ext_score)
        point_stack_external = ev.evaluate_with_fixed_threshold(
            y_true=y_ext.values,
            proba=stack_ext_proba,
            threshold=float(stack_thr_j),
        )
        external_rows.append(
            _to_metric_row(
                scenario=scenario,
                dataset="external",
                model="stack_nnls",
                metrics=point_stack_external,
                threshold_j_train=float(stack_thr_j),
                threshold_f1_train=float(stack_thr_f1),
                threshold_used=float(point_stack_external["threshold_used"]),
            )
        )

    threshold_rows.append(
        {
            "scenario": scenario,
            "model": "stack_nnls",
            "threshold_j_train": float(stack_thr_j),
            "threshold_f1_train": float(stack_thr_f1),
        }
    )

    for model_name, coef in zip(base_names, meta.coef_):
        weight_rows.append(
            {
                "scenario": scenario,
                "model": model_name,
                "weight": float(coef),
            }
        )
    weight_rows.append(
        {
            "scenario": scenario,
            "model": "intercept",
            "weight": float(meta.intercept_),
        }
    )

    metrics_df = pd.DataFrame(internal_rows + external_rows)
    weights_df = pd.DataFrame(weight_rows)
    thresholds_df = pd.DataFrame(threshold_rows)
    return metrics_df, weights_df, thresholds_df


def build_compare_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build wide comparison table with delta = without_smote - with_smote.
    """
    part = metrics_df[["dataset", "model", "scenario"] + METRIC_COLS].copy()
    wide = part.pivot_table(
        index=["dataset", "model"],
        columns="scenario",
        values=METRIC_COLS,
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{scenario}" for metric, scenario in wide.columns]
    wide = wide.reset_index()

    for metric in METRIC_COLS:
        with_col = f"{metric}_{SCENARIO_WITH_SMOTE}"
        no_col = f"{metric}_{SCENARIO_NO_SMOTE}"
        delta_col = f"{metric}_delta_no_smote_minus_smote"
        if with_col in wide.columns and no_col in wide.columns:
            wide[delta_col] = wide[no_col] - wide[with_col]
        else:
            wide[delta_col] = np.nan

    return wide.sort_values(["dataset", "model"]).reset_index(drop=True)


def print_stack_summary(compare_df: pd.DataFrame) -> None:
    sub = compare_df[compare_df["model"] == "stack_nnls"].copy()
    if sub.empty:
        return
    for _, row in sub.iterrows():
        ds = row["dataset"]
        auc_with = row.get(f"auc_{SCENARIO_WITH_SMOTE}", np.nan)
        auc_no = row.get(f"auc_{SCENARIO_NO_SMOTE}", np.nan)
        auprc_with = row.get(f"auprc_{SCENARIO_WITH_SMOTE}", np.nan)
        auprc_no = row.get(f"auprc_{SCENARIO_NO_SMOTE}", np.nan)
        d_auc = row.get("auc_delta_no_smote_minus_smote", np.nan)
        d_auprc = row.get("auprc_delta_no_smote_minus_smote", np.nan)
        print(
            f"[Summary][{ds}] stack_nnls "
            f"AUC: with={auc_with:.4f}, without={auc_no:.4f}, delta={d_auc:+.4f}; "
            f"AUPRC: with={auprc_with:.4f}, without={auprc_no:.4f}, delta={d_auprc:+.4f}"
        )


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    try:
        import os

        os.chdir(base_dir)
    except OSError:
        pass

    print("Loading datasets...")
    X_train, y_train = sv.load_dataset()
    X_ext, y_ext, _ = ev.load_external_dataset(train_columns=list(X_train.columns))

    print("\n=== Scenario: with SMOTE ===")
    metrics_with, weights_with, thresholds_with = run_one_scenario(
        X_train=X_train,
        y_train=y_train,
        X_ext=X_ext,
        y_ext=y_ext,
        use_smote=True,
    )
    print("\n=== Scenario: without SMOTE ===")
    metrics_without, weights_without, thresholds_without = run_one_scenario(
        X_train=X_train,
        y_train=y_train,
        X_ext=X_ext,
        y_ext=y_ext,
        use_smote=False,
    )

    all_metrics = pd.concat([metrics_with, metrics_without], ignore_index=True)
    all_weights = pd.concat([weights_with, weights_without], ignore_index=True)
    all_thresholds = pd.concat([thresholds_with, thresholds_without], ignore_index=True)

    compare_all = build_compare_table(all_metrics)
    compare_internal = compare_all[compare_all["dataset"] == "internal_cv"].reset_index(drop=True)
    compare_external = compare_all[compare_all["dataset"] == "external"].reset_index(drop=True)

    out_path = base_dir / OUTPUT_FILE
    with pd.ExcelWriter(out_path) as writer:
        all_metrics.to_excel(writer, sheet_name="all_metrics", index=False)
        all_metrics[all_metrics["dataset"] == "internal_cv"].to_excel(
            writer, sheet_name="internal_metrics", index=False
        )
        all_metrics[all_metrics["dataset"] == "external"].to_excel(
            writer, sheet_name="external_metrics", index=False
        )
        compare_internal.to_excel(writer, sheet_name="internal_compare", index=False)
        compare_external.to_excel(writer, sheet_name="external_compare", index=False)
        all_weights.to_excel(writer, sheet_name="nnls_weights", index=False)
        all_thresholds.to_excel(writer, sheet_name="train_thresholds", index=False)

    print_stack_summary(compare_all)
    print(f"\nDone. Comparison file saved to: {out_path}")


if __name__ == "__main__":
    main()

