#%% -*- coding: utf-8 -*-
"""
External validation for stack_views_nnls on external_validation_merged.xlsx.

Workflow:
1. Train base models and meta NNLS stacker using the same pipeline as stack_views_nnls.py.
2. Predict on external dataset.
3. Plot ROC-with-CI and clinical decision curves (same style as stack_views_nnls.py).
4. Use stratified bootstrap (original class counts) with BCa CI and high rounds.
"""

from __future__ import annotations

import copy
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

import stack_views_nnls as sv


EXTERNAL_FILE = "External_2.0.xlsx"
EXTERNAL_MAX_AGE = 8
BOOTSTRAP_ROUNDS = 5000

METRICS_FILE = "external_stack_nnls_metrics_bootstrap_bca.xlsx"
PRED_FILE = "external_stack_nnls_predictions_bootstrap_bca.xlsx"
IMPUTED_PROBA_FILE = "external_stack_nnls_imputed_with_all_model_proba.xlsx"
ROC_FIG = "figures/external_roc_ci_bootstrap_bca.png"
PR_FIG = "figures/external_pr_ci_bootstrap_bca.png"
DCA_FIG = "figures/external_decision_curve_bootstrap_bca.png"
EXTERNAL_SINGLE_ROC_FIG_TEMPLATE = "figures/external_{model}_roc.png"
EXTERNAL_SINGLE_PR_FIG_TEMPLATE = "figures/external_{model}_pr.png"
EXTERNAL_COMBINED_ROC_FIG = "figures/combined_external_roc_ABCDE.png"
EXTERNAL_COMBINED_PR_FIG = "figures/combined_external_pr_ABCDE.png"


def fmt_ci(point: float, ci: tuple[float, float]) -> str:
    lo, hi = ci
    if np.isnan(lo) or np.isnan(hi):
        return f"{point:.3f} (nan)"
    return f"{point:.3f} ({lo:.3f}-{hi:.3f})"


def _clip_ci(lo: float, hi: float, lower: float = 0.0, upper: float = 1.0) -> tuple[float, float]:
    return float(max(lower, lo)), float(min(upper, hi))


def _percentile_ci(
    values: np.ndarray,
    alpha: float = 0.05,
    lower: float = 0.0,
    upper: float = 1.0,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(arr, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return _clip_ci(float(lo), float(hi), lower=lower, upper=upper)


def _bca_ci(
    boot_values: np.ndarray,
    point_estimate: float,
    jack_values: np.ndarray,
    alpha: float = 0.05,
    lower: float = 0.0,
    upper: float = 1.0,
) -> tuple[float, float]:
    boot = np.asarray(boot_values, dtype=float)
    boot = boot[np.isfinite(boot)]
    jack = np.asarray(jack_values, dtype=float)
    jack = jack[np.isfinite(jack)]
    if boot.size < 10 or jack.size < 3:
        return _percentile_ci(boot, alpha=alpha, lower=lower, upper=upper)

    eps = 1.0 / (2.0 * boot.size)
    p = float(np.mean(boot < point_estimate))
    p = min(max(p, eps), 1.0 - eps)
    z0 = NormalDist().inv_cdf(p)

    jack_mean = float(np.mean(jack))
    diff = jack_mean - jack
    denom = float(np.sum(diff ** 2))
    if denom <= 1e-15:
        a = 0.0
    else:
        num = float(np.sum(diff ** 3))
        a = num / (6.0 * (denom ** 1.5))

    def _adjusted_quantile(prob: float) -> float:
        z = NormalDist().inv_cdf(prob)
        den = 1.0 - a * (z0 + z)
        if abs(den) < 1e-12:
            return float("nan")
        return NormalDist().cdf(z0 + (z0 + z) / den)

    q_low = _adjusted_quantile(alpha / 2.0)
    q_high = _adjusted_quantile(1.0 - alpha / 2.0)
    if not np.isfinite(q_low) or not np.isfinite(q_high) or q_low <= 0.0 or q_high >= 1.0 or q_low >= q_high:
        return _percentile_ci(boot, alpha=alpha, lower=lower, upper=upper)

    lo, hi = np.quantile(boot, [q_low, q_high])
    return _clip_ci(float(lo), float(hi), lower=lower, upper=upper)


def _bootstrap_stats_with_bca(
    boot_values: np.ndarray,
    point_estimate: float,
    jack_values: np.ndarray,
    alpha: float = 0.05,
    lower: float = 0.0,
    upper: float = 1.0,
) -> dict:
    boot = np.asarray(boot_values, dtype=float)
    boot = boot[np.isfinite(boot)]
    n_boot = int(boot.size)
    mean = float(np.mean(boot)) if n_boot > 0 else float("nan")
    std = float(np.std(boot, ddof=1)) if n_boot > 1 else 0.0
    sem = float(std / np.sqrt(n_boot)) if n_boot > 0 else float("nan")
    ci = _bca_ci(
        boot_values=boot,
        point_estimate=point_estimate,
        jack_values=jack_values,
        alpha=alpha,
        lower=lower,
        upper=upper,
    )
    return {"mean": mean, "std": std, "sem": sem, "n_boot": n_boot, "ci": ci}


def evaluate_with_fixed_threshold(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(proba, dtype=float)
    pred = (p_arr >= threshold).astype(int)

    auc = roc_auc_score(y_arr, p_arr)
    auprc = average_precision_score(y_arr, p_arr)
    acc = accuracy_score(y_arr, pred)
    rec = recall_score(y_arr, pred)
    f1 = f1_score(y_arr, pred)
    tn = ((y_arr == 0) & (pred == 0)).sum()
    fp = ((y_arr == 0) & (pred == 1)).sum()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return {
        "auc": auc,
        "auprc": auprc,
        "accuracy": acc,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "threshold_used": float(threshold),
    }


def smooth_pr_curve_for_plot(
    recall: np.ndarray,
    precision: np.ndarray,
    n_points: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    recall_arr = np.asarray(recall, dtype=float)
    precision_arr = np.asarray(precision, dtype=float)
    order = np.argsort(recall_arr)
    recall_sorted = recall_arr[order]
    precision_sorted = precision_arr[order]

    # Use the precision envelope for display only; AUPRC calculation is unchanged.
    precision_env = np.maximum.accumulate(precision_sorted[::-1])[::-1]
    uniq_recall, inverse = np.unique(recall_sorted, return_inverse=True)
    uniq_precision = np.zeros_like(uniq_recall, dtype=float)
    np.maximum.at(uniq_precision, inverse, precision_env)

    recall_grid = np.linspace(0.0, 1.0, n_points)
    precision_smooth = np.interp(
        recall_grid,
        uniq_recall,
        uniq_precision,
        left=uniq_precision[0],
        right=uniq_precision[-1],
    )
    return recall_grid, precision_smooth


def format_model_name(name: str) -> str:
    return sv.format_model_name(name)


EXTERNAL_MODEL_COLORS = {
    "blood_extratrees": "#FF7F0E",
    "blood_xgb": "#1F77B4",
    "allergen_cat": "#2CA02C",
    "history_logreg": "#9467BD",
    "stack_nnls": "#D62728",
}


def get_model_color(name: str, idx: int) -> str:
    return EXTERNAL_MODEL_COLORS.get(name, sv.get_model_color(name, idx))


def style_axis(ax: plt.Axes) -> None:
    sv.style_axis(ax)


def metrics_and_roc_fixed_threshold_bootstrap_bca(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    n_boot: int = BOOTSTRAP_ROUNDS,
    seed: int = sv.RANDOM_STATE,
    progress_prefix: str = "",
) -> tuple[dict, dict, dict, dict]:
    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(proba, dtype=float)
    if np.unique(y_arr).shape[0] < 2:
        raise ValueError("Need both classes for metric CI and ROC.")

    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]
    if len(pos_idx) < 2 or len(neg_idx) < 2:
        raise ValueError("Need at least 2 positive and 2 negative samples for bootstrap BCa.")

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    n_boot_eff = max(1000, int(n_boot))
    rng = np.random.default_rng(seed)
    progress_every = max(1, n_boot_eff // 20)
    prefix = f"{progress_prefix} " if progress_prefix else ""
    print(
        f"{prefix}Bootstrap start: rounds={n_boot_eff}, "
        f"sample_pos={n_pos}, sample_neg={n_neg}"
    )

    metric_keys = ["auc", "auprc", "accuracy", "recall", "specificity", "f1"]
    boot_vals = {k: [] for k in metric_keys}
    for i in range(n_boot_eff):
        idx_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        idx_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        idx = np.concatenate([idx_pos, idx_neg])
        rng.shuffle(idx)

        y_s = y_arr[idx]
        p_s = p_arr[idx]
        m = evaluate_with_fixed_threshold(y_s, p_s, threshold=threshold)
        for k in metric_keys:
            boot_vals[k].append(m[k])
        done = i + 1
        if done % progress_every == 0 or done == n_boot_eff:
            pct = 100.0 * done / n_boot_eff
            print(f"{prefix}Bootstrap progress: {done}/{n_boot_eff} ({pct:.1f}%)", flush=True)

    jack_vals = {k: [] for k in metric_keys}
    n = len(y_arr)
    keep_mask = np.ones(n, dtype=bool)
    for i in range(n):
        keep_mask[i] = False
        y_j = y_arr[keep_mask]
        p_j = p_arr[keep_mask]
        keep_mask[i] = True
        if np.unique(y_j).shape[0] < 2:
            continue
        m_j = evaluate_with_fixed_threshold(y_j, p_j, threshold=threshold)
        for k in metric_keys:
            jack_vals[k].append(m_j[k])

    point = evaluate_with_fixed_threshold(y_arr, p_arr, threshold=threshold)
    full_fpr, full_tpr, _ = roc_curve(y_arr, p_arr)
    full_precision, full_recall, _ = precision_recall_curve(y_arr, p_arr)
    ci = {}
    boot_stats = {}
    for k in metric_keys:
        stat = _bootstrap_stats_with_bca(
            boot_values=np.asarray(boot_vals[k], dtype=float),
            point_estimate=float(point[k]),
            jack_values=np.asarray(jack_vals[k], dtype=float),
            alpha=0.05,
            lower=0.0,
            upper=1.0,
        )
        ci[k] = stat["ci"]
        boot_stats[k] = stat

    roc = {
        "full_fpr": full_fpr,
        "full_tpr": full_tpr,
        "full_recall": full_recall[::-1],
        "full_precision": full_precision[::-1],
        "prevalence": float(y_arr.mean()),
        "auc_ci": ci["auc"],
    }
    print(f"{prefix}Bootstrap complete.")
    return point, ci, roc, boot_stats


def load_external_dataset(train_columns: List[str]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    path = sv.BASE_DIR / EXTERNAL_FILE
    if not path.exists():
        raise FileNotFoundError(f"External file not found: {path}")

    raw = pd.read_excel(path)
    raw = raw.copy()
    if "age" in raw.columns:
        age = pd.to_numeric(raw["age"], errors="coerce")
        raw = raw[age <= EXTERNAL_MAX_AGE].reset_index(drop=True)

    drop_cols = [c for c in sv.DROP_COLUMNS if c in raw.columns]
    if sv.RECORD_COUNT_COL in raw.columns:
        drop_cols.append(sv.RECORD_COUNT_COL)
    df = raw.drop(columns=drop_cols, errors="ignore")

    if sv.TARGET not in df.columns:
        raise KeyError(f"Target column {sv.TARGET} missing in external dataset.")

    y_ext = df[sv.TARGET].astype(int)
    X_ext = df.drop(columns=[sv.TARGET])

    if sv.USE_ENGINEERED_ALLERGEN_FEATURES:
        X_ext = sv.add_allergen_features(X_ext)

    for col in train_columns:
        if col not in X_ext.columns:
            X_ext[col] = np.nan
    X_ext = X_ext[train_columns]
    return X_ext, y_ext, raw


def build_imputed_external_dataset(
    X_train: pd.DataFrame,
    X_ext: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build an external sample-level imputed dataset with the SAME per-view
    imputation logic as model training/inference.
    Missing values are filled only where the corresponding view can impute.
    """
    cat_cols = sv.DEFAULT_CATEGORICAL
    models = sv.build_models()

    base_df = X_ext.copy()
    imputed_df = X_ext.copy()
    model_fill_rows = []

    for model_name, (_, view_cols, scale_numeric) in models.items():
        _, X_ext_view = sv.prepare_view_split(
            train_df=X_train,
            test_df=X_ext,
            view_cols=view_cols,
            categorical_cols=cat_cols,
            scale_numeric=scale_numeric,
        )
        cols = list(X_ext_view.columns)
        if not cols:
            continue

        before_missing = base_df[cols].isna()
        can_fill = before_missing & X_ext_view[cols].notna()
        fill_count = int(can_fill.to_numpy().sum())
        if fill_count > 0:
            # Fill only missing cells; keep observed external values unchanged.
            imputed_df.loc[:, cols] = imputed_df[cols].where(~can_fill, X_ext_view[cols])

        model_fill_rows.append(
            {
                "model": model_name,
                "view_col_count": len(cols),
                "filled_cells": fill_count,
                "scale_numeric": bool(scale_numeric),
            }
        )

    col_stats_rows = []
    for col in imputed_df.columns:
        miss_before = int(base_df[col].isna().sum())
        miss_after = int(imputed_df[col].isna().sum())
        col_stats_rows.append(
            {
                "column": col,
                "missing_before": miss_before,
                "missing_after": miss_after,
                "filled_count": miss_before - miss_after,
            }
        )

    model_fill_df = pd.DataFrame(model_fill_rows)
    col_stats_df = pd.DataFrame(col_stats_rows)
    return imputed_df, model_fill_df, col_stats_df


def plot_roc_with_ci(
    roc_results: Dict[str, dict],
    model_order: List[str],
    out_path: Path,
) -> None:
    if not roc_results:
        return
    sv.setup_publication_style()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=sv.FIG_SIZE_STANDARD, facecolor="white")
    style_axis(ax)
    for idx, name in enumerate(model_order):
        roc = roc_results.get(name)
        if roc is None:
            continue
        full_fpr = roc["full_fpr"]
        full_tpr = roc["full_tpr"]
        auc_ci = roc["auc_ci"]
        auc_point = roc["auc_point"]
        color = get_model_color(name, idx)

        ax.plot(
            full_fpr,
            full_tpr,
            color=color,
            lw=sv.MODEL_LINE_WIDTH,
            solid_capstyle="round",
            label=f"{format_model_name(name)}  AUC {auc_point:.3f} [{auc_ci[0]:.3f}-{auc_ci[1]:.3f}]",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", lw=sv.REF_LINE_WIDTH, color="#6b7280", alpha=0.9)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    sv.save_figure_with_pdf(fig, out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"ROC with CI saved to {out_path}")


def plot_pr_with_ci(
    curve_results: Dict[str, dict],
    model_order: List[str],
    out_path: Path,
) -> None:
    if not curve_results:
        return
    sv.setup_publication_style()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=sv.FIG_SIZE_STANDARD, facecolor="white")
    style_axis(ax)
    baseline = None
    curve_mins: List[float] = []
    curve_maxs: List[float] = []
    for idx, name in enumerate(model_order):
        curve = curve_results.get(name)
        if curve is None:
            continue
        full_recall = curve["full_recall"]
        full_precision = curve["full_precision"]
        smooth_recall, smooth_precision = smooth_pr_curve_for_plot(full_recall, full_precision)
        auprc_ci = curve["auprc_ci"]
        auprc_point = curve["auprc_point"]
        baseline = curve["prevalence"]
        color = get_model_color(name, idx)
        curve_mins.append(float(np.nanmin(smooth_precision)))
        curve_maxs.append(float(np.nanmax(smooth_precision)))

        ax.plot(
            smooth_recall,
            smooth_precision,
            color=color,
            lw=sv.MODEL_LINE_WIDTH,
            solid_capstyle="round",
            label=f"{format_model_name(name)}  AUPRC {auprc_point:.3f} [{auprc_ci[0]:.3f}-{auprc_ci[1]:.3f}]",
        )

    if baseline is not None:
        ax.axhline(
            baseline,
            color="#6b7280",
            linestyle="--",
            lw=sv.REF_LINE_WIDTH,
            label=f"Prevalence {baseline:.3f}",
        )
        curve_mins.append(float(baseline))
        curve_maxs.append(float(baseline))
    ax.set_xlim(0.0, 1.0)
    if curve_mins and curve_maxs:
        y_min = min(curve_mins)
        y_max = max(curve_maxs)
        pad = max(0.02, 0.08 * max(y_max - y_min, 0.05))
        ax.set_ylim(max(0.0, y_min - pad), min(1.02, y_max + pad))
    else:
        ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    sv.save_figure_with_pdf(fig, out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"PR with CI saved to {out_path}")


def plot_single_model_roc_with_ci(
    model_name: str,
    model_index: int,
    curve: dict,
    out_path: Path,
) -> None:
    sv.setup_publication_style()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=sv.FIG_SIZE_COMPACT, facecolor="white")
    style_axis(ax)
    color = get_model_color(model_name, model_index)
    full_fpr = np.asarray(curve["full_fpr"], dtype=float)
    full_tpr = np.asarray(curve["full_tpr"], dtype=float)
    auc_point = float(curve["auc_point"])
    auc_ci = tuple(curve.get("auc_ci", (np.nan, np.nan)))
    ax.plot(
        full_fpr,
        full_tpr,
        color=color,
        lw=sv.MODEL_LINE_WIDTH,
        solid_capstyle="round",
        label=f"{format_model_name(model_name)}  AUC {auc_point:.3f} [{auc_ci[0]:.3f}-{auc_ci[1]:.3f}]",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", lw=sv.REF_LINE_WIDTH, color="#6b7280", alpha=0.9)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    sv.save_figure_with_pdf(fig, out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[External] Single-model ROC saved to {out_path}")


def plot_single_model_pr_with_ci(
    model_name: str,
    model_index: int,
    curve: dict,
    out_path: Path,
) -> None:
    sv.setup_publication_style()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=sv.FIG_SIZE_COMPACT, facecolor="white")
    style_axis(ax)
    color = get_model_color(model_name, model_index)
    full_recall = np.asarray(curve["full_recall"], dtype=float)
    full_precision = np.asarray(curve["full_precision"], dtype=float)
    smooth_recall, smooth_precision = smooth_pr_curve_for_plot(full_recall, full_precision)
    auprc_point = float(curve["auprc_point"])
    auprc_ci = tuple(curve.get("auprc_ci", (np.nan, np.nan)))
    prevalence = float(curve.get("prevalence", np.nan))
    ax.plot(
        smooth_recall,
        smooth_precision,
        color=color,
        lw=sv.MODEL_LINE_WIDTH,
        solid_capstyle="round",
        label=f"{format_model_name(model_name)}  AUPRC {auprc_point:.3f} [{auprc_ci[0]:.3f}-{auprc_ci[1]:.3f}]",
    )
    if np.isfinite(prevalence):
        ax.axhline(
            prevalence,
            color="#6b7280",
            linestyle="--",
            lw=sv.REF_LINE_WIDTH,
            label=f"Prevalence {prevalence:.3f}",
        )
    ax.set_xlim(0.0, 1.0)
    try:
        sv.set_pr_axis_limits(
            ax,
            float(np.nanmin(smooth_precision)),
            float(np.nanmax(smooth_precision)),
            prevalence=prevalence if np.isfinite(prevalence) else None,
        )
    except Exception:
        ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    sv.save_figure_with_pdf(fig, out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[External] Single-model PR saved to {out_path}")


def export_external_single_curves_and_collages(
    roc_results: Dict[str, dict],
    model_order: List[str],
) -> None:
    roc_image_names: List[str] = []
    pr_image_names: List[str] = []
    figures_dir = sv.BASE_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for idx, model_name in enumerate(model_order):
        curve = roc_results.get(model_name)
        if curve is None:
            continue
        roc_rel = Path(EXTERNAL_SINGLE_ROC_FIG_TEMPLATE.format(model=model_name))
        pr_rel = Path(EXTERNAL_SINGLE_PR_FIG_TEMPLATE.format(model=model_name))
        roc_out = sv.BASE_DIR / roc_rel
        pr_out = sv.BASE_DIR / pr_rel
        plot_single_model_roc_with_ci(
            model_name=model_name,
            model_index=idx,
            curve=curve,
            out_path=roc_out,
        )
        plot_single_model_pr_with_ci(
            model_name=model_name,
            model_index=idx,
            curve=curve,
            out_path=pr_out,
        )
        roc_image_names.append(roc_rel.name)
        pr_image_names.append(pr_rel.name)

    if len(roc_image_names) != 5 or len(pr_image_names) != 5:
        print(
            f"[External] Skip 3+2 collage: need 5 models, "
            f"got ROC={len(roc_image_names)}, PR={len(pr_image_names)}."
        )
        return

    try:
        import make_figure_collages_3plus2 as collage_builder

        roc_collage = collage_builder.build_collage_3_plus_2(
            figures_dir=figures_dir,
            image_names=roc_image_names,
            output_name=Path(EXTERNAL_COMBINED_ROC_FIG).name,
        )
        pr_collage = collage_builder.build_collage_3_plus_2(
            figures_dir=figures_dir,
            image_names=pr_image_names,
            output_name=Path(EXTERNAL_COMBINED_PR_FIG).name,
        )
        print(f"[External] Combined ROC collage saved to {roc_collage}")
        print(f"[External] Combined PR collage saved to {pr_collage}")
    except Exception as exc:
        print(f"[External] Warning: failed to build 3+2 collages. reason={exc}")


def train_base_and_stack_for_external(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_ext: pd.DataFrame,
) -> tuple[Dict[str, np.ndarray], Dict[str, float], List[str], Dict[str, Dict[str, float]]]:
    cat_cols = sv.DEFAULT_CATEGORICAL
    kf = StratifiedKFold(n_splits=sv.K_SPLITS, shuffle=True, random_state=sv.RANDOM_STATE)
    models = sv.build_models()

    base_names: List[str] = []
    oof_probs: List[np.ndarray] = []
    ext_probs: Dict[str, np.ndarray] = {}
    train_thresholds: Dict[str, Dict[str, float]] = {}

    for name, (model, view, scale_flag) in models.items():
        print(f"[External] OOF and fit base model: {name}")
        oof_proba, _, _ = sv.oof_proba_for_model(
            base_model=model,
            X=X_train,
            y=y_train,
            view_cols=view,
            categorical_cols=cat_cols,
            kf=kf,
            scale_numeric=scale_flag,
            return_fold_curves=False,
        )
        oof_probs.append(oof_proba)
        base_names.append(name)
        thr_j, thr_f1, _ = sv._select_thresholds(y_train.values, oof_proba)
        train_thresholds[name] = {"threshold_j_train": float(thr_j), "threshold_f1_train": float(thr_f1)}

        X_tr_view, X_ext_view = sv.prepare_view_split(
            train_df=X_train,
            test_df=X_ext,
            view_cols=view,
            categorical_cols=cat_cols,
            scale_numeric=scale_flag,
        )
        X_tr_bal, y_tr_bal = sv.resample_training_data(X_tr_view, y_train, cat_cols)
        try:
            final_model = clone(model)
        except Exception:
            final_model = copy.deepcopy(model)
        final_model.fit(X_tr_bal, y_tr_bal)
        ext_probs[name] = final_model.predict_proba(X_ext_view)[:, 1]

    base_oof_matrix = np.column_stack(oof_probs)
    meta = LinearRegression(positive=True, fit_intercept=True)
    meta.fit(sv._logit(base_oof_matrix), y_train.values)

    base_ext_matrix = np.column_stack([ext_probs[n] for n in base_names])
    stack_score_ext = sv._logit(base_ext_matrix) @ meta.coef_ + meta.intercept_
    ext_probs["stack_nnls"] = sv._sigmoid(stack_score_ext)
    # Development thresholds use NNLS-layer OOF; external parameters use
    # the separate NNLS model fitted on all development base OOF features.
    meta_oof_result = sv.nnls_oof_predictions(base_oof_matrix, y_train.values)
    stack_train_proba = meta_oof_result["nnls_oof"]
    np.savez_compressed(sv.BASE_DIR / "external_development_nnls_predictions.npz",
        y=y_train.to_numpy(), base_prob_matrix=base_oof_matrix,
        nnls_oof=stack_train_proba, fold_ids=meta_oof_result["meta_fold_ids"],
        counts=meta_oof_result["meta_oof_counts"],
        coef=meta.coef_, intercept=meta.intercept_,
        evaluation_version=sv.NNLS_EVALUATION_VERSION)
    thr_j, thr_f1, _ = sv._select_thresholds(y_train.values, stack_train_proba)
    train_thresholds["stack_nnls"] = {"threshold_j_train": float(thr_j), "threshold_f1_train": float(thr_f1)}

    weights = {name: float(w) for name, w in zip(base_names, meta.coef_)}
    weights["intercept"] = float(meta.intercept_)
    return ext_probs, weights, base_names, train_thresholds


def main() -> None:
    try:
        # Keep output paths consistent with stack_views_nnls.py.
        import os

        os.chdir(sv.BASE_DIR)
    except OSError:
        pass
    sv.setup_publication_style()

    X_train, y_train = sv.load_dataset()
    X_ext, y_ext, ext_raw = load_external_dataset(train_columns=list(X_train.columns))
    X_ext_imputed, impute_model_fill, impute_col_stats = build_imputed_external_dataset(
        X_train=X_train,
        X_ext=X_ext,
    )

    if y_ext.nunique() < 2:
        raise ValueError("External dataset has <2 classes after filtering; cannot run ROC/AUC.")

    print(f"Training set: {len(X_train)} rows, positive rate={y_train.mean():.3f}")
    print(f"External set: {len(X_ext)} rows, positive rate={y_ext.mean():.3f}")
    print(
        "Stratified bootstrap for CI (original class counts): "
        f"n_boot={BOOTSTRAP_ROUNDS}, sample_pos=n_pos, sample_neg=n_neg"
    )

    model_probs_raw, nnls_weights, base_names, train_thresholds = train_base_and_stack_for_external(
        X_train=X_train,
        y_train=y_train,
        X_ext=X_ext,
    )
    model_order = base_names + ["stack_nnls"]

    # Use raw probabilities directly.
    model_probs = {name: np.asarray(model_probs_raw[name], dtype=float) for name in model_order}
    print("[External] Using raw probabilities only.")

    # Metric CI by BCa bootstrap; ROC band by bootstrap percentile.
    metrics_rows = []
    metrics_raw_rows = []
    roc_results: Dict[str, dict] = {}
    for name in model_order:
        proba = model_probs[name]
        fixed_thr = train_thresholds[name]["threshold_j_train"]
        point, ci, roc, boot_stats = metrics_and_roc_fixed_threshold_bootstrap_bca(
            y_true=y_ext.values,
            proba=proba,
            threshold=fixed_thr,
            n_boot=BOOTSTRAP_ROUNDS,
            seed=sv.RANDOM_STATE,
            progress_prefix=f"[External][{name}]",
        )
        roc["auc_point"] = point["auc"]
        roc["auprc_point"] = point["auprc"]
        roc["auprc_ci"] = ci["auprc"]
        roc_results[name] = roc

        metrics_rows.append(
            {
                "model": name,
                "AUC": fmt_ci(point["auc"], ci["auc"]),
                "AUPRC": fmt_ci(point["auprc"], ci["auprc"]),
                "Accuracy": fmt_ci(point["accuracy"], ci["accuracy"]),
                "Recall": fmt_ci(point["recall"], ci["recall"]),
                "Specificity": fmt_ci(point["specificity"], ci["specificity"]),
                "F1": fmt_ci(point["f1"], ci["f1"]),
                "threshold_j_train": train_thresholds[name]["threshold_j_train"],
                "threshold_f1_train": train_thresholds[name]["threshold_f1_train"],
            }
        )
        metrics_raw_rows.append(
            {
                "model": name,
                "auc": point["auc"],
                "auprc": point["auprc"],
                "accuracy": point["accuracy"],
                "recall": point["recall"],
                "specificity": point["specificity"],
                "f1": point["f1"],
                "threshold_used": point["threshold_used"],
                "threshold_j_train": train_thresholds[name]["threshold_j_train"],
                "threshold_f1_train": train_thresholds[name]["threshold_f1_train"],
                "auc_ci_low": ci["auc"][0],
                "auc_ci_high": ci["auc"][1],
                "auprc_ci_low": ci["auprc"][0],
                "auprc_ci_high": ci["auprc"][1],
                "accuracy_ci_low": ci["accuracy"][0],
                "accuracy_ci_high": ci["accuracy"][1],
                "recall_ci_low": ci["recall"][0],
                "recall_ci_high": ci["recall"][1],
                "specificity_ci_low": ci["specificity"][0],
                "specificity_ci_high": ci["specificity"][1],
                "f1_ci_low": ci["f1"][0],
                "f1_ci_high": ci["f1"][1],
                "auc_boot_mean": boot_stats["auc"]["mean"],
                "auc_boot_std": boot_stats["auc"]["std"],
                "auc_boot_sem": boot_stats["auc"]["sem"],
                "auprc_boot_mean": boot_stats["auprc"]["mean"],
                "auprc_boot_std": boot_stats["auprc"]["std"],
                "auprc_boot_sem": boot_stats["auprc"]["sem"],
                "accuracy_boot_mean": boot_stats["accuracy"]["mean"],
                "accuracy_boot_std": boot_stats["accuracy"]["std"],
                "accuracy_boot_sem": boot_stats["accuracy"]["sem"],
                "recall_boot_mean": boot_stats["recall"]["mean"],
                "recall_boot_std": boot_stats["recall"]["std"],
                "recall_boot_sem": boot_stats["recall"]["sem"],
                "specificity_boot_mean": boot_stats["specificity"]["mean"],
                "specificity_boot_std": boot_stats["specificity"]["std"],
                "specificity_boot_sem": boot_stats["specificity"]["sem"],
                "f1_boot_mean": boot_stats["f1"]["mean"],
                "f1_boot_std": boot_stats["f1"]["std"],
                "f1_boot_sem": boot_stats["f1"]["sem"],
                "n_boot": BOOTSTRAP_ROUNDS,
                "ci_method": "Stratified bootstrap BCa (original class counts)",
                "roc_band_ci_method": "Bootstrap percentile",
            }
        )
        print(
            f"[External] {name}: "
            f"AUC={point['auc']:.3f}, AUPRC={point['auprc']:.3f}, ACC={point['accuracy']:.3f}, "
            f"RECALL={point['recall']:.3f}, SPEC={point['specificity']:.3f}, "
            f"F1={point['f1']:.3f}, THR_FIXED={fixed_thr:.3f}"
        )

    # Save metrics
    metrics_path = sv.BASE_DIR / METRICS_FILE
    with pd.ExcelWriter(metrics_path) as writer:
        pd.DataFrame(metrics_rows).to_excel(writer, sheet_name="metrics_ci", index=False)
        pd.DataFrame(metrics_raw_rows).to_excel(writer, sheet_name="metrics_raw", index=False)
        pd.DataFrame(
            [{"name": k, "weight": v} for k, v in nnls_weights.items()]
        ).to_excel(writer, sheet_name="nnls_weights", index=False)
        pd.DataFrame(
            [{"model": k, **v} for k, v in train_thresholds.items()]
        ).to_excel(writer, sheet_name="train_thresholds", index=False)
    print(f"External metrics saved to {metrics_path}")

    # Save external predictions
    pred_df = pd.DataFrame(
        {
            "Asthma": y_ext.values,
            **{f"proba_{name}": model_probs[name] for name in model_order},
            **{
                f"pred_{name}": (model_probs[name] >= train_thresholds[name]["threshold_j_train"]).astype(int)
                for name in model_order
            },
        }
    )
    if "Unique patient number" in ext_raw.columns:
        pred_df.insert(0, "Unique patient number", ext_raw["Unique patient number"].values)
    pred_path = sv.BASE_DIR / PRED_FILE
    pred_df.to_excel(pred_path, index=False)
    print(f"External predictions saved to {pred_path}")

    # Save per-sample imputed data + all-model probabilities.
    imputed_with_proba = pd.concat(
        [
            pred_df[["Unique patient number", "Asthma"]] if "Unique patient number" in pred_df.columns else pred_df[["Asthma"]],
            X_ext_imputed.reset_index(drop=True),
            pd.DataFrame(
                {
                    **{f"proba_{name}": model_probs[name] for name in model_order},
                    **{
                        f"pred_{name}": (
                            model_probs[name] >= train_thresholds[name]["threshold_j_train"]
                        ).astype(int)
                        for name in model_order
                    },
                }
            ),
        ],
        axis=1,
    )
    imputed_proba_path = sv.BASE_DIR / IMPUTED_PROBA_FILE
    with pd.ExcelWriter(imputed_proba_path) as writer:
        imputed_with_proba.to_excel(writer, sheet_name="samples_imputed_proba", index=False)
        impute_model_fill.to_excel(writer, sheet_name="impute_model_fill", index=False)
        impute_col_stats.to_excel(writer, sheet_name="impute_col_stats", index=False)
    print(f"External imputed samples + probabilities saved to {imputed_proba_path}")

    # Plot curves
    plot_roc_with_ci(roc_results=roc_results, model_order=model_order, out_path=sv.BASE_DIR / ROC_FIG)
    plot_pr_with_ci(curve_results=roc_results, model_order=model_order, out_path=sv.BASE_DIR / PR_FIG)
    sv.plot_decision_curves(
        y_true=y_ext.values,
        prob_dict={name: model_probs[name] for name in model_order},
        thresholds=np.linspace(0.01, 0.99, 99),
        out_path=sv.BASE_DIR / DCA_FIG,
    )
    export_external_single_curves_and_collages(
        roc_results=roc_results,
        model_order=model_order,
    )


if __name__ == "__main__":
    main()

# %%
