"""
Global feature importance for NNLS stacking using SHAP linearity (LOGIT scale).

We compute explanations in the logit (log-odds) space (clipped to [-5, 5]):

  z_j(x) = logit(p_j(x)) = log(p_j(x) / (1 - p_j(x)))

Then define an NNLS meta score with an intercept:

  Z(x) = b + Σ_j w_j * z_j(x)

where b is the fitted intercept and w_j are non-negative coefficients fitted
by NNLS on out-of-fold base logits. The intercept changes the expected value
but does not contribute a feature-level SHAP value.

By the SHAP linearity axiom, the ensemble SHAP value for feature i is:

  Φ_i(x) = Σ_j w_j * φ_{i,j}(x)

where φ_{i,j}(x) is the signed SHAP value of feature i for base model j on the
logit output (and 0 if the feature is not in that model's view).

Global Feature Importance (GFI) is:

  GFI_i = mean_x |Φ_i(x)|

We apply the absolute value after the weighted summation to preserve directionality
and allow cancellation across views.
"""
# %%
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import stack_views_nnls as base

LOGIT_EPS = 1e-6
LOGIT_MIN = -5.0
LOGIT_MAX = 5.0
IGNORED_FEATURES = {"total ige", "total ige_log1p"}
PLOT_MAX_FEATURES = 20
OTHER_FEATURES_LABEL = "Other features"
FIGURE_DPI = 300


def save_matplotlib_figure(fig, out_path: Path, **savefig_kwargs) -> Path:
    """Save a matplotlib figure as PNG plus a same-name PDF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI, **savefig_kwargs)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=FIGURE_DPI, **savefig_kwargs)
    return out_path


def _filter_view(view_cols: Sequence[str]) -> List[str]:
    return [c for c in view_cols if c.lower() not in IGNORED_FEATURES]


def build_plot_bundle_with_other_features(
    agg: np.ndarray,
    feature_values: pd.DataFrame,
    global_df: pd.DataFrame,
    top_n: int = PLOT_MAX_FEATURES,
    other_label: str = OTHER_FEATURES_LABEL,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, int]:
    """Keep the top (n-1) features and aggregate the rest into one synthetic row."""
    ranked_features = global_df["feature"].tolist()
    agg_df = pd.DataFrame(agg, columns=feature_values.columns)

    if len(ranked_features) <= top_n:
        plot_df = global_df.copy()
        plot_features = feature_values.reindex(columns=ranked_features, fill_value=0.0).copy()
        plot_shap = agg_df.reindex(columns=ranked_features, fill_value=0.0).to_numpy(dtype=float)
        return plot_shap, plot_features, plot_df, 0

    keep_features = ranked_features[: top_n - 1]
    other_features = ranked_features[top_n - 1 :]

    plot_shap_df = agg_df.reindex(columns=keep_features, fill_value=0.0).copy()
    other_shap = agg_df[other_features].sum(axis=1)
    plot_shap_df[other_label] = other_shap

    plot_features = feature_values.reindex(columns=keep_features, fill_value=0.0).copy()
    # Force the aggregated synthetic row to be treated as categorical so it is rendered in grey.
    plot_features[other_label] = pd.Categorical([other_label] * len(plot_features))

    plot_df = global_df[global_df["feature"].isin(keep_features)].copy()
    plot_df = pd.concat(
        [
            plot_df,
            pd.DataFrame(
                [
                    {
                        "feature": other_label,
                        "global_importance": float(np.mean(np.abs(other_shap.to_numpy(dtype=float)))),
                        "mean_signed_shap": float(np.mean(other_shap.to_numpy(dtype=float))),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return plot_shap_df.to_numpy(dtype=float), plot_features, plot_df, len(other_features)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute global feature importance (SHAP+NNLS).")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names to include (default: all models from stack_views_nnls.py).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("global_feature_importance.xlsx"),
        help="Output Excel path.",
    )
    parser.add_argument(
        "--best-params",
        type=Path,
        default=None,
        help="Optional best_params.json path; overrides stack_views_nnls.py default if provided.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=base.K_SPLITS,
        help="StratifiedKFold splits used to compute OOF base logits for NNLS weights.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=base.RANDOM_STATE,
        help="Random seed.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples used for SHAP aggregation (0 = use all rows).",
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=200,
        help="Background sample size for SHAP explainers.",
    )
    parser.add_argument(
        "--per-model",
        action="store_true",
        help="Also output per-model mean(|SHAP|) feature importance.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _read_best_params(path: Path) -> Dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("results"), dict):
        return data["results"]
    if isinstance(data, dict):
        return data
    return {}


def _apply_best_params(models: Dict[str, Tuple[object, List[str], bool]], params: Dict[str, dict]) -> None:
    for name, (model, _, _) in models.items():
        base.apply_tuned_params(model, name, params)


def _select_eval_indices(y: pd.Series, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or max_samples >= len(y):
        return np.arange(len(y))
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=seed)
    train_idx, _ = next(splitter.split(np.zeros(len(y)), y))
    return train_idx


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, LOGIT_EPS, 1.0 - LOGIT_EPS)
    z = np.log(p / (1.0 - p))
    return np.clip(z, LOGIT_MIN, LOGIT_MAX)


def compute_nnls_weights_from_oof(
    models: Dict[str, Tuple[object, List[str], bool]],
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: Sequence[str],
    splits: int,
    seed: int,
) -> Tuple[Dict[str, float], float]:
    kf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    base_logits: List[np.ndarray] = []
    names: List[str] = []
    for name, (model, view_cols, scale_flag) in models.items():
        proba_oof, _, _ = base.oof_proba_for_model(
            model,
            X,
            y,
            view_cols=view_cols,
            categorical_cols=categorical_cols,
            kf=kf,
            scale_numeric=scale_flag,
        )
        base_logits.append(_logit(proba_oof))
        names.append(name)

    base_matrix = np.column_stack(base_logits)
    meta = LinearRegression(positive=True, fit_intercept=True)
    meta.fit(base_matrix, y.values)
    weights = {n: float(w) for n, w in zip(names, meta.coef_)}
    intercept = float(meta.intercept_)
    return weights, intercept


def _ensure_2d_class1(shap_values) -> np.ndarray:
    # TreeExplainer may return list[class] or array with class axis.
    if isinstance(shap_values, list):
        if len(shap_values) == 0:
            raise ValueError("Empty SHAP list output.")
        if len(shap_values) == 1:
            return np.asarray(shap_values[0])
        return np.asarray(shap_values[1])
    arr = np.asarray(shap_values)
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        return arr[..., 1]
    return arr


def compute_shap_values_logit(
    model: object,
    X_background: pd.DataFrame,
    X_eval: pd.DataFrame,
    model_name: str,
    background_size: int,
    seed: int,
) -> np.ndarray:
    import shap

    bg_n = min(background_size, len(X_background))
    background = X_background.sample(n=bg_n, random_state=seed) if bg_n > 0 else X_background

    module = model.__class__.__module__.lower()
    is_linear = hasattr(model, "coef_") and hasattr(model, "decision_function")
    if is_linear:
        explainer = shap.LinearExplainer(model, background)
        sv = explainer.shap_values(X_eval)
        return np.asarray(sv)

    if "xgboost" in module or "catboost" in module:
        try:
            explainer = shap.TreeExplainer(model, background, model_output="raw")
            sv = explainer.shap_values(X_eval, check_additivity=False)
            return _ensure_2d_class1(sv)
        except Exception as exc:
            print(f"[{model_name}] TreeExplainer(raw) failed, fallback to permutation: {exc}")

    def predict_fn(data):
        proba = model.predict_proba(data)[:, 1]
        return _logit(proba)

    masker = shap.maskers.Independent(background)
    explainer = shap.Explainer(predict_fn, masker, algorithm="permutation")
    max_evals = max(2 * X_eval.shape[1] + 1, 50)
    explanation = explainer(X_eval, max_evals=max_evals)
    return np.asarray(explanation.values)


def fit_model_on_full_data(
    base_model: object,
    X_view: pd.DataFrame,
    y: pd.Series,
    categorical_cols: Sequence[str],
) -> object:
    X_bal, y_bal = base.resample_training_data(X_view, y, categorical_cols)
    try:
        model = clone(base_model)
    except Exception:
        model = copy.deepcopy(base_model)
    model.fit(X_bal, y_bal)
    return model


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    X, y = base.load_dataset()
    categorical_cols = list(base.DEFAULT_CATEGORICAL)

    models = base.build_models()
    if args.models:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in requested if m not in models]
        if unknown:
            raise SystemExit(f"Unknown model(s): {unknown}. Available: {sorted(models)}")
        models = {k: models[k] for k in requested}
    if args.best_params is not None:
        tuned = _read_best_params(args.best_params)
        _apply_best_params(models, tuned)
    # Drop Total IgE from all views for SHAP aggregation.
    models = {
        name: (model_obj, _filter_view(view_cols), scale_flag)
        for name, (model_obj, view_cols, scale_flag) in models.items()
    }

    print("Computing NNLS weights from OOF base logits...")
    weights, intercept = compute_nnls_weights_from_oof(
        models=models,
        X=X,
        y=y,
        categorical_cols=categorical_cols,
        splits=args.splits,
        seed=args.seed,
    )
    weights_rows = [{"model": k, "weight": v} for k, v in weights.items()]
    weights_rows.append({"model": "(intercept)", "weight": intercept})
    weights_df = pd.DataFrame(weights_rows).sort_values("weight", ascending=False).reset_index(
        drop=True
    )
    print("NNLS weights:")
    for row in weights_df.itertuples(index=False):
        print(f"  {row.model}: {row.weight:.6f}")
    print(f"NNLS intercept: {intercept:.6f}")

    eval_idx = _select_eval_indices(y, args.max_samples, args.seed)
    X_eval_raw = X.iloc[eval_idx].reset_index(drop=True)
    print(f"SHAP evaluation rows: {len(X_eval_raw)} / {len(X)}")

    shap_by_model: Dict[str, pd.DataFrame] = {}
    per_model_importance: Dict[str, pd.Series] = {}

    for model_name, (model_obj, view_cols, scale_flag) in models.items():
        w = weights.get(model_name, 0.0)
        if w <= 0:
            print(f"Skip {model_name} (weight={w:.6f})")
            continue

        print(f"Fitting {model_name} for SHAP (weight={w:.6f}) ...")
        X_view_full, X_view_eval = base.prepare_view_split(
            X,
            X_eval_raw,
            view_cols=view_cols,
            categorical_cols=categorical_cols,
            scale_numeric=scale_flag,
        )
        fitted_model = fit_model_on_full_data(model_obj, X_view_full, y, categorical_cols)

        print(f"Computing SHAP values (logit) for {model_name} ...")
        sv = compute_shap_values_logit(
            fitted_model,
            X_background=X_view_full,
            X_eval=X_view_eval,
            model_name=model_name,
            background_size=args.background_size,
            seed=args.seed,
        )
        sv = np.asarray(sv)
        if sv.shape[0] != len(X_view_eval) or sv.shape[1] != X_view_eval.shape[1]:
            raise ValueError(
                f"{model_name} SHAP shape mismatch: got {sv.shape}, expected "
                f"({len(X_view_eval)}, {X_view_eval.shape[1]})"
            )
        shap_df = pd.DataFrame(sv, columns=X_view_eval.columns)
        shap_by_model[model_name] = shap_df

        if args.per_model:
            per_model_importance[model_name] = shap_df.abs().mean(axis=0)

    if not shap_by_model:
        raise SystemExit("No SHAP values computed (all model weights are 0?).")

    union_features = sorted({c for df in shap_by_model.values() for c in df.columns})
    agg = np.zeros((len(X_eval_raw), len(union_features)), dtype=float)
    for model_name, shap_df in shap_by_model.items():
        w = weights.get(model_name, 0.0)
        aligned = shap_df.reindex(columns=union_features, fill_value=0.0).to_numpy(dtype=float)
        agg += w * aligned

    global_importance = np.mean(np.abs(agg), axis=0)
    mean_signed = np.mean(agg, axis=0)
    global_df = (
        pd.DataFrame(
            {
                "feature": union_features,
                "global_importance": global_importance,
                "mean_signed_shap": mean_signed,
            }
        )
        .sort_values("global_importance", ascending=False)
        .reset_index(drop=True)
    )

    out_path = args.out
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    with pd.ExcelWriter(out_path) as writer:
        weights_df.to_excel(writer, sheet_name="nnls_weights", index=False)
        global_df.to_excel(writer, sheet_name="global_importance", index=False)

        if args.per_model and per_model_importance:
            per_model_df = pd.DataFrame(per_model_importance).fillna(0.0)
            per_model_df.index.name = "feature"
            per_model_df = per_model_df.reset_index()
            per_model_df.to_excel(writer, sheet_name="per_model_importance", index=False)

    print(f"Saved: {out_path}")
    print("Top 20 features:")
    for row in global_df.head(20).itertuples(index=False):
        print(f"  {row.feature}: {row.global_importance:.6f}")

    # SHAP 可视化（蜂群图 + 条形图）
    try:
        import shap
        import matplotlib.pyplot as plt

        figs_dir = Path(__file__).resolve().parent / "figures"
        figs_dir.mkdir(parents=True, exist_ok=True)

        X_union_eval = X_eval_raw.reindex(columns=union_features, fill_value=0.0)
        plot_shap, plot_feature_values, plot_global_df, other_feature_count = (
            build_plot_bundle_with_other_features(
                agg=agg,
                feature_values=X_union_eval,
                global_df=global_df,
                top_n=PLOT_MAX_FEATURES,
                other_label=OTHER_FEATURES_LABEL,
            )
        )
        if other_feature_count > 0:
            print(
                f"Plotting top {PLOT_MAX_FEATURES - 1} features plus "
                f"'{OTHER_FEATURES_LABEL}' aggregated from {other_feature_count} omitted features."
            )

        # 蜂群图（全局聚合后的 shap 值）
        bee_path = figs_dir / "shap_beeswarm.png"
        plt.figure()
        plot_label_map = {
            "Gender": "Sex",
            "age": "Age",
            "mite_dust": "Dust mite",
            "mold_mix": "Mold",
        }
        plot_feature_display_names = [
            plot_label_map.get(name, name) for name in plot_feature_values.columns
        ]
        shap.summary_plot(
            plot_shap,
            features=plot_feature_values,
            feature_names=plot_feature_display_names,
            show=False,
            max_display=plot_feature_values.shape[1],
            sort=False,
        )
        plt.tight_layout()
        save_matplotlib_figure(plt.gcf(), bee_path)
        plt.close()
        print(f"Bee swarm saved: {bee_path}")

        # 条形图（mean |shap|）
        bar_path = figs_dir / "shap_bar.png"
        fig, ax = plt.subplots(figsize=(6.5, 5))
        top_df = plot_global_df.iloc[::-1].copy()
        top_df["feature"] = top_df["feature"].replace(plot_label_map)
        colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(top_df)))
        ax.barh(top_df["feature"], top_df["global_importance"], color=colors, edgecolor="none")
        ax.set_xlabel("Mean |SHAP|")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.grid(axis="x", linestyle="--", linewidth=0.6, color="#888888", alpha=0.4)
        ax.tick_params(axis="y", length=0)
        fig.tight_layout()
        save_matplotlib_figure(fig, bar_path)
        plt.close(fig)
        print(f"Bar plot saved: {bar_path}")

        # Per-base-model SHAP plots (bar + beeswarm).
        for model_name, model_shap_df in shap_by_model.items():
            model_global_df = (
                pd.DataFrame(
                    {
                        "feature": model_shap_df.columns,
                        "global_importance": model_shap_df.abs().mean(axis=0).to_numpy(dtype=float),
                        "mean_signed_shap": model_shap_df.mean(axis=0).to_numpy(dtype=float),
                    }
                )
                .sort_values("global_importance", ascending=False)
                .reset_index(drop=True)
            )
            model_feature_values = X_eval_raw.reindex(columns=model_shap_df.columns, fill_value=0.0)
            model_plot_shap, model_plot_feature_values, model_plot_df, _ = (
                build_plot_bundle_with_other_features(
                    agg=model_shap_df.to_numpy(dtype=float),
                    feature_values=model_feature_values,
                    global_df=model_global_df,
                    top_n=PLOT_MAX_FEATURES,
                    other_label=OTHER_FEATURES_LABEL,
                )
            )

            safe_model_name = "".join(
                ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in model_name
            )
            model_bee_path = figs_dir / f"base_{safe_model_name}_beeswarm.png"
            plt.figure()
            model_feature_display_names = [
                plot_label_map.get(name, name) for name in model_plot_feature_values.columns
            ]
            shap.summary_plot(
                model_plot_shap,
                features=model_plot_feature_values,
                feature_names=model_feature_display_names,
                show=False,
                max_display=model_plot_feature_values.shape[1],
                sort=False,
            )
            plt.tight_layout()
            save_matplotlib_figure(plt.gcf(), model_bee_path)
            plt.close()

            model_bar_path = figs_dir / f"base_{safe_model_name}_bar.png"
            fig, ax = plt.subplots(figsize=(6.5, 5))
            model_top_df = model_plot_df.iloc[::-1].copy()
            model_top_df["feature"] = model_top_df["feature"].replace(plot_label_map)
            model_colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(model_top_df)))
            ax.barh(
                model_top_df["feature"],
                model_top_df["global_importance"],
                color=model_colors,
                edgecolor="none",
            )
            ax.set_xlabel("Mean |SHAP|")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.grid(axis="x", linestyle="--", linewidth=0.6, color="#888888", alpha=0.4)
            ax.tick_params(axis="y", length=0)
            fig.tight_layout()
            save_matplotlib_figure(fig, model_bar_path)
            plt.close(fig)
            print(f"[{model_name}] beeswarm saved: {model_bee_path}")
            print(f"[{model_name}] bar saved: {model_bar_path}")

        try:
            combined_path = figs_dir / "shap_combined.png"
            bar_img = plt.imread(bar_path)
            bee_img = plt.imread(bee_path)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(bar_img)
            axes[0].axis("off")
            axes[0].text(
                0.03,
                0.97,
                "A",
                transform=axes[0].transAxes,
                ha="left",
                va="top",
                fontsize=16,
                fontweight="bold",
                color="black",
            )

            axes[1].imshow(bee_img)
            axes[1].axis("off")
            axes[1].text(
                0.03,
                0.97,
                "B",
                transform=axes[1].transAxes,
                ha="left",
                va="top",
                fontsize=16,
                fontweight="bold",
                color="black",
            )

            fig.tight_layout()
            save_matplotlib_figure(fig, combined_path)
            plt.close(fig)
            print(f"Combined plot saved: {combined_path}")
        except Exception as combine_exc:
            print(f"Warning: failed to combine SHAP plots: {combine_exc}")
    except Exception as exc:  # pragma: no cover
        print(f"Warning: failed to plot SHAP figures: {exc}")


if __name__ == "__main__":
    main()

# %%
