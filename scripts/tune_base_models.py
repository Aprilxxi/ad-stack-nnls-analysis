# %%
"""
Bayesian hyperparameter tuning for base models.

Runs Optuna TPE optimization with view-specific preprocessing that matches
stack_views_nnls.py and writes best params to a JSON file.
"""
from __future__ import annotations

import argparse
import copy
import json

import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from sklearn import clone
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

try:
    import optuna
except ImportError as exc:  # pragma: no cover - runtime import guard
    raise SystemExit("Optuna is required: pip install optuna") from exc

import stack_views_nnls as base


MetricFn = Callable[[np.ndarray, np.ndarray], float]
DATA_FILE = "train_pruned_ratio_0p04.xlsx"
DEFAULT_MODELS = (
    "blood_xgb,blood_extratrees,allergen_cat,history_logreg"
)
try:
    _CATBOOST_SUPPORTS_ALLOW_WRITING_FILES = (
        "allow_writing_files" in base.CatBoostClassifier().get_params()
    )
except Exception:
    _CATBOOST_SUPPORTS_ALLOW_WRITING_FILES = False


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune base models with Bayesian optimization (Optuna)."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=DEFAULT_MODELS,
        help="Comma-separated model names to tune.",
    )
    parser.add_argument("--trials", type=int, default=50, help="Number of trials.")
    parser.add_argument(
        "--metric",
        type=str,
        default="auc",
        choices=["auc", "recall", "f1", "accuracy"],
        help="Optimization metric.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=1,
        help="CV splits. Set <=1 to use a single holdout split.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Holdout validation size when splits <= 1.",
    )
    parser.add_argument("--seed", type=int, default=base.RANDOM_STATE, help="Random seed.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds per model (default: 3600).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("best_params.json"),
        help="Output JSON path.",
    )
    parser.add_argument("--study-name", type=str, default=None, help="Optuna study name.")
    return parser.parse_args(argv)


def build_views() -> Dict[str, List[str]]:
    blood_view = base.apply_demo_policy(base.BLOOD_VIEW, "blood")
    allergen_view = base.apply_demo_policy(base.ALLERGEN_VIEW, "allergen")
    # Drop Total IgE (and its log variant) from the allergen view.
    allergen_view = [
        c
        for c in allergen_view
        if c.lower() not in {"total ige", "total ige_log1p"}
    ]
    history_view = base.apply_demo_policy(base.HISTORY_VIEW, "history")
    return {
        "blood_xgb": blood_view,
        "blood_extratrees": blood_view,
        "allergen_cat": allergen_view,
        "history_rf": history_view,
        "history_logreg": history_view,
    }


def build_scale_flags() -> Dict[str, bool]:
    return {
        "blood_xgb": False,
        "blood_extratrees": False,
        "allergen_cat": False,
        "history_rf": False,
        "history_logreg": True,
    }


def build_model_from_trial(
    model_name: str, trial: optuna.Trial, seed: int
) -> Tuple[object, Dict[str, object], Dict[str, object]]:
    if model_name == "blood_xgb":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 100.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        fixed = {
            "random_state": seed,
            "eval_metric": "logloss",
            "n_jobs": -1,
            "scale_pos_weight": 1.0,
        }
        model = base.xgb.XGBClassifier(**params, **fixed)
        return model, params, fixed

    if model_name == "blood_extratrees":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "max_depth": trial.suggest_categorical(
                "max_depth", [None, 4, 6, 8, 10, 12, 16, 20]
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
        fixed = {
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": seed,
        }
        model = base.ExtraTreesClassifier(**params, **fixed)
        return model, params, fixed

    if model_name == "allergen_cat":
        params = {
            "iterations": trial.suggest_int("iterations", 300, 1200),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
        fixed = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "class_weights": [2.0, 0.8],
            "random_seed": seed,
            "verbose": 0,
        }
        if _CATBOOST_SUPPORTS_ALLOW_WRITING_FILES:
            fixed["allow_writing_files"] = False
        model = base.CatBoostClassifier(**params, **fixed)
        return model, params, fixed

    if model_name == "history_rf":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_categorical(
                "max_depth", [None, 4, 6, 8, 10, 12]
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
        fixed = {
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": seed,
        }
        model = base.RandomForestClassifier(**params, **fixed)
        return model, params, fixed

    if model_name == "history_logreg":
        params = {"C": trial.suggest_float("C", 0.01, 10.0, log=True)}
        fixed = {
            "penalty": "l2",
            "solver": "liblinear",
            "max_iter": 500,
            "class_weight": "balanced",
            "random_state": seed,
        }
        model = base.LogisticRegression(**params, **fixed)
        return model, params, fixed

    raise ValueError(f"Unknown model name: {model_name}")


def get_metric_fn(name: str) -> MetricFn:
    if name == "auc":
        return lambda y_true, y_prob: roc_auc_score(y_true, y_prob)
    if name == "recall":
        return lambda y_true, y_prob: recall_score(y_true, (y_prob >= 0.5).astype(int))
    if name == "f1":
        return lambda y_true, y_prob: f1_score(y_true, (y_prob >= 0.5).astype(int))
    if name == "accuracy":
        return lambda y_true, y_prob: accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    raise ValueError(f"Unknown metric: {name}")


def normalize_model_list(models: str | Sequence[str] | None) -> List[str]:
    if models is None:
        return [m for m in DEFAULT_MODELS.split(",") if m]
    if isinstance(models, str):
        return [m.strip() for m in models.split(",") if m.strip()]
    return [m for m in models if m]


def evaluate_model_holdout(
    base_model: object,
    X: np.ndarray,
    y: np.ndarray,
    view_cols: List[str],
    categorical_cols: List[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    scale_numeric: bool,
    metric_fn: MetricFn,
) -> float:
    X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    X_train, X_test = base.prepare_view_split(
        X_train_raw, X_test_raw, view_cols, categorical_cols, scale_numeric
    )
    X_train_bal, y_train_bal = base.resample_training_data(
        X_train, y_train, categorical_cols
    )

    try:
        model = clone(base_model)
    except Exception:
        model = copy.deepcopy(base_model)
    model.fit(X_train_bal, y_train_bal)
    prob = model.predict_proba(X_test)[:, 1]
    return metric_fn(y_test, prob)


def evaluate_model_cv(
    base_model: object,
    X: np.ndarray,
    y: np.ndarray,
    view_cols: List[str],
    categorical_cols: List[str],
    kf: StratifiedKFold,
    scale_numeric: bool,
    metric_fn: MetricFn,
) -> float:
    scores: List[float] = []
    for train_idx, test_idx in kf.split(X, y):
        score = evaluate_model_holdout(
            base_model,
            X,
            y,
            view_cols,
            categorical_cols,
            train_idx,
            test_idx,
            scale_numeric,
            metric_fn,
        )
        scores.append(score)

    return float(np.mean(scores)) if scores else float("nan")


def tune_one_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    view_cols: List[str],
    categorical_cols: List[str],
    scale_numeric: bool,
    metric_fn: MetricFn,
    splits: int,
    val_size: float,
    trials: int,
    timeout: int | None,
    seed: int,
    study_name: str | None,
) -> Dict[str, object]:
    def count_trials(study_obj: optuna.Study) -> Tuple[int, int]:
        total = len(study_obj.trials)
        completed = sum(
            1 for t in study_obj.trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        return total, completed

    def objective(trial: optuna.Trial) -> float:
        model, _, _ = build_model_from_trial(model_name, trial, seed=seed)
        kf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
        return evaluate_model_cv(
            model, X, y, view_cols, categorical_cols, kf, scale_numeric, metric_fn
        )

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, study_name=study_name
    )
    if splits <= 1:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size, random_state=seed
        )
        train_idx, test_idx = next(splitter.split(X, y))
    interrupted = False
    try:
        if splits <= 1:
            def objective_holdout(trial: optuna.Trial) -> float:
                model, _, _ = build_model_from_trial(model_name, trial, seed=seed)
                return evaluate_model_holdout(
                    model,
                    X,
                    y,
                    view_cols,
                    categorical_cols,
                    train_idx,
                    test_idx,
                    scale_numeric,
                    metric_fn,
                )

            study.optimize(objective_holdout, n_trials=trials, timeout=timeout)
        else:
            study.optimize(objective, n_trials=trials, timeout=timeout)
    except KeyboardInterrupt:
        interrupted = True
        total, completed = count_trials(study)
        print(f"Interrupted {model_name} after {total} trials (completed: {completed}).")

    total, completed = count_trials(study)
    try:
        best_trial = study.best_trial
    except ValueError:
        return {
            "best_score": float("nan"),
            "best_params": {},
            "fixed_params": {},
            "trials": total,
            "completed_trials": completed,
            "interrupted": interrupted,
        }

    _, _, fixed_params = build_model_from_trial(model_name, best_trial, seed=seed)
    return {
        "best_score": best_trial.value,
        "best_params": best_trial.params,
        "fixed_params": fixed_params,
        "trials": total,
        "completed_trials": completed,
        "interrupted": interrupted,
    }


def run_tuning(
    models: str | Sequence[str] | None = None,
    trials: int = 50,
    metric: str = "auc",
    splits: int | None = 1,
    val_size: float = 0.2,
    seed: int | None = None,
    timeout: int | None = 3600,
    out: str | Path | None = "best_params.json",
    study_name: str | None = None,
) -> Dict[str, object]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if splits is None:
        splits = 1
    if seed is None:
        seed = base.RANDOM_STATE

    # Keep tuning on the intended pruned training dataset even if base defaults change.
    base.DATA_FILE = DATA_FILE
    X, y = base.load_dataset()
    metric_fn = get_metric_fn(metric)
    views = build_views()
    scale_flags = build_scale_flags()
    categorical_cols = list(base.DEFAULT_CATEGORICAL)

    model_names = normalize_model_list(models)
    supported = set(views)
    unknown = [m for m in model_names if m not in supported]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Supported: {sorted(supported)}")

    results: Dict[str, object] = {}
    start_ts = time.time()
    interrupted = False
    for model_name in model_names:
        print(f"Tuning {model_name}...")
        result = tune_one_model(
            model_name=model_name,
            X=X,
            y=y,
            view_cols=views[model_name],
            categorical_cols=categorical_cols,
            scale_numeric=scale_flags[model_name],
            metric_fn=metric_fn,
            splits=splits,
            val_size=val_size,
            trials=trials,
            timeout=timeout,
            seed=seed,
            study_name=study_name,
        )
        results[model_name] = result
        if result.get("interrupted"):
            interrupted = True
        if not np.isnan(result["best_score"]):
            print(f"{model_name} best {metric}: {result['best_score']:.4f}")
        else:
            print(f"{model_name} best {metric}: N/A (no completed trials)")

    payload = {
        "metric": metric,
        "splits": splits,
        "val_size": val_size,
        "seed": seed,
        "elapsed_seconds": round(time.time() - start_ts, 2),
        "interrupted": interrupted,
        "results": results,
    }
    if out is not None:
        out_path = Path(out)
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        print(f"Saved: {out_path}")

    return payload


def main() -> None:
    args = parse_args()
    try:
        run_tuning(
            models=args.models,
            trials=args.trials,
            metric=args.metric,
            splits=args.splits,
            val_size=args.val_size,
            seed=args.seed,
            timeout=args.timeout,
            out=args.out,
            study_name=args.study_name,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
