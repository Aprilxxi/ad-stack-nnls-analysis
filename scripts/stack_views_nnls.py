"""
View-specific base models + non-negative stacking (NNLS) on asthma dataset.

Base learners are configured in build_models() and trained per-view with
stratified CV. Stacking uses non-negative linear regression on OOF
logits (log-odds) of base probabilities with an intercept.
Development evaluation pools five-fold NNLS OOF predictions on fixed base OOF
features; this does not constitute outer validation of the complete pipeline.
"""

# %%
from __future__ import annotations

import copy
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
try:
    from imblearn.over_sampling import SMOTEN
except ImportError:  # pragma: no cover - older imbalanced-learn
    SMOTEN = None
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import KNNImputer, SimpleImputer


RANDOM_STATE = 42
K_SPLITS = 5
BASE_DIR = Path(__file__).resolve().parent
# DATA_FILE = "train_full_merged_allergen_with_record_count.xlsx"
DATA_FILE = 'train_pruned_ratio_0p04.xlsx'
BEST_PARAMS_FILE = "best_params.json"
DROP_COLUMNS = ["Unique patient number", "Date of birth", "diagnosis"]
TARGET = "Asthma"
KNN_IMPUTER_NEIGHBORS = 7
SMOTE_NEIGHBORS = 5
RECORD_COUNT_COL = "病历记录数量"
RECORD_COUNT_MIN = 5
USE_ENGINEERED_ALLERGEN_FEATURES = False
PROBA_EPS = 1e-6
LOGIT_MIN = -5.0
LOGIT_MAX = 5.0
BOOTSTRAP_ROUNDS = 5000
ENABLE_BOOTSTRAP = True
PREDICTION_CACHE_PATH: Optional[Path] = BASE_DIR / "prediction_cache.pkl"
BOOTSTRAP_RAW_CACHE_PATH: Optional[Path] = BASE_DIR / "bootstrap_raw_cache.pkl"
BOOTSTRAP_SUMMARY_CACHE_PATH: Optional[Path] = BASE_DIR / "bootstrap_summary_cache.pkl"
PREDICTION_ENTRY_CACHE_DIR: Optional[Path] = BASE_DIR / "prediction_cache_entries"
BOOTSTRAP_RAW_ENTRY_CACHE_DIR: Optional[Path] = BASE_DIR / "bootstrap_raw_cache_entries"
BOOTSTRAP_SUMMARY_ENTRY_CACHE_DIR: Optional[Path] = BASE_DIR / "bootstrap_summary_cache_entries"
CACHE_SCHEMA_VERSION = 4
NNLS_EVALUATION_VERSION = "pooled_meta_oof_v1"
IGNORED_ALLERGEN_FEATURES = {"total ige", "total ige_log1p"}
BASE_DIR = Path(__file__).resolve().parent

DEMO_COLS = ["Gender", "age"]
# "all": keep demo features in all views
# "history_only": keep demo features only in DEMO_KEEP_VIEW
# "separate": drop from views and add a dedicated demo model
# "none": drop from all views without adding a demo model
DEMO_POLICY = "all"

# View-specific features (aligned with pred_xiaochuan.py)
BLOOD_VIEW = [
    "Hemoglobin concentration",
    "Absolute value of monocytes",
    "Percentage of basophils",
    "Absolute value of basophils",
    "Percentage of eosinophils",
    "Absolute value of eosinophils",
    "Large platelet ratio",
    "Mean corpuscular volume",
    "Mean platelet volume",
    "Percentage of neutrophils",
    "Absolute value of neutrophils",
    "Percentage of monocytes",
    "Average hemoglobin content",
    "White blood cell count",
    "Red blood cell distribution width CV",
    "Red blood cell distribution width SD",
    "Hematocrit",
    "Red blood cell count",
    "Platelet distribution width",
    "Platelet hematocrit",
    "Platelet count",
    "Average hemoglobin concentration",
    "Percentage of lymphocytes",
    "Absolute value of lymphocytes",
    "C-reactive protein",
    "Gender",
    "age",
]

ALLERGEN_TEST_COLS = [
    "House dust mite/dust mite",
    "Cat hair",
    "Cockroach",
    "dog fur",
    "Milk",
    "Egg white",
    "shrimp",
    "Peanut",
    "crab",
    "Humulus scandens",
    "Cross-reactive carbohydrate antigenic determinants",
    "Mugwort",
    "Beef",
    "Willow/Poplar/elm",
    "Mutton",
    "House dust",
    "Penicillium penicillium/Mycospora/Aspergillus fumigatus/Cladosporium",
    "Common ragweed",
    "Cod/lobster/scallops",
    "Soybeans",
]

# Merge related allergen tests to capture sparse but clinically similar signals.
MERGED_ALLERGEN_MAPPING: Dict[str, List[str]] = {
    "mite_dust": ["House dust mite/dust mite", "House dust"],
    "cat_dander": ["Cat hair"],
    "dog_dander": ["dog fur"],
    "cockroach": ["Cockroach"],
    "milk": ["Milk"],
    "egg_white": ["Egg white"],
    "beef_mutton": ["Beef", "Mutton"],
    "peanut_soy": ["Peanut", "Soybeans"],
    "shrimp_crab": ["shrimp", "crab"],
    "fish_mix": ["Cod/lobster/scallops"],
    "mold_mix": ["Penicillium penicillium/Mycospora/Aspergillus fumigatus/Cladosporium"],
    "mugwort": ["Mugwort"],
    "humulus": ["Humulus scandens"],
    "ragweed": ["Common ragweed"],
    "tree_pollen": ["Willow/Poplar/elm"],
    "house_dust": ["House dust"],
    "ccd": ["Cross-reactive carbohydrate antigenic determinants"],
}
MERGED_ALLERGEN_COLS = list(MERGED_ALLERGEN_MAPPING.keys())

ALLERGEN_RAW_NUMERIC = [
    "Total IgE",
]

ALLERGEN_ENGINEERED_NUMERIC = [
    "Total IgE_log1p",
    "allergen_positive_count",
    "allergen_positive_rate",
    "allergen_merged_positive_count",
    "allergen_merged_positive_rate",
]

ALLERGEN_VIEW = list(dict.fromkeys(MERGED_ALLERGEN_COLS + ALLERGEN_RAW_NUMERIC + DEMO_COLS))
# Drop Total IgE from the allergen view
ALLERGEN_VIEW = [c for c in ALLERGEN_VIEW if c.lower() not in IGNORED_ALLERGEN_FEATURES]
if USE_ENGINEERED_ALLERGEN_FEATURES:
    ALLERGEN_VIEW = list(
        dict.fromkeys(
            ALLERGEN_VIEW + ALLERGEN_ENGINEERED_NUMERIC
        )
    )

HISTORY_BASE_COLS = [
    "Urticaria - Past",
    "Rhinitis - past",
    "Urticaria - Family history",
    "Rhinitis - Family history",
    "Asthma - Family history",
]
HISTORY_VIEW = HISTORY_BASE_COLS + DEMO_COLS

# Default categorical columns: merged allergen/history binaries + Gender
DEFAULT_CATEGORICAL = set(MERGED_ALLERGEN_COLS + HISTORY_BASE_COLS + ["Gender"])


PLOT_COLORS = {
    # High-contrast, colorblind-friendly model palette.
    "blood_extratrees": "#FF7F0E",
    "blood_xgb": "#1F77B4",
    "allergen_cat": "#2CA02C",
    "history_logreg": "#9467BD",
    "stack_nnls": "#D62728",
}
FALLBACK_COLORS = [
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#17BECF",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
]
FIG_SIZE_STANDARD = (7.2, 5.6)
FIG_SIZE_COMPACT = (6.6, 5.2)
PLOT_DPI = 300
MODEL_LINE_WIDTH = 2.1
REF_LINE_WIDTH = 1.2
FOLD_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
]


def setup_publication_style() -> None:
    """Apply a restrained publication-style look for all plots."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4b5563",
            "axes.labelcolor": "#1f2937",
            "axes.titlesize": 12.5,
            "axes.titleweight": "normal",
            "axes.labelsize": 11,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "grid.color": "#d1d5db",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.45,
        }
    )


def save_figure_with_pdf(fig: plt.Figure, out_path: Path, **savefig_kwargs) -> Path:
    """Save a figure as PNG plus a same-name PDF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PLOT_DPI, **savefig_kwargs)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=PLOT_DPI, **savefig_kwargs)
    return out_path


def format_model_name(name: str) -> str:
    parts = name.split("_")
    return " ".join(p.upper() if p in {"xgb", "nnls"} else p.capitalize() for p in parts)


def get_model_color(name: str, idx: int) -> str:
    return PLOT_COLORS.get(name, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


def get_fold_color(fold_idx: int) -> str:
    idx = (max(1, int(fold_idx)) - 1) % len(FOLD_COLORS)
    return FOLD_COLORS[idx]


def style_axis(ax: plt.Axes, with_grid: bool = False) -> None:
    ax.set_axisbelow(True)
    if with_grid:
        ax.grid(True)
    else:
        ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4b5563")
    ax.spines["bottom"].set_color("#4b5563")
    ax.tick_params(labelsize=10, direction="out", width=0.8, length=4)


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


def set_pr_axis_limits(
    ax: plt.Axes,
    precision_min: float,
    precision_max: float,
    prevalence: float | None = None,
) -> None:
    """Set compact PR-axis limits while keeping prevalence near the lower bound."""
    if not np.isfinite(precision_min) or not np.isfinite(precision_max):
        ax.set_ylim(0.0, 1.02)
        return

    y_min = float(precision_min)
    y_max = float(precision_max)
    prev = float(prevalence) if prevalence is not None and np.isfinite(prevalence) else None
    if prev is not None:
        y_min = min(y_min, prev)
        y_max = max(y_max, prev)

    if prev is not None:
        # Keep prevalence slightly above 0 when curves allow it.
        lower = max(0.0, prev - 0.02)
        if y_min < lower:
            lower = max(0.0, y_min - 0.01)
    else:
        span = max(0.05, y_max - y_min)
        lower = max(0.0, y_min - 0.08 * span)

    span = max(0.06, y_max - lower)
    upper = min(1.0, y_max + 0.12 * span)
    if upper <= lower + 0.06:
        upper = min(1.0, lower + 0.06)
    ax.set_ylim(lower, min(1.02, upper + 0.005))


def has_finite_ci(ci_value: object) -> bool:
    try:
        if ci_value is None:
            return False
        lo = float(ci_value[0])  # type: ignore[index]
        hi = float(ci_value[1])  # type: ignore[index]
    except Exception:
        return False
    return bool(np.isfinite(lo) and np.isfinite(hi))


def hash_ndarray(arr: np.ndarray) -> str:
    arr_c = np.ascontiguousarray(arr)
    h = hashlib.sha1()
    h.update(str(arr_c.dtype).encode("utf-8"))
    h.update(np.asarray(arr_c.shape, dtype=np.int64).tobytes())
    h.update(arr_c.tobytes())
    return h.hexdigest()


def hash_series(series: pd.Series) -> str:
    vals = pd.util.hash_pandas_object(series, index=True).to_numpy(dtype=np.uint64, copy=False)
    return hashlib.sha1(vals.tobytes()).hexdigest()


def hash_dataframe(df: pd.DataFrame) -> str:
    vals = pd.util.hash_pandas_object(df, index=True).to_numpy(dtype=np.uint64, copy=False)
    return hashlib.sha1(vals.tobytes()).hexdigest()


def stable_hash_obj(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def model_config_signature(
    models: Dict[str, Tuple[object, Sequence[str], bool]],
) -> str:
    rows = []
    for name in sorted(models):
        model, view, scale_flag = models[name]
        try:
            params = model.get_params(deep=True)  # type: ignore[attr-defined]
        except Exception:
            params = {"repr": repr(model)}
        rows.append(
            {
                "name": name,
                "scale_numeric": bool(scale_flag),
                "view_cols": list(view),
                "params": params,
            }
        )
    return stable_hash_obj(rows)


def make_prediction_cache_key(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, Tuple[object, Sequence[str], bool]],
) -> str:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "nnls_evaluation_version": NNLS_EVALUATION_VERSION,
        "data_file": DATA_FILE,
        "x_hash": hash_dataframe(X),
        "y_hash": hash_series(y),
        "model_sig": model_config_signature(models),
        "k_splits": K_SPLITS,
        "random_state": RANDOM_STATE,
        "demo_policy": DEMO_POLICY,
        "use_engineered_allergen_features": USE_ENGINEERED_ALLERGEN_FEATURES,
        "proba_eps": PROBA_EPS,
        "logit_min": LOGIT_MIN,
        "logit_max": LOGIT_MAX,
    }
    return stable_hash_obj(payload)


def make_bootstrap_cache_key(
    label: str,
    calibration_tag: str,
    y_true: np.ndarray,
    proba: np.ndarray,
    n_boot: int,
    seed: int,
    grid: np.ndarray,
) -> str:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "label": label,
        "calibration_tag": calibration_tag,
        "y_hash": hash_ndarray(np.asarray(y_true, dtype=int)),
        "p_hash": hash_ndarray(np.asarray(proba, dtype=float)),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "grid_hash": hash_ndarray(np.asarray(grid, dtype=float)),
    }
    return stable_hash_obj(payload)


def load_layer_cache(cache_path: Optional[Path], layer_name: str) -> Dict[str, object]:
    if cache_path is None or not cache_path.exists():
        return {}
    try:
        raw = pickle.loads(cache_path.read_bytes())
    except Exception as exc:  # pragma: no cover
        print(f"Warning: failed to load {layer_name} cache {cache_path}: {exc}")
        return {}
    if not isinstance(raw, dict):
        print(f"Warning: {layer_name} cache {cache_path} is invalid; ignoring.")
        return {}

    entries = raw.get("entries", raw)
    if not isinstance(entries, dict):
        print(f"Warning: {layer_name} cache {cache_path} has non-dict entries; ignoring.")
        return {}

    print(f"Loaded {layer_name} cache from {cache_path} (entries={len(entries)})")
    return entries


def save_layer_cache(cache_path: Optional[Path], entries: Dict[str, object], layer_name: str) -> None:
    if cache_path is None:
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "layer": layer_name,
            "entries": entries,
        }
        cache_path.write_bytes(pickle.dumps(payload))
        print(f"{layer_name} cache saved to {cache_path} (entries={len(entries)})")
    except Exception as exc:  # pragma: no cover
        print(f"Warning: failed to save {layer_name} cache {cache_path}: {exc}")


def load_entry_cache(cache_dir: Optional[Path], cache_key: str, layer_name: str) -> object:
    if cache_dir is None:
        return None
    path = cache_dir / f"{cache_key}.pkl"
    if not path.exists():
        return None
    try:
        raw = pickle.loads(path.read_bytes())
    except Exception as exc:  # pragma: no cover
        print(f"Warning: failed to load {layer_name} entry {path}: {exc}")
        return None
    if isinstance(raw, dict) and "entry" in raw:
        return raw["entry"]
    return raw


def save_entry_cache(cache_dir: Optional[Path], cache_key: str, entry: object, layer_name: str) -> None:
    if cache_dir is None:
        return
    path = cache_dir / f"{cache_key}.pkl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "layer": layer_name,
            "cache_key": cache_key,
            "entry": entry,
        }
        path.write_bytes(pickle.dumps(payload))
    except Exception as exc:  # pragma: no cover
        print(f"Warning: failed to save {layer_name} entry {path}: {exc}")


def load_prediction_model_entry(prediction_cache_key: str, model_name: str) -> object:
    if PREDICTION_ENTRY_CACHE_DIR is None:
        return None
    path = PREDICTION_ENTRY_CACHE_DIR / prediction_cache_key / f"{model_name}.pkl"
    if not path.exists():
        return None
    try:
        raw = pickle.loads(path.read_bytes())
    except Exception as exc:  # pragma: no cover
        print(f"Warning: failed to load prediction entry {path}: {exc}")
        return None
    if isinstance(raw, dict) and "entry" in raw:
        return raw["entry"]
    return raw


def save_prediction_model_entry(prediction_cache_key: str, model_name: str, entry: object) -> None:
    if PREDICTION_ENTRY_CACHE_DIR is None:
        return
    path = PREDICTION_ENTRY_CACHE_DIR / prediction_cache_key / f"{model_name}.pkl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "layer": "prediction_entry",
            "prediction_cache_key": prediction_cache_key,
            "model": model_name,
            "entry": entry,
        }
        path.write_bytes(pickle.dumps(payload))
        print(f"Saved base-model prediction entry: {model_name}")
    except Exception as exc:  # pragma: no cover
        print(f"Warning: failed to save prediction entry {path}: {exc}")


def apply_demo_policy(view_cols: Sequence[str], view_name: str) -> List[str]:
    if DEMO_POLICY == "all":
        return list(view_cols)
    if DEMO_POLICY in {"none", "separate"}:
        return [c for c in view_cols if c not in DEMO_COLS]
    if DEMO_POLICY == "history_only" and view_name != "history":
        return [c for c in view_cols if c not in DEMO_COLS]
    return list(view_cols)


def load_best_params() -> Dict[str, dict]:
    path = Path(__file__).resolve().parent / BEST_PARAMS_FILE
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print(f"Warning: failed to read {path}; skip tuned params.")
        return {}
    if isinstance(data, dict) and isinstance(data.get("results"), dict):
        return data["results"]
    if isinstance(data, dict):
        return data
    return {}


def apply_tuned_params(model, model_name: str, tuned: Dict[str, dict]) -> None:
    entry = tuned.get(model_name)
    if not isinstance(entry, dict):
        return
    merged = {}
    for key in ("fixed_params", "best_params"):
        params = entry.get(key)
        if isinstance(params, dict):
            merged.update(params)
    if not merged:
        return
    try:
        valid = model.get_params()
    except Exception:
        merged.pop("allow_writing_files", None)
        if merged:
            model.set_params(**merged)
        return
    if "allow_writing_files" in merged and "allow_writing_files" not in valid:
        merged.pop("allow_writing_files", None)
    filtered = {k: v for k, v in merged.items() if k in valid}
    unknown = [k for k in merged if k not in valid]
    if unknown:
        unknown.sort()
        print(f"Warning: {model_name} ignored params: {unknown}")
    if filtered:
        model.set_params(**filtered)


def pick_data_file() -> Path:
    """Resolve and validate the configured dataset file path."""
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / DATA_FILE
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return data_path


def add_allergen_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if "Total IgE" in X.columns:
        ige = pd.to_numeric(X["Total IgE"], errors="coerce")
        X["Total IgE_log1p"] = np.log1p(ige.clip(lower=0))

    raw_cols = [c for c in ALLERGEN_TEST_COLS if c in X.columns]

    for merged_name, cols in MERGED_ALLERGEN_MAPPING.items():
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            continue
        if merged_name not in X.columns:
            X[merged_name] = (X[valid_cols] > 0).any(axis=1).astype(int)

    merged_cols = [c for c in MERGED_ALLERGEN_COLS if c in X.columns]
    count_source_cols = raw_cols if raw_cols else merged_cols
    if count_source_cols:
        vals = X[count_source_cols]
        pos_count = (vals > 0).sum(axis=1)
        tested = vals.notna().sum(axis=1).replace(0, np.nan)
        X["allergen_positive_count"] = pos_count
        X["allergen_positive_rate"] = (pos_count / tested).fillna(0)

    if merged_cols:
        merged_vals = X[merged_cols]
        merged_pos = (merged_vals > 0).sum(axis=1)
        merged_tested = merged_vals.notna().sum(axis=1).replace(0, np.nan)
        X["allergen_merged_positive_count"] = merged_pos
        X["allergen_merged_positive_rate"] = (merged_pos / merged_tested).fillna(0)

    return X


def load_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset, apply filters, and return feature matrix and target."""
    path = pick_data_file()
    df = pd.read_excel(path)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    if RECORD_COUNT_COL in df.columns:
        record_counts = pd.to_numeric(df[RECORD_COUNT_COL], errors="coerce")
        df = df[record_counts > RECORD_COUNT_MIN].reset_index(drop=True)
    else:
        print(f"Warning: {RECORD_COUNT_COL} not found; skipping record-count filter.")
    if "age" in df.columns:
        df = df[df["age"] <= 8].reset_index(drop=True)
    if TARGET not in df.columns:
        raise KeyError(f"Target column {TARGET} missing.")
    y = df[TARGET].astype(int)
    drop_cols = [TARGET]
    if RECORD_COUNT_COL in df.columns:
        drop_cols.append(RECORD_COUNT_COL)
    X = df.drop(columns=drop_cols)
    if USE_ENGINEERED_ALLERGEN_FEATURES:
        X = add_allergen_features(X)
    return X, y


def prepare_view_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    view_cols: Sequence[str],
    categorical_cols: Sequence[str],
    scale_numeric: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Impute per-view features using train-fit and test-transform to avoid leakage."""
    cols = [c for c in view_cols if c in train_df.columns]
    cat_cols = [c for c in cols if c in categorical_cols]
    num_cols = [c for c in cols if c not in cat_cols]

    train_sel = train_df[cols].copy()
    test_sel = test_df[cols].copy()

    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        train_cat = pd.DataFrame(
            cat_imputer.fit_transform(train_sel[cat_cols]),
            columns=cat_cols,
            index=train_sel.index,
        )
        test_cat = pd.DataFrame(
            cat_imputer.transform(test_sel[cat_cols]),
            columns=cat_cols,
            index=test_sel.index,
        )
        cat_min = train_sel[cat_cols].min().fillna(0)
        cat_max = train_sel[cat_cols].max().fillna(1)
        train_cat = train_cat.round().clip(lower=cat_min, upper=cat_max, axis="columns")
        test_cat = test_cat.round().clip(lower=cat_min, upper=cat_max, axis="columns")
        train_cat = train_cat.astype(int)
        test_cat = test_cat.astype(int)
    else:
        train_cat = pd.DataFrame(index=train_sel.index)
        test_cat = pd.DataFrame(index=test_sel.index)

    if num_cols:
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_sel[num_cols])
        test_scaled = scaler.transform(test_sel[num_cols])
        imputer = KNNImputer(n_neighbors=KNN_IMPUTER_NEIGHBORS)
        train_imp = imputer.fit_transform(train_scaled)
        test_imp = imputer.transform(test_scaled)
        if scale_numeric:
            train_num = train_imp
            test_num = test_imp
        else:
            train_num = scaler.inverse_transform(train_imp)
            test_num = scaler.inverse_transform(test_imp)
        train_num_df = pd.DataFrame(train_num, columns=num_cols, index=train_sel.index)
        test_num_df = pd.DataFrame(test_num, columns=num_cols, index=test_sel.index)
    else:
        train_num_df = pd.DataFrame(index=train_sel.index)
        test_num_df = pd.DataFrame(index=test_sel.index)

    X_train = pd.concat([train_cat, train_num_df], axis=1)
    X_test = pd.concat([test_cat, test_num_df], axis=1)
    X_train = X_train.reindex(columns=cols)
    X_test = X_test.reindex(columns=cols)
    return X_train, X_test


def prepare_view(
    X: pd.DataFrame,
    view_cols: Sequence[str],
    categorical_cols: Sequence[str],
    scale_numeric: bool = False,
) -> pd.DataFrame:
    X_train, _ = prepare_view_split(X, X, view_cols, categorical_cols, scale_numeric)
    return X_train


def resample_training_data(
    X_train: pd.DataFrame, y_train: pd.Series, categorical_cols: Sequence[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    if y_train.nunique() < 2:
        return X_train, y_train
    class_counts = y_train.value_counts()
    min_count = int(class_counts.min())
    if min_count < 2:
        return X_train, y_train
    k_neighbors = min(SMOTE_NEIGHBORS, min_count - 1)
    if k_neighbors < 1:
        return X_train, y_train

    cat_cols = [c for c in categorical_cols if c in X_train.columns]
    if cat_cols:
        if len(cat_cols) == X_train.shape[1]:
            if SMOTEN is not None:
                sampler = SMOTEN(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
            else:
                sampler = RandomOverSampler(random_state=RANDOM_STATE)
        else:
            cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]
            sampler = SMOTENC(
                categorical_features=cat_indices,
                random_state=RANDOM_STATE,
                k_neighbors=k_neighbors,
            )
    else:
        sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)
    return X_res, y_res


def _clip_proba(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.clip(p, PROBA_EPS, 1.0 - PROBA_EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip_proba(p)
    z = np.log(p / (1.0 - p))
    return np.clip(z, LOGIT_MIN, LOGIT_MAX)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def compute_decision_curve(
    y_true: np.ndarray, proba: np.ndarray, thresholds: Sequence[float]
) -> pd.DataFrame:
    """Compute net benefit across thresholds for a single model."""
    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(proba, dtype=float)
    n = len(y_arr)
    rows = []
    for thr in thresholds:
        if thr <= 0.0 or thr >= 1.0:
            continue
        pred = p_arr >= thr
        tp = ((pred == 1) & (y_arr == 1)).sum()
        fp = ((pred == 1) & (y_arr == 0)).sum()
        nb = (tp / n) - (fp / n) * (thr / (1.0 - thr))
        rows.append((thr, nb))
    if not rows:
        return pd.DataFrame(columns=["threshold", "net_benefit", "treat_all", "treat_none"])
    df = pd.DataFrame(rows, columns=["threshold", "net_benefit"])
    prevalence = y_arr.mean()
    df["treat_all"] = prevalence - (1.0 - prevalence) * (df["threshold"] / (1.0 - df["threshold"]))
    df["treat_none"] = 0.0
    return df


def threshold_metric_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: Sequence[float],
) -> pd.DataFrame:
    """Evaluate metrics across a threshold grid for one model."""
    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(proba, dtype=float)
    if y_arr.shape[0] != p_arr.shape[0] or y_arr.shape[0] == 0:
        return pd.DataFrame(
            columns=[
                "threshold",
                "accuracy",
                "recall",
                "specificity",
                "f1",
                "youden_j",
            ]
        )

    rows: List[dict] = []
    for thr in thresholds:
        t = float(thr)
        if t <= 0.0 or t >= 1.0:
            continue
        pred = (p_arr >= t).astype(int)
        acc = float(accuracy_score(y_arr, pred))
        rec = float(recall_score(y_arr, pred))
        f1 = float(f1_score(y_arr, pred))
        tn = int(((y_arr == 0) & (pred == 0)).sum())
        fp = int(((y_arr == 0) & (pred == 1)).sum())
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
        youden_j = float(rec + spec - 1.0) if np.isfinite(spec) else float("nan")
        rows.append(
            {
                "threshold": t,
                "accuracy": acc,
                "recall": rec,
                "specificity": spec,
                "f1": f1,
                "youden_j": youden_j,
            }
        )
    return pd.DataFrame(rows)


def plot_threshold_metric_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    out_path: Path,
    model_name: str = "stack_nnls",
    thresholds: Sequence[float] | None = None,
) -> None:
    """Plot metric-vs-threshold curves for one model."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    df = threshold_metric_curve(y_true=y_true, proba=proba, thresholds=thresholds)
    if df.empty:
        print("Threshold metric curve: no data to plot.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD, facecolor="white")
    style_axis(ax, with_grid=True)

    curve_spec = [
        ("accuracy", "Accuracy", "#6b7280", "-", 2.0),
        ("recall", "Recall", "#d97706", "-", 2.0),
        ("specificity", "Specificity", "#7c3aed", "-", 2.0),
        ("f1", "F1", "#b91c1c", "-", 2.0),
        ("youden_j", "Youden J", "#1f4e79", "-", 2.2),
    ]
    for key, label, color, linestyle, lw in curve_spec:
        vals = df[key].to_numpy(dtype=float)
        if np.all(~np.isfinite(vals)):
            continue
        ax.plot(
            df["threshold"].to_numpy(dtype=float),
            vals,
            color=color,
            linestyle=linestyle,
            lw=lw,
            alpha=0.95,
            label=label,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.02)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.05), fontsize=8.8, ncol=2)
    fig.tight_layout()
    save_figure_with_pdf(fig, out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Threshold metric curve saved to {out_path}")


def plot_decision_curves(
    y_true: np.ndarray,
    prob_dict: Dict[str, np.ndarray],
    thresholds: Sequence[float],
    out_path: Path,
) -> None:
    """Plot clinical decision curves for multiple models on one figure."""
    if not prob_dict:
        return
    setup_publication_style()
    curves = {
        name: compute_decision_curve(y_true, proba, thresholds)
        for name, proba in prob_dict.items()
    }
    has_data = any(not df.empty for df in curves.values())
    if not has_data:
        print("Decision curve: no data to plot.")
        return

    # Set a sensible lower bound to avoid treat_all collapsing the y-axis.
    min_model_nb: float | None = None
    for df in curves.values():
        if df.empty:
            continue
        cur_min = df["net_benefit"].min()
        min_model_nb = cur_min if min_model_nb is None else min(min_model_nb, cur_min)
    if min_model_nb is None:
        min_model_nb = -0.05
    clip_floor = min_model_nb - 0.05

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD, facecolor="white")
    style_axis(ax)
    max_nb_values: List[float] = []
    for idx, (name, df) in enumerate(curves.items()):
        if df.empty:
            continue
        max_nb_values.append(float(df["net_benefit"].max()))
        ax.plot(
            df["threshold"],
            df["net_benefit"],
            label=format_model_name(name),
            color=get_model_color(name, idx),
            lw=MODEL_LINE_WIDTH,
            solid_capstyle="round",
            alpha=0.95,
        )
    first_df = next((df for df in curves.values() if not df.empty), None)
    if first_df is not None:
        treat_all_full = first_df["treat_all"].to_numpy()
        treat_all_plot = np.where(treat_all_full > clip_floor, treat_all_full, np.nan)
        finite_treat_all = treat_all_plot[np.isfinite(treat_all_plot)]
        if finite_treat_all.size > 0:
            max_nb_values.append(float(np.nanmax(finite_treat_all)))
        ax.plot(
            first_df["threshold"],
            treat_all_plot,
            label="Treat all",
            linestyle="--",
            color="#646a73",
            lw=1.5,
            alpha=0.9,
        )
        ax.plot(
            first_df["threshold"],
            first_df["treat_none"],
            label="Treat none",
            linestyle=":",
            color="#8b9098",
            lw=1.5,
            alpha=0.95,
        )
        max_nb_values.append(0.0)
    ax.set_xlim(0.0, 1.0)
    y_top = max(0.06, (max(max_nb_values) + 0.04) if max_nb_values else 0.2)
    ax.set_ylim(bottom=clip_floor, top=y_top)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    save_figure_with_pdf(fig, out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Decision curve saved to {out_path}")


def oof_proba_for_model(
    base_model,
    X: pd.DataFrame,
    y: pd.Series,
    view_cols: Sequence[str],
    categorical_cols: Sequence[str],
    kf: StratifiedKFold,
    scale_numeric: bool = False,
    return_fold_curves: bool = False,
) -> Tuple[
    np.ndarray,
    List[dict],
    List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]],
]:
    """Generate out-of-fold probabilities and fold-level metrics."""
    proba_oof = np.zeros(len(X))
    fold_metrics: List[dict] = []
    fold_curves: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]] = []
    for train_idx, test_idx in kf.split(X, y):
        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_train, X_test = prepare_view_split(
            X_train_raw, X_test_raw, view_cols, categorical_cols, scale_numeric
        )
        X_train_bal, y_train_bal = resample_training_data(X_train, y_train, categorical_cols)

        try:
            model = clone(base_model)
        except Exception:
            model = copy.deepcopy(base_model)

        model.fit(X_train_bal, y_train_bal)
        proba = model.predict_proba(X_test)[:, 1]
        proba_oof[test_idx] = proba
        thr_j, thr_f1, _ = _select_thresholds(y_test.values, proba)
        thr_used = thr_j
        pred = (proba >= thr_used).astype(int)
        auc = roc_auc_score(y_test, proba)
        auprc = average_precision_score(y_test, proba)
        acc = accuracy_score(y_test, pred)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        tn_fp = ((y_test == 0) & (pred == 0)).sum(), ((y_test == 0) & (pred == 1)).sum()
        tn, fp = tn_fp
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        if return_fold_curves:
            fpr, tpr, _ = roc_curve(y_test, proba)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, proba)
            fold_curves.append(
                (fpr, tpr, float(auc), recall_curve[::-1], precision_curve[::-1], float(auprc))
            )
        fold_metrics.append(
            {
                "auc": auc,
                "auprc": auprc,
                "accuracy": acc,
                "recall": rec,
                "specificity": spec,
                "f1": f1,
                "threshold_j": thr_j,
                "threshold_f1": thr_f1,
            }
        )
    return proba_oof, fold_metrics, fold_curves


def build_models():
    """Build base models with their view columns and scaling options."""
    blood_et = ExtraTreesClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        min_samples_split=4,
        max_features="sqrt",
        bootstrap=True,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    blood_xgb = xgb.XGBClassifier(
        learning_rate=0.02,
        n_estimators=1200,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=2,
        reg_lambda=50,
        gamma=0.0,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
        scale_pos_weight=1.0,
    )

    allergen_cat = CatBoostClassifier(
        iterations=600,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=20,
        loss_function="Logloss",
        eval_metric="AUC",
        # Reduce positive-class dominance noise in this dataset.
        class_weights=[2.0, 0.8],
        subsample=0.8,
        random_seed=RANDOM_STATE,
        verbose=0,
    )

    history_lr = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    blood_view = apply_demo_policy(BLOOD_VIEW, "blood")
    allergen_view = apply_demo_policy(ALLERGEN_VIEW, "allergen")
    history_view = apply_demo_policy(HISTORY_VIEW, "history")

    models = {
        "blood_extratrees": (blood_et, blood_view, False),
        "blood_xgb": (blood_xgb, blood_view, False),
        "allergen_cat": (allergen_cat, allergen_view, False),
        "history_logreg": (history_lr, history_view, True),
    }

    if DEMO_POLICY == "separate":
        demo_lr = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            max_iter=500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        models["demo_logreg"] = (demo_lr, DEMO_COLS, True)

    tuned = load_best_params()
    if tuned:
        for name, (model, _, _) in models.items():
            apply_tuned_params(model, name, tuned)

    return models


def stack_with_nnls(base_matrix: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, LinearRegression]:
    """Fit NNLS-like linear stacker in logit space and return probabilities."""
    # positive=True approximates NNLS in logit space before sigmoid.
    meta = LinearRegression(positive=True, fit_intercept=True)
    meta.fit(base_matrix, y)
    score = base_matrix @ meta.coef_ + meta.intercept_
    stacked_proba = _sigmoid(score)
    return stacked_proba, meta


def _evaluate_with_threshold(y_true: np.ndarray, proba: np.ndarray) -> dict:
    thr_j, thr_f1, _ = _select_thresholds(y_true, proba)
    thr_used = thr_j
    pred = (proba >= thr_used).astype(int)
    auc = roc_auc_score(y_true, proba)
    auprc = average_precision_score(y_true, proba)
    acc = accuracy_score(y_true, pred)
    rec = recall_score(y_true, pred)
    f1 = f1_score(y_true, pred)
    tn = ((y_true == 0) & (pred == 0)).sum()
    fp = ((y_true == 0) & (pred == 1)).sum()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return {
        "auc": auc,
        "auprc": auprc,
        "accuracy": acc,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "thr_used": thr_used,
        "thr_j": thr_j,
        "thr_f1": thr_f1,
    }


def bootstrap_metrics_and_roc(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_boot: int = BOOTSTRAP_ROUNDS,
    seed: int = RANDOM_STATE,
    grid: np.ndarray | None = None,
) -> Tuple[dict, dict, dict, dict]:
    """Bootstrap metrics and ROC summaries in one pass."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_vals = {
        k: []
        for k in ["auc", "auprc", "accuracy", "recall", "specificity", "f1", "thr_used", "thr_j", "thr_f1"]
    }
    tpr_store = []
    auc_store = []
    idx_matrix = rng.integers(0, n, size=(n_boot, n))
    for idx in idx_matrix:
        y_s = y_true[idx]
        p_s = proba[idx]
        if np.unique(y_s).shape[0] < 2:
            continue
        m = _evaluate_with_threshold(y_s, p_s)
        for k, v in m.items():
            boot_vals[k].append(v)
        fpr, tpr, _ = roc_curve(y_s, p_s)
        interp_tpr = np.interp(grid, fpr, tpr, left=0.0, right=1.0)
        tpr_store.append(interp_tpr)
        auc_store.append(roc_auc_score(y_s, p_s))

    point = _evaluate_with_threshold(y_true, proba)
    full_fpr, full_tpr, _ = roc_curve(y_true, proba)
    full_precision, full_recall, _ = precision_recall_curve(y_true, proba)
    ci = {}
    for k, v in boot_vals.items():
        if v:
            lo, hi = np.percentile(v, [2.5, 97.5])
        else:
            lo, hi = np.nan, np.nan
        ci[k] = (lo, hi)

    if not tpr_store:
        mean_tpr = np.full_like(grid, np.nan, dtype=float)
        lo_tpr = hi_tpr = mean_tpr
        auc_mean = float("nan")
        auc_ci = (float("nan"), float("nan"))
    else:
        tpr_arr = np.vstack(tpr_store)
        mean_tpr = tpr_arr.mean(axis=0)
        lo_tpr, hi_tpr = np.percentile(tpr_arr, [2.5, 97.5], axis=0)
        auc_mean = float(np.mean(auc_store))
        auc_ci = tuple(np.percentile(auc_store, [2.5, 97.5]).astype(float))

    roc = {
        "grid": grid,
        "mean_tpr": mean_tpr,
        "lo_tpr": lo_tpr,
        "hi_tpr": hi_tpr,
        "auc_mean": auc_mean,
        "auc_ci": auc_ci,
        "full_fpr": full_fpr,
        "full_tpr": full_tpr,
        "full_recall": full_recall[::-1],
        "full_precision": full_precision[::-1],
        "prevalence": float(np.mean(y_true)),
    }
    raw = {
        "grid": np.asarray(grid, dtype=float),
        "boot_vals": {k: np.asarray(v, dtype=float) for k, v in boot_vals.items()},
        "tpr_arr": np.vstack(tpr_store).astype(float) if tpr_store else np.empty((0, len(grid)), dtype=float),
        "auc_store": np.asarray(auc_store, dtype=float),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }
    return point, ci, roc, raw


def summarize_bootstrap_from_raw(
    y_true: np.ndarray,
    proba: np.ndarray,
    raw: dict,
) -> Tuple[dict, dict, dict]:
    """Rebuild point/CI/curve summaries from cached bootstrap raw outputs."""
    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(proba, dtype=float)
    point = _evaluate_with_threshold(y_arr, p_arr)
    full_fpr, full_tpr, _ = roc_curve(y_arr, p_arr)
    full_precision, full_recall, _ = precision_recall_curve(y_arr, p_arr)

    grid = np.asarray(raw.get("grid", np.linspace(0.0, 1.0, 101)), dtype=float)
    if grid.ndim != 1:
        grid = np.linspace(0.0, 1.0, 101)
    tpr_arr = np.asarray(raw.get("tpr_arr", np.empty((0, len(grid)))), dtype=float)
    auc_store = np.asarray(raw.get("auc_store", np.array([])), dtype=float)

    boot_vals_raw = raw.get("boot_vals", {})
    boot_vals = boot_vals_raw if isinstance(boot_vals_raw, dict) else {}
    ci = {}
    for key in ["auc", "auprc", "accuracy", "recall", "specificity", "f1", "thr_used", "thr_j", "thr_f1"]:
        vals = np.asarray(boot_vals.get(key, np.array([])), dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            ci[key] = (np.nan, np.nan)
        else:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            ci[key] = (float(lo), float(hi))

    if tpr_arr.size == 0 or tpr_arr.ndim != 2:
        mean_tpr = np.full_like(grid, np.nan, dtype=float)
        lo_tpr = mean_tpr.copy()
        hi_tpr = mean_tpr.copy()
    else:
        mean_tpr = np.nanmean(tpr_arr, axis=0)
        lo_tpr, hi_tpr = np.nanpercentile(tpr_arr, [2.5, 97.5], axis=0)

    auc_store = auc_store[np.isfinite(auc_store)]
    if auc_store.size == 0:
        auc_mean = float("nan")
        auc_ci = (float("nan"), float("nan"))
    else:
        auc_mean = float(np.mean(auc_store))
        auc_ci = tuple(np.percentile(auc_store, [2.5, 97.5]).astype(float))

    roc = {
        "grid": grid,
        "mean_tpr": mean_tpr,
        "lo_tpr": lo_tpr,
        "hi_tpr": hi_tpr,
        "auc_mean": auc_mean,
        "auc_ci": auc_ci,
        "full_fpr": full_fpr,
        "full_tpr": full_tpr,
        "full_recall": full_recall[::-1],
        "full_precision": full_precision[::-1],
        "prevalence": float(np.mean(y_arr)),
    }
    return point, ci, roc


def find_legacy_bootstrap_summary_entry(
    summary_cache: Dict[str, object],
    y_true: np.ndarray,
    proba: np.ndarray,
    used_keys: set[str],
    atol: float = 1e-8,
) -> Optional[Tuple[str, Tuple[dict, dict, dict]]]:
    """Find a legacy bootstrap-summary entry by matching point-metric signature."""
    target = _evaluate_with_threshold(np.asarray(y_true).astype(int), np.asarray(proba, dtype=float))
    metric_keys = ["auc", "auprc", "accuracy", "recall", "specificity", "f1", "thr_used", "thr_j", "thr_f1"]
    target_vec = []
    for key in metric_keys:
        val = target.get(key)
        if val is None:
            return None
        target_vec.append(float(val))
    target_arr = np.asarray(target_vec, dtype=float)

    best_key: Optional[str] = None
    best_entry: Optional[Tuple[dict, dict, dict]] = None
    best_score = float("inf")

    for cache_key, entry in summary_cache.items():
        if cache_key in used_keys:
            continue
        if not isinstance(entry, (tuple, list)) or len(entry) != 3:
            continue
        point_raw, ci_raw, roc_raw = entry
        if not isinstance(point_raw, dict) or not isinstance(ci_raw, dict) or not isinstance(roc_raw, dict):
            continue

        cand_vec = []
        ok = True
        for key in metric_keys:
            val = point_raw.get(key)
            if val is None:
                ok = False
                break
            try:
                cand_vec.append(float(val))
            except Exception:
                ok = False
                break
        if not ok:
            continue

        cand_arr = np.asarray(cand_vec, dtype=float)
        diff = np.abs(cand_arr - target_arr)
        if not np.all(np.isfinite(diff)):
            continue
        if float(np.max(diff)) > float(atol):
            continue
        score = float(np.sum(diff))
        if score < best_score:
            best_score = score
            best_key = cache_key
            best_entry = (dict(point_raw), dict(ci_raw), dict(roc_raw))

    if best_key is None or best_entry is None:
        return None
    return best_key, best_entry


def _select_thresholds(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, float, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return Youden-J and F1-optimal thresholds with ROC curve points."""
    fpr, tpr, thr = roc_curve(y_true, proba)
    j_scores = tpr - fpr
    idx_j = int(np.argmax(j_scores))
    thr_j = thr[idx_j] if len(thr) > 0 else 0.5

    best_f1 = -1.0
    best_thr_f1 = 0.5
    for t in np.unique(thr):
        y_pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr_f1 = t
    return float(thr_j), float(best_thr_f1), (fpr, tpr, thr)


def nnls_oof_predictions(base_prob_matrix: np.ndarray, y_true: np.ndarray) -> dict:
    """Pool one NNLS validation prediction per record in original input order.

    Base-model OOF features are held fixed; only the NNLS layer is cross-fitted.
    Fold-specific classification summaries retain their validation-fold Youden
    thresholds and are auxiliary, not fixed-threshold validation estimates.
    """
    base_logit_matrix = _logit(np.asarray(base_prob_matrix, dtype=float))
    y = pd.Series(np.asarray(y_true, dtype=int))
    if base_logit_matrix.ndim != 2 or base_logit_matrix.shape[0] != len(y):
        raise ValueError("NNLS feature/label shape mismatch")
    if not np.isfinite(base_logit_matrix).all():
        raise ValueError("Non-finite NNLS input")
    nnls_oof = np.full(len(y), np.nan, dtype=float)
    fold_ids = np.zeros(len(y), dtype=int)
    counts = np.zeros(len(y), dtype=int)
    meta_kf = StratifiedKFold(n_splits=K_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    meta_metric_rows = []
    meta_roc_curves = []
    meta_pr_curves = []
    for fold_idx, (tr, te) in enumerate(meta_kf.split(base_logit_matrix, y), 1):
        meta = LinearRegression(positive=True, fit_intercept=True)
        meta.fit(base_logit_matrix[tr], y.values[tr])
        meta_score = base_logit_matrix[te] @ meta.coef_ + meta.intercept_
        meta_pred = _sigmoid(meta_score)
        if np.intersect1d(tr, te).size or np.any(counts[te]):
            raise RuntimeError("NNLS folds overlap or repeat validation records")
        nnls_oof[te] = meta_pred
        fold_ids[te] = fold_idx
        counts[te] += 1

        thr_j, thr_f1, (fpr, tpr, _) = _select_thresholds(y.values[te], meta_pred)
        thr_used = thr_j  # Use Youden J threshold
        meta_class = (meta_pred >= thr_used).astype(int)
        auc = roc_auc_score(y.values[te], meta_pred)
        auprc = average_precision_score(y.values[te], meta_pred)
        precision_curve, recall_curve, _ = precision_recall_curve(y.values[te], meta_pred)
        acc = accuracy_score(y.values[te], meta_class)
        rec = recall_score(y.values[te], meta_class)
        f1 = f1_score(y.values[te], meta_class)
        tn = ((y.values[te] == 0) & (meta_class == 0)).sum()
        fp = ((y.values[te] == 0) & (meta_class == 1)).sum()
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        meta_roc_curves.append((fold_idx, fpr, tpr, auc))
        meta_pr_curves.append((fold_idx, recall_curve[::-1], precision_curve[::-1], auprc))
        meta_metric_rows.append(
            {
                "fold": fold_idx,
                "auc": auc,
                "auprc": auprc,
                "accuracy": acc,
                "recall": rec,
                "specificity": spec,
                "f1": f1,
                "threshold_j": thr_j,
                "threshold_f1": thr_f1,
            }
        )
        print(
            f"[meta fold {fold_idx}] AUC={auc:.3f}, AUPRC={auprc:.3f}, ACC={acc:.3f}, "
            f"RECALL={rec:.3f}, SPEC={spec:.3f}, F1={f1:.3f}, "
            f"THR_J={thr_j:.3f}, THR_F1={thr_f1:.3f}"
        )

    if not np.all(counts == 1) or not np.isfinite(nnls_oof).all():
        raise RuntimeError("Incomplete or duplicate NNLS OOF predictions")
    return {
        "nnls_oof": nnls_oof, "meta_fold_ids": fold_ids,
        "meta_oof_counts": counts, "meta_metric_rows": meta_metric_rows,
        "meta_roc_curves": meta_roc_curves, "meta_pr_curves": meta_pr_curves,
        "evaluation_version": NNLS_EVALUATION_VERSION,
        "y_hash": hash_ndarray(y.to_numpy()),
    }


def main():
    """Train or load cached predictions, evaluate, and export plots/metrics."""
    # Save fold metrics to Excel
    try:
        os.chdir(BASE_DIR)
    except OSError:
        pass
    setup_publication_style()
    X, y = load_dataset()
    cat_cols = DEFAULT_CATEGORICAL
    kf = StratifiedKFold(n_splits=K_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    models = build_models()
    prediction_cache = load_layer_cache(PREDICTION_CACHE_PATH, "prediction")
    prediction_cache_key = make_prediction_cache_key(X, y, models)
    base_probs: List[np.ndarray] = []
    names: List[str] = []
    base_metric_rows: List[dict] = []
    base_fold_rocs: Dict[
        str,
        List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]],
    ] = {}
    meta_metric_rows: List[dict] = []
    meta_roc_curves: List[Tuple[int, np.ndarray, np.ndarray, float]] = []
    meta_pr_curves: List[Tuple[int, np.ndarray, np.ndarray, float]] = []
    bootstrap_raw_cache: Dict[str, object] = {}
    bootstrap_summary_cache: Dict[str, object] = {}
    base_prob_matrix = np.empty((len(y), 0), dtype=float)
    nnls_oof = np.full(len(y), np.nan, dtype=float)
    stacked_final_fit = np.full(len(y), np.nan, dtype=float)
    meta_fold_ids = np.zeros(len(y), dtype=int)
    meta_oof_counts = np.zeros(len(y), dtype=int)
    meta_coef = np.empty((0,), dtype=float)
    meta_intercept = float("nan")
    if ENABLE_BOOTSTRAP:
        bootstrap_raw_cache = load_layer_cache(BOOTSTRAP_RAW_CACHE_PATH, "bootstrap_raw")
        bootstrap_summary_cache = load_layer_cache(BOOTSTRAP_SUMMARY_CACHE_PATH, "bootstrap_summary")
    cached_prediction = prediction_cache.get(prediction_cache_key)
    use_prediction_cache = False
    if isinstance(cached_prediction, dict):
        try:
            names = list(cached_prediction["names"])
            base_prob_matrix = np.asarray(cached_prediction["base_prob_matrix"], dtype=float)
            base_metric_rows = list(cached_prediction["base_metric_rows"])
            base_fold_rocs = cached_prediction["base_fold_rocs"]
            meta_metric_rows = list(cached_prediction["meta_metric_rows"])
            meta_roc_curves = cached_prediction["meta_roc_curves"]
            meta_pr_curves = cached_prediction["meta_pr_curves"]
            if cached_prediction["evaluation_version"] != NNLS_EVALUATION_VERSION:
                raise ValueError("Obsolete NNLS evaluation definition")
            if cached_prediction["y_hash"] != hash_ndarray(y.to_numpy()):
                raise ValueError("NNLS cached label/order mismatch")
            nnls_oof = np.asarray(cached_prediction["nnls_oof"], dtype=float)
            stacked_final_fit = np.asarray(cached_prediction["stacked_final_fit"], dtype=float)
            meta_fold_ids = np.asarray(cached_prediction["meta_fold_ids"], dtype=int)
            meta_oof_counts = np.asarray(cached_prediction["meta_oof_counts"], dtype=int)
            if not np.all(meta_oof_counts == 1) or not np.isfinite(nnls_oof).all():
                raise ValueError("NNLS OOF cache has missing or duplicate predictions")
            expected_folds = np.zeros(len(y), dtype=int)
            for fold, (_, te) in enumerate(kf.split(base_prob_matrix, y), 1):
                expected_folds[te] = fold
            if not np.array_equal(meta_fold_ids, expected_folds):
                raise ValueError("NNLS OOF cached fold/order mismatch")
            meta_coef = np.asarray(cached_prediction["meta_coef"], dtype=float)
            meta_intercept = float(cached_prediction["meta_intercept"])
            if base_prob_matrix.shape[0] != len(y) or nnls_oof.shape != (len(y),) or stacked_final_fit.shape != (len(y),):
                raise ValueError("prediction length mismatch")
            if base_prob_matrix.shape[1] != len(names):
                raise ValueError("base model count mismatch")
            if meta_coef.shape[0] != len(names):
                raise ValueError("meta coef count mismatch")
            use_prediction_cache = True
            print("Using cached prediction layer (base/meta predictions).")
        except Exception as exc:
            print(f"Prediction cache invalid; recomputing. reason={exc}")
            names = []
            base_metric_rows = []
            base_fold_rocs = {}
            meta_metric_rows = []
            meta_roc_curves = []
            meta_pr_curves = []
    if use_prediction_cache:
        print("Skip base-model training and meta CV (prediction cache hit).")
    else:
        for name, (model, view, scale_flag) in models.items():
            cached_model_entry = load_prediction_model_entry(prediction_cache_key, name)
            loaded_from_entry = False
            if isinstance(cached_model_entry, dict):
                try:
                    proba = np.asarray(cached_model_entry["proba"], dtype=float)
                    metrics = list(cached_model_entry["metrics"])
                    fold_curves = list(cached_model_entry["fold_curves"])
                    if proba.shape[0] != len(y):
                        raise ValueError("prediction length mismatch")
                    loaded_from_entry = True
                    print(f"Using cached base-model entry for {name}")
                except Exception as exc:
                    print(f"Base-model entry invalid for {name}; retraining. reason={exc}")
            if not loaded_from_entry:
                print(f"Training {name} on view size={len(view)}")
                proba, metrics, fold_curves = oof_proba_for_model(
                    model,
                    X,
                    y,
                    view_cols=view,
                    categorical_cols=cat_cols,
                    kf=kf,
                    scale_numeric=scale_flag,
                    return_fold_curves=True,
                )
                save_prediction_model_entry(
                    prediction_cache_key=prediction_cache_key,
                    model_name=name,
                    entry={
                        "proba": np.asarray(proba, dtype=float),
                        "metrics": metrics,
                        "fold_curves": fold_curves,
                        "n_samples": int(len(y)),
                    },
                )
            base_probs.append(proba)
            names.append(name)
            base_fold_rocs[name] = fold_curves
            for fold_idx, m in enumerate(metrics, 1):
                row = {"model": name, "fold": fold_idx}
                row.update(m)
                base_metric_rows.append(row)

    if not use_prediction_cache:
        if not base_probs:
            raise RuntimeError("No base-model predictions generated.")
        base_prob_matrix = np.column_stack(base_probs)
        base_logit_matrix = _logit(base_prob_matrix)
        meta_oof_result = nnls_oof_predictions(base_prob_matrix, y.values)
        nnls_oof = meta_oof_result["nnls_oof"]
        meta_fold_ids = meta_oof_result["meta_fold_ids"]
        meta_oof_counts = meta_oof_result["meta_oof_counts"]
        meta_metric_rows = meta_oof_result["meta_metric_rows"]
        meta_roc_curves = meta_oof_result["meta_roc_curves"]
        meta_pr_curves = meta_oof_result["meta_pr_curves"]

        # Final all-development coefficients are retained for external prediction.
        stacked_final_fit, meta = stack_with_nnls(base_logit_matrix, y.values)
        meta_coef = np.asarray(meta.coef_, dtype=float)
        meta_intercept = float(meta.intercept_)
        prediction_cache[prediction_cache_key] = {
            "names": names,
            "base_prob_matrix": base_prob_matrix,
            "base_metric_rows": base_metric_rows,
            "base_fold_rocs": base_fold_rocs,
            "meta_metric_rows": meta_metric_rows,
            "meta_roc_curves": meta_roc_curves,
            "meta_pr_curves": meta_pr_curves,
            "nnls_oof": nnls_oof,
            "stacked_final_fit": stacked_final_fit,
            "meta_fold_ids": meta_fold_ids,
            "meta_oof_counts": meta_oof_counts,
            "evaluation_version": NNLS_EVALUATION_VERSION,
            "y_hash": hash_ndarray(y.to_numpy()),
            "meta_coef": meta_coef,
            "meta_intercept": meta_intercept,
        }
        save_layer_cache(PREDICTION_CACHE_PATH, prediction_cache, "prediction")

    base_metrics_df = pd.DataFrame(base_metric_rows)
    ci_rows = []
    roc_results = {}
    raw_model_probs = {name: base_prob_matrix[:, i] for i, name in enumerate(names)}
    raw_model_probs["stack_nnls"] = nnls_oof
    np.savez_compressed(BASE_DIR / "development_nnls_predictions.npz",
        y=y.to_numpy(), record_order=np.arange(len(y)),
        base_prob_matrix=base_prob_matrix, names=np.asarray(names, dtype=str),
        nnls_oof=nnls_oof, stacked_final_fit=stacked_final_fit,
        fold_ids=meta_fold_ids, counts=meta_oof_counts,
        coef=meta_coef, intercept=meta_intercept,
        evaluation_version=NNLS_EVALUATION_VERSION)
    model_probs = raw_model_probs
    calibration_tag = "raw"
    print("Using raw probabilities only.")

    def fmt_ci(pt: float, ci: Tuple[float, float]) -> str:
        lo, hi = ci
        if np.isnan(lo) or np.isnan(hi):
            return f"{pt:.3f} (nan)"
        return f"{pt:.3f} ({lo:.3f}-{hi:.3f})"

    def report(y_true, y_pred_prob, label):
        grid_boot = np.linspace(0.0, 1.0, 101)
        cache_key = make_bootstrap_cache_key(
            label=label,
            calibration_tag=calibration_tag,
            y_true=np.asarray(y_true),
            proba=np.asarray(y_pred_prob, dtype=float),
            n_boot=BOOTSTRAP_ROUNDS,
            seed=RANDOM_STATE,
            grid=grid_boot,
        )
        summary_entry = bootstrap_summary_cache.get(cache_key)
        if ENABLE_BOOTSTRAP and summary_entry is None:
            summary_entry = load_entry_cache(
                BOOTSTRAP_SUMMARY_ENTRY_CACHE_DIR, cache_key, "bootstrap_summary"
            )
            if summary_entry is not None:
                bootstrap_summary_cache[cache_key] = summary_entry
        if ENABLE_BOOTSTRAP and summary_entry is not None:
            print(f"Using cached bootstrap summary for {label}")
            point, ci, roc = summary_entry  # type: ignore[assignment]
        else:
            raw_entry = bootstrap_raw_cache.get(cache_key)
            if ENABLE_BOOTSTRAP and raw_entry is None:
                raw_entry = load_entry_cache(
                    BOOTSTRAP_RAW_ENTRY_CACHE_DIR, cache_key, "bootstrap_raw"
                )
                if raw_entry is not None:
                    bootstrap_raw_cache[cache_key] = raw_entry
            if ENABLE_BOOTSTRAP and raw_entry is not None:
                print(f"Using cached bootstrap raw outputs for {label}")
                point, ci, roc = summarize_bootstrap_from_raw(y_true=y_true, proba=y_pred_prob, raw=raw_entry)  # type: ignore[arg-type]
                bootstrap_summary_cache[cache_key] = (point, ci, roc)
                save_entry_cache(
                    BOOTSTRAP_SUMMARY_ENTRY_CACHE_DIR,
                    cache_key,
                    (point, ci, roc),
                    "bootstrap_summary",
                )
                save_layer_cache(BOOTSTRAP_SUMMARY_CACHE_PATH, bootstrap_summary_cache, "bootstrap_summary")
            elif ENABLE_BOOTSTRAP:
                # Bootstrap reuse requires exact label/prediction hashes.
                print(f"Running bootstrap for {label}")
                point, ci, roc, raw = bootstrap_metrics_and_roc(
                    y_true,
                    y_pred_prob,
                    n_boot=BOOTSTRAP_ROUNDS,
                    seed=RANDOM_STATE,
                    grid=grid_boot,
                )
                bootstrap_raw_cache[cache_key] = raw
                bootstrap_summary_cache[cache_key] = (point, ci, roc)
                save_entry_cache(BOOTSTRAP_RAW_ENTRY_CACHE_DIR, cache_key, raw, "bootstrap_raw")
                save_entry_cache(
                    BOOTSTRAP_SUMMARY_ENTRY_CACHE_DIR,
                    cache_key,
                    (point, ci, roc),
                    "bootstrap_summary",
                )
                save_layer_cache(BOOTSTRAP_RAW_CACHE_PATH, bootstrap_raw_cache, "bootstrap_raw")
                save_layer_cache(BOOTSTRAP_SUMMARY_CACHE_PATH, bootstrap_summary_cache, "bootstrap_summary")
            else:
                point = _evaluate_with_threshold(y_true, y_pred_prob)
                ci = {
                    k: (np.nan, np.nan)
                    for k in ["auc", "auprc", "accuracy", "recall", "f1", "specificity"]
                }
                roc = None
        point = dict(point) if isinstance(point, dict) else {}
        ci = dict(ci) if isinstance(ci, dict) else {}
        point_eval = _evaluate_with_threshold(y_true, y_pred_prob)
        for k, v in point_eval.items():
            keep_value = False
            if k in point:
                try:
                    keep_value = bool(np.isfinite(float(point[k])))
                except Exception:
                    keep_value = False
            if not keep_value:
                point[k] = v
        for key in ["auc", "auprc", "accuracy", "recall", "f1", "specificity"]:
            val = ci.get(key)
            if not has_finite_ci(val):
                ci[key] = (np.nan, np.nan)
            else:
                ci[key] = (float(val[0]), float(val[1]))  # type: ignore[index]
        thr_used = point["thr_j"]
        print(
            f"{label}: AUC={point['auc']:.3f}, AUPRC={point['auprc']:.3f}, "
            f"ACC={point['accuracy']:.3f}, RECALL={point['recall']:.3f}, "
            f"F1={point['f1']:.3f}, SPEC={point['specificity']:.3f}, "
            f"THR_USED={thr_used:.3f}, THR_J={point['thr_j']:.3f}, THR_F1={point['thr_f1']:.3f}"
        )
        if roc is not None:
            roc = dict(roc) if isinstance(roc, dict) else {}
            if "full_fpr" not in roc or "full_tpr" not in roc:
                full_fpr, full_tpr, _ = roc_curve(y_true, y_pred_prob)
                roc["full_fpr"] = full_fpr
                roc["full_tpr"] = full_tpr
            if "full_recall" not in roc or "full_precision" not in roc:
                full_precision, full_recall, _ = precision_recall_curve(y_true, y_pred_prob)
                roc["full_recall"] = full_recall[::-1]
                roc["full_precision"] = full_precision[::-1]
            if "prevalence" not in roc:
                roc["prevalence"] = float(np.mean(y_true))
            if not has_finite_ci(roc.get("auc_ci")):
                roc["auc_ci"] = ci["auc"]
            roc["auc_point"] = point["auc"]
            roc["auprc_point"] = point["auprc"]
            roc["auprc_ci"] = ci["auprc"]
            roc_results[label] = roc
        if ENABLE_BOOTSTRAP:
            ci_rows.append(
                {
                    "model": label,
                    "AUC": fmt_ci(point["auc"], ci["auc"]),
                    "AUPRC": fmt_ci(point["auprc"], ci["auprc"]),
                    "Accuracy": fmt_ci(point["accuracy"], ci["accuracy"]),
                    "Recall": fmt_ci(point["recall"], ci["recall"]),
                    "F1": fmt_ci(point["f1"], ci["f1"]),
                    "Specificity": fmt_ci(point["specificity"], ci["specificity"]),
                }
            )

    for name in names:
        report(y.values, model_probs[name], name)
    report(y.values, model_probs["stack_nnls"], "stack_nnls")
    print("NNLS weights:")
    for name, w in zip(names, meta_coef):
        print(f"  {name}: {w:.4f}")
    print(f"  (intercept): {meta_intercept:.4f}")

    # Save fold metrics to Excel
    meta_metrics_df = pd.DataFrame(meta_metric_rows)
    with pd.ExcelWriter(BASE_DIR / "stack_nnls_metrics.xlsx") as writer:
        base_metrics_df.to_excel(writer, sheet_name="base_folds", index=False)
        meta_metrics_df.to_excel(writer, sheet_name="meta_folds", index=False)
        if ENABLE_BOOTSTRAP and ci_rows:
            ci_df = pd.DataFrame(ci_rows)
            ci_df.to_excel(writer, sheet_name="metrics_ci", index=False)
    print("Fold metrics saved to stack_nnls_metrics.xlsx")

    # Save meta fold ROC curves to figures/meta_roc_cv.png
    fig_path = BASE_DIR / "figures/meta_roc_cv.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIG_SIZE_COMPACT, facecolor="white")
    style_axis(ax)
    for fold_idx, fpr, tpr, auc in meta_roc_curves:
        fold_color = get_fold_color(fold_idx)
        ax.plot(
            fpr,
            tpr,
            color=fold_color,
            linestyle="-",
            lw=1.8,
            alpha=0.95,
            label=f"Fold {fold_idx} (AUC={auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", lw=REF_LINE_WIDTH, color="#6b7280", alpha=0.9)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8.5, title="CV fold", title_fontsize=9)
    fig.tight_layout()
    save_figure_with_pdf(fig, fig_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Meta fold ROC curves saved to {fig_path}")

    # Save meta fold PR curves to figures/meta_pr_cv.png
    pr_meta_fig = BASE_DIR / "figures/meta_pr_cv.png"
    pr_meta_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIG_SIZE_COMPACT, facecolor="white")
    style_axis(ax)
    pr_min = float("inf")
    pr_max = float("-inf")
    for fold_idx, fold_recall, fold_precision, fold_auprc in meta_pr_curves:
        fold_color = get_fold_color(fold_idx)
        smooth_recall, smooth_precision = smooth_pr_curve_for_plot(fold_recall, fold_precision)
        pr_min = min(pr_min, float(np.nanmin(smooth_precision)))
        pr_max = max(pr_max, float(np.nanmax(smooth_precision)))
        ax.plot(
            smooth_recall,
            smooth_precision,
            color=fold_color,
            linestyle="-",
            lw=1.8,
            alpha=0.95,
            label=f"Fold {fold_idx} (AUPRC={fold_auprc:.3f})",
        )
    prevalence = float(y.mean())
    ax.axhline(prevalence, linestyle="--", lw=REF_LINE_WIDTH, color="#6b7280", alpha=0.9, label=f"Prevalence {prevalence:.3f}")
    ax.set_xlim(0.0, 1.0)
    set_pr_axis_limits(ax, pr_min, pr_max, prevalence=prevalence)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right", fontsize=8.5, title="CV fold", title_fontsize=9)
    fig.tight_layout()
    save_figure_with_pdf(fig, pr_meta_fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Meta fold PR curves saved to {pr_meta_fig}")

    # Save base-model fold ROC curves (one figure per base model)
    for mname, curves in base_fold_rocs.items():
        if not curves:
            continue
        bm_fig = BASE_DIR / f"figures/{mname}_roc_folds.png"
        bm_fig.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=FIG_SIZE_COMPACT, facecolor="white")
        style_axis(ax)
        for idx, (fpr, tpr, fold_auc, fold_recall, fold_precision, fold_auprc) in enumerate(curves, 1):
            fold_color = get_fold_color(idx)
            ax.plot(
                fpr,
                tpr,
                color=fold_color,
                linestyle="-",
                lw=1.8,
                alpha=0.95,
                label=f"Fold {idx} (AUC={fold_auc:.3f})",
            )
        ax.plot([0, 1], [0, 1], linestyle="--", lw=REF_LINE_WIDTH, color="#6b7280", alpha=0.9)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=8.5, title="CV fold", title_fontsize=9)
        fig.tight_layout()
        save_figure_with_pdf(fig, bm_fig, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Base fold ROC saved to {bm_fig}")

        bm_pr_fig = BASE_DIR / f"figures/{mname}_pr_folds.png"
        bm_pr_fig.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=FIG_SIZE_COMPACT, facecolor="white")
        style_axis(ax)
        pr_min = float("inf")
        pr_max = float("-inf")
        for idx, (_, _, _, fold_recall, fold_precision, fold_auprc) in enumerate(curves, 1):
            fold_color = get_fold_color(idx)
            smooth_recall, smooth_precision = smooth_pr_curve_for_plot(fold_recall, fold_precision)
            pr_min = min(pr_min, float(np.nanmin(smooth_precision)))
            pr_max = max(pr_max, float(np.nanmax(smooth_precision)))
            ax.plot(
                smooth_recall,
                smooth_precision,
                color=fold_color,
                linestyle="-",
                lw=1.8,
                alpha=0.95,
                label=f"Fold {idx} (AUPRC={fold_auprc:.3f})",
            )
        prevalence = float(y.mean())
        ax.axhline(
            prevalence,
            linestyle="--",
            lw=REF_LINE_WIDTH,
            color="#6b7280",
            alpha=0.9,
            label=f"Prevalence {prevalence:.3f}",
        )
        ax.set_xlim(0.0, 1.0)
        set_pr_axis_limits(ax, pr_min, pr_max, prevalence=prevalence)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="upper right", fontsize=8.5, title="CV fold", title_fontsize=9)
        fig.tight_layout()
        save_figure_with_pdf(fig, bm_pr_fig, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Base fold PR saved to {bm_pr_fig}")

    # Save clinical decision curves for base + meta models
    decision_fig = BASE_DIR / "figures/decision_curve.png"
    thresholds = np.linspace(0.01, 0.99, 99)
    plot_decision_curves(y.values, model_probs, thresholds, decision_fig)
    threshold_fig = BASE_DIR / "figures/stack_nnls_threshold_metrics.png"
    plot_threshold_metric_curve(
        y_true=y.values,
        proba=model_probs["stack_nnls"],
        out_path=threshold_fig,
        model_name="stack_nnls",
        thresholds=np.linspace(0.01, 0.99, 99),
    )

    # Save overall ROC/PR for base + meta.
    if ENABLE_BOOTSTRAP:
        model_names_with_curve = [m for m in model_probs if roc_results.get(m) is not None]
        show_ci_info = bool(model_names_with_curve) and all(
            has_finite_ci(roc_results[m].get("auprc_ci")) and has_finite_ci(roc_results[m].get("auc_ci"))
            for m in model_names_with_curve
        )

        roc_fig = BASE_DIR / "figures/roc_ci.png"
        roc_fig.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=FIG_SIZE_COMPACT, facecolor="white")
        style_axis(ax)
        for idx, mname in enumerate(model_probs):
            roc = roc_results.get(mname)
            if roc is None:
                continue
            grid_local = roc.get("grid", np.linspace(0.0, 1.0, 101))
            mean_tpr, lo_tpr, hi_tpr = roc["mean_tpr"], roc["lo_tpr"], roc["hi_tpr"]
            auc_mean, auc_ci = roc["auc_mean"], roc["auc_ci"]
            auc_point = roc.get("auc_point", auc_mean)
            color = get_model_color(mname, idx)
            if show_ci_info:
                ax.plot(
                    grid_local,
                    mean_tpr,
                    color=color,
                    lw=MODEL_LINE_WIDTH,
                    solid_capstyle="round",
                    label=f"{format_model_name(mname)} (AUC {auc_point:.3f} [{auc_ci[0]:.3f}-{auc_ci[1]:.3f}])",
                )
                ax.fill_between(grid_local, lo_tpr, hi_tpr, color=color, alpha=0.14)
            else:
                full_fpr = roc.get("full_fpr", grid_local)
                full_tpr = roc.get("full_tpr", mean_tpr)
                ax.plot(
                    full_fpr,
                    full_tpr,
                    color=color,
                    lw=MODEL_LINE_WIDTH,
                    solid_capstyle="round",
                    label=f"{format_model_name(mname)} (AUC {auc_point:.3f})",
                )
        ax.plot([0, 1], [0, 1], linestyle="--", lw=REF_LINE_WIDTH, color="#6b7280", alpha=0.9)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=8.5)
        fig.tight_layout()
        save_figure_with_pdf(fig, roc_fig, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        if show_ci_info:
            print(f"ROC with CI saved to {roc_fig}")
        else:
            print(f"ROC saved to {roc_fig}")

        pr_fig = BASE_DIR / "figures/pr_ci.png"
        pr_fig.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=FIG_SIZE_COMPACT, facecolor="white")
        style_axis(ax)
        pr_min = float("inf")
        pr_max = float("-inf")
        for idx, mname in enumerate(model_probs):
            curve = roc_results.get(mname)
            if curve is None:
                continue
            full_recall = curve.get("full_recall")
            full_precision = curve.get("full_precision")
            if full_recall is None or full_precision is None:
                continue
            smooth_recall, smooth_precision = smooth_pr_curve_for_plot(full_recall, full_precision)
            auprc_point = curve.get("auprc_point", float("nan"))
            auprc_ci = curve.get("auprc_ci", (float("nan"), float("nan")))
            color = get_model_color(mname, idx)
            if show_ci_info:
                label = f"{format_model_name(mname)} (AUPRC {auprc_point:.3f} [{auprc_ci[0]:.3f}-{auprc_ci[1]:.3f}])"
            else:
                label = f"{format_model_name(mname)} (AUPRC {auprc_point:.3f})"
            pr_min = min(pr_min, float(np.nanmin(smooth_precision)))
            pr_max = max(pr_max, float(np.nanmax(smooth_precision)))
            ax.plot(
                smooth_recall,
                smooth_precision,
                color=color,
                lw=MODEL_LINE_WIDTH,
                solid_capstyle="round",
                label=label,
            )
        prevalence = float(y.mean())
        ax.axhline(
            prevalence,
            linestyle="--",
            lw=REF_LINE_WIDTH,
            color="#6b7280",
            alpha=0.9,
            label=f"Prevalence {prevalence:.3f}",
        )
        ax.set_xlim(0.0, 1.0)
        set_pr_axis_limits(ax, pr_min, pr_max, prevalence=prevalence)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.02, 0.10),
            fontsize=8.3,
            frameon=False,
            borderaxespad=0.2,
            labelspacing=0.35,
            handlelength=1.6,
        )
        fig.tight_layout()
        save_figure_with_pdf(fig, pr_fig, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"PR saved to {pr_fig}")


if __name__ == "__main__":
    main()

# %%

