"""Calibration analysis for the canonical multi-view Stack-NNLS model.

The script evaluates the *raw* Stack-NNLS probabilities without changing the
primary predictive model.  It reports calibration-in-the-large (intercept),
calibration slope, Brier score, expected calibration error (ECE), maximum
calibration error (MCE), decile calibration tables, bootstrap percentile CIs,
and a two-panel reliability diagram for the internal and external cohorts.

The script requires the current NNLS-layer pooled OOF prediction cache and
uses the external prediction workbook produced by the analysis pipeline.
Legacy full-development fitted scores are not accepted as OOF predictions.
Patient-level predictions are used locally but are not written to
the aggregate-results directory unless --write-predictions is explicitly set.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


SCRIPT_VERSION = "1.1.0-nnls-pooled-oof"
DEFAULT_BOOTSTRAP_ROUNDS = 2000
DEFAULT_SEED = 42
N_BINS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing stack_views_nnls.py and canonical analysis artefacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for aggregate calibration outputs. Defaults to <analysis-dir>/calibration_results.",
    )
    parser.add_argument(
        "--internal-cache",
        type=Path,
        default=None,
        help="Optional prediction_cache.pkl containing the canonical internal OOF stack probabilities.",
    )
    parser.add_argument(
        "--external-predictions",
        type=Path,
        default=None,
        help="Optional external prediction workbook with Asthma and proba_stack_nnls columns.",
    )
    parser.add_argument(
        "--bootstrap-rounds",
        type=int,
        default=DEFAULT_BOOTSTRAP_ROUNDS,
        help="Number of stratified bootstrap resamples for calibration CIs.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Deprecated: regenerate the verified NNLS OOF artefacts with stack_views_nnls.py before calibration.",
    )
    parser.add_argument(
        "--write-predictions",
        action="store_true",
        help="Write local patient-level probabilities to the output directory. Avoid for shared repositories.",
    )
    return parser.parse_args()


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logit(prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=float), 1e-8, 1.0 - 1e-8)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    out[~positive] = ez / (1.0 + ez)
    return out


def fit_calibration_intercept(y: np.ndarray, p: np.ndarray) -> float:
    """Fit an offset-only calibration intercept with slope fixed at one."""
    y_arr = np.asarray(y, dtype=float)
    offset = logit(p)
    alpha = 0.0
    for _ in range(60):
        mu = sigmoid(offset + alpha)
        grad = float(np.sum(y_arr - mu))
        hess = float(np.sum(mu * (1.0 - mu)))
        if hess <= 1e-12:
            break
        step = grad / hess
        alpha += step
        if abs(step) < 1e-10:
            break
    return float(alpha)


def fit_calibration_slope(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """Fit logistic calibration intercept and slope using Newton–Raphson."""
    y_arr = np.asarray(y, dtype=float)
    x = logit(p)
    design = np.column_stack([np.ones(x.shape[0]), x])
    beta = np.array([0.0, 1.0], dtype=float)
    for _ in range(80):
        mu = sigmoid(design @ beta)
        grad = design.T @ (y_arr - mu)
        weights = np.clip(mu * (1.0 - mu), 1e-8, None)
        hess = (design.T * weights) @ design
        hess.flat[::3] += 1e-10
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        beta += step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return float(beta[0]), float(beta[1])


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    """Create equal-frequency calibration bins with deterministic ordering."""
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(p, dtype=float)
    order = np.argsort(p_arr, kind="mergesort")
    groups = np.array_split(order, min(n_bins, len(order)))
    rows: List[Dict[str, Any]] = []
    for bin_index, idx in enumerate(groups, start=1):
        if len(idx) == 0:
            continue
        pred = p_arr[idx]
        obs = y_arr[idx]
        rows.append(
            {
                "bin": bin_index,
                "n": int(len(idx)),
                "events": int(obs.sum()),
                "mean_predicted": float(pred.mean()),
                "observed_fraction": float(obs.mean()),
                "absolute_gap": float(abs(pred.mean() - obs.mean())),
                "predicted_min": float(pred.min()),
                "predicted_max": float(pred.max()),
            }
        )
    return pd.DataFrame(rows)


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(p, dtype=float)
    bins = calibration_bins(y_arr, p_arr)
    intercept = fit_calibration_intercept(y_arr, p_arr)
    slope_intercept, slope = fit_calibration_slope(y_arr, p_arr)
    return {
        "n": float(len(y_arr)),
        "events": float(y_arr.sum()),
        "prevalence": float(y_arr.mean()),
        "mean_predicted": float(p_arr.mean()),
        "brier_score": float(brier_score_loss(y_arr, p_arr)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_model_intercept": slope_intercept,
        "ece": float(np.average(bins["absolute_gap"], weights=bins["n"])),
        "mce": float(bins["absolute_gap"].max()),
        "auc": float(roc_auc_score(y_arr, p_arr)),
    }


def stratified_bootstrap_indices(y: np.ndarray, rounds: int, seed: int) -> Iterable[np.ndarray]:
    y_arr = np.asarray(y, dtype=int)
    pos = np.flatnonzero(y_arr == 1)
    neg = np.flatnonzero(y_arr == 0)
    if len(pos) < 2 or len(neg) < 2:
        raise ValueError("Calibration bootstrap requires at least two observations in each class.")
    rng = np.random.default_rng(seed)
    for _ in range(int(rounds)):
        sampled_pos = rng.choice(pos, size=len(pos), replace=True)
        sampled_neg = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([sampled_pos, sampled_neg])
        rng.shuffle(idx)
        yield idx


def bootstrap_calibration(
    y: np.ndarray,
    p: np.ndarray,
    rounds: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    metric_names = [
        "brier_score",
        "calibration_intercept",
        "calibration_slope",
        "ece",
        "mce",
    ]
    values = {name: [] for name in metric_names}
    for idx in stratified_bootstrap_indices(y, rounds, seed):
        result = calibration_metrics(np.asarray(y)[idx], np.asarray(p)[idx])
        for name in metric_names:
            values[name].append(result[name])
    out: Dict[str, Dict[str, float]] = {}
    for name, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        out[name] = {
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
            "bootstrap_mean": float(np.mean(arr)),
            "bootstrap_sd": float(np.std(arr, ddof=1)),
        }
    return out


def load_internal_predictions(analysis_dir: Path, cache_path: Path, force_recompute: bool):
    """Load verified NNLS-layer OOF scores; never use full-development fitted scores.

    Upstream base OOF features are fixed. This is not an outer validation of the
    entire stacked pipeline. Run the development analysis first when absent.
    """
    sv_path = analysis_dir / "stack_views_nnls.py"
    if str(analysis_dir) not in sys.path:
        sys.path.insert(0, str(analysis_dir))
    sv = load_module("jhir_stack_views_nnls", sv_path)
    # The external-validation script imports the manuscript module by its
    # original name.  Alias the dynamically loaded canonical module so that
    # both scripts use exactly the same constants and model definitions.
    sys.modules["stack_views_nnls"] = sv
    X, y = sv.load_dataset()
    if force_recompute:
        raise ValueError("Run stack_views_nnls.py to regenerate verified NNLS OOF predictions before calibration.")
    cache = sv.load_layer_cache(cache_path, "prediction")
    key = sv.make_prediction_cache_key(X, y, sv.build_models())
    entry = cache.get(key)
    required = {"nnls_oof", "meta_fold_ids", "meta_oof_counts", "base_prob_matrix", "evaluation_version", "y_hash"}
    if isinstance(entry, dict) and required.issubset(entry):
        if entry["evaluation_version"] != sv.NNLS_EVALUATION_VERSION:
            raise ValueError("Obsolete NNLS evaluation definition in calibration input.")
        if entry["y_hash"] != sv.hash_ndarray(y.to_numpy()):
            raise ValueError("Calibration input labels or row order do not match the development cohort.")
        p = np.asarray(entry["nnls_oof"], dtype=float)
        counts = np.asarray(entry["meta_oof_counts"])
        fold_ids = np.asarray(entry["meta_fold_ids"])
        if p.shape != (len(y),) or not np.isfinite(p).all() or not ((p > 0) & (p < 1)).all():
            raise ValueError("Invalid pooled NNLS OOF prediction vector.")
        if counts.shape != p.shape or not np.all(counts == 1) or fold_ids.shape != p.shape:
            raise ValueError("Each development row must have exactly one NNLS OOF prediction.")
        expected_folds = np.zeros(len(y), dtype=int)
        splitter = sv.StratifiedKFold(n_splits=sv.K_SPLITS, shuffle=True, random_state=sv.RANDOM_STATE)
        for fold, (_, valid) in enumerate(splitter.split(X, y), start=1):
            expected_folds[valid] = fold
        if not np.array_equal(fold_ids, expected_folds):
            raise ValueError("NNLS OOF fold assignments do not match the development row order and labels.")
        source = f"{cache_path.name}:{key}:nnls_oof (NNLS-layer pooled OOF; fixed upstream base OOF features)"
        return np.asarray(y, dtype=int), p, source, sv
    raise RuntimeError("Current prediction cache lacks verified nnls_oof scores. Run the pooled-OOF development analysis first; legacy stacked scores are not accepted.")


def load_external_predictions(analysis_dir: Path, pred_path: Path, force_recompute: bool, sv):
    """Load canonical external probabilities, or train the external pipeline."""
    if pred_path.exists() and not force_recompute:
        df = pd.read_excel(pred_path)
        required = {"Asthma", "proba_stack_nnls"}
        if required.issubset(df.columns):
            return (
                df["Asthma"].astype(int).to_numpy(),
                df["proba_stack_nnls"].astype(float).to_numpy(),
                f"{pred_path.name}:proba_stack_nnls",
            )

    ext_path = analysis_dir / "external_validate_stack_views_nnls_delong.py"
    ext = load_module("jhir_external_validation", ext_path)
    X_train, y_train = sv.load_dataset()
    X_ext, y_ext, _ = ext.load_external_dataset(train_columns=list(X_train.columns))
    model_probs, _, _, _ = ext.train_base_and_stack_for_external(
        X_train=X_train,
        y_train=y_train,
        X_ext=X_ext,
    )
    return np.asarray(y_ext, dtype=int), np.asarray(model_probs["stack_nnls"], dtype=float), "recomputed_external"


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def plot_calibration(
    dataset_results: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharex=True, sharey=True)
    for ax, (label, result) in zip(axes, dataset_results.items()):
        bins = result["bins"]
        ax.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", lw=1.2, label="Perfect calibration")
        ax.plot(
            bins["mean_predicted"],
            bins["observed_fraction"],
            marker="o",
            color="#1f77b4" if label == "Internal validation" else "#d62728",
            lw=2.0,
            label="Stack-NNLS",
        )
        ax.set_title(label)
        ax.set_xlabel("Mean predicted probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.22)
        m = result["metrics"]
        ax.text(
            0.04,
            0.94,
            f"Brier = {m['brier_score']:.3f}\nSlope = {m['calibration_slope']:.2f}\nIntercept = {m['calibration_intercept']:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#cbd5e1", "alpha": 0.9},
        )
    axes[0].set_ylabel("Observed event fraction")
    axes[1].legend(loc="lower right", frameon=False)
    fig.suptitle("Calibration of the Stack-NNLS model", y=1.02, fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    output_dir = (args.output_dir or analysis_dir / "calibration_results").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.bootstrap_rounds < 100:
        raise ValueError("Use at least 100 bootstrap rounds for a useful calibration CI.")

    cache_path = (args.internal_cache or analysis_dir / "prediction_cache.pkl").resolve()
    ext_pred_path = (
        args.external_predictions
        or analysis_dir / "external_stack_nnls_predictions_bootstrap_bca.xlsx"
    ).resolve()
    script_paths = [
        analysis_dir / "stack_views_nnls.py",
        analysis_dir / "external_validate_stack_views_nnls_delong.py",
        Path(__file__).resolve(),
    ]
    script_hashes = {str(p.name): sha256_file(p) for p in script_paths if p.exists()}

    y_int, p_int, internal_source, sv = load_internal_predictions(
        analysis_dir=analysis_dir,
        cache_path=cache_path,
        force_recompute=args.force_recompute,
    )
    y_ext, p_ext, external_source = load_external_predictions(
        analysis_dir=analysis_dir,
        pred_path=ext_pred_path,
        force_recompute=args.force_recompute,
        sv=sv,
    )

    datasets = {
        "Internal validation": (y_int, p_int, internal_source),
        "External validation": (y_ext, p_ext, external_source),
    }
    results: Dict[str, Dict[str, Any]] = {}
    summary_rows = []
    bin_rows = []
    prediction_rows = []
    for label, (y, p, source) in datasets.items():
        metrics = calibration_metrics(y, p)
        ci = bootstrap_calibration(y, p, args.bootstrap_rounds, args.seed)
        bins = calibration_bins(y, p)
        results[label] = {
            "metrics": metrics,
            "bootstrap_ci": ci,
            "bins": bins.to_dict(orient="records"),
            "source": source,
        }
        row = {"dataset": label, **metrics, "source": source}
        for metric, interval in ci.items():
            row[f"{metric}_ci_low"] = interval["ci_low"]
            row[f"{metric}_ci_high"] = interval["ci_high"]
        summary_rows.append(row)
        for bin_row in bins.to_dict(orient="records"):
            bin_rows.append({"dataset": label, **bin_row})
        if args.write_predictions:
            prediction_rows.extend(
                {"dataset": label, "row": int(i), "outcome": int(yy), "predicted_probability": float(pp)}
                for i, (yy, pp) in enumerate(zip(y, p))
            )

    plot_path = output_dir / "calibration_plot.png"
    plot_input = {
        label: {"metrics": value["metrics"], "bins": pd.DataFrame(value["bins"])}
        for label, value in results.items()
    }
    plot_calibration(plot_input, plot_path)

    summary_df = pd.DataFrame(summary_rows)
    bins_df = pd.DataFrame(bin_rows)
    workbook_path = output_dir / "calibration_metrics.xlsx"
    with pd.ExcelWriter(workbook_path) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        bins_df.to_excel(writer, sheet_name="calibration_bins", index=False)
        if prediction_rows:
            pd.DataFrame(prediction_rows).to_excel(writer, sheet_name="predictions", index=False)

    report_lines = [
        "# Stack-NNLS calibration analysis",
        "",
        f"Script version: {SCRIPT_VERSION}",
        f"Bootstrap: stratified percentile bootstrap, {args.bootstrap_rounds} resamples, seed {args.seed}",
        "",
        "The primary model was not refit or recalibrated. Raw Stack-NNLS probabilities were assessed in the internal OOF cohort and the independent external cohort.",
        "",
        "| Dataset | Brier score | Calibration intercept | Calibration slope | ECE | MCE | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, result in results.items():
        m = result["metrics"]
        report_lines.append(
            f"| {label} | {m['brier_score']:.3f} ({result['bootstrap_ci']['brier_score']['ci_low']:.3f}–{result['bootstrap_ci']['brier_score']['ci_high']:.3f}) | "
            f"{m['calibration_intercept']:.3f} ({result['bootstrap_ci']['calibration_intercept']['ci_low']:.3f}–{result['bootstrap_ci']['calibration_intercept']['ci_high']:.3f}) | "
            f"{m['calibration_slope']:.3f} ({result['bootstrap_ci']['calibration_slope']['ci_low']:.3f}–{result['bootstrap_ci']['calibration_slope']['ci_high']:.3f}) | "
            f"{m['ece']:.3f} ({result['bootstrap_ci']['ece']['ci_low']:.3f}–{result['bootstrap_ci']['ece']['ci_high']:.3f}) | "
            f"{m['mce']:.3f} ({result['bootstrap_ci']['mce']['ci_low']:.3f}–{result['bootstrap_ci']['mce']['ci_high']:.3f}) | {m['auc']:.3f} |"
        )
    report_lines.extend(
        [
            "",
            "Interpretation: an intercept near 0 and slope near 1 indicate good overall calibration. Positive intercepts indicate systematic underprediction; negative intercepts indicate systematic overprediction. Slopes below 1 indicate predictions that are too extreme.",
            "",
            "## Provenance",
            "",
            f"- Internal source: `{internal_source}`",
            f"- External source: `{external_source}`",
            "- Script SHA-256:",
        ]
    )
    report_lines.extend(f"  - `{name}`: `{digest}`" for name, digest in script_hashes.items())
    report_path = output_dir / "calibration_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    json_path = output_dir / "calibration_results.json"
    json_path.write_text(
        json.dumps(
            jsonable(
                {
                    "script_version": SCRIPT_VERSION,
                    "bootstrap_rounds": args.bootstrap_rounds,
                    "seed": args.seed,
                    "internal_source": internal_source,
                    "external_source": external_source,
                    "script_sha256": script_hashes,
                    "datasets": results,
                }
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Calibration summary saved to {workbook_path}")
    print(f"Calibration report saved to {report_path}")
    print(f"Calibration plot saved to {plot_path} and {plot_path.with_suffix('.pdf')}")
    for label, result in results.items():
        m = result["metrics"]
        print(
            f"{label}: Brier={m['brier_score']:.4f}, intercept={m['calibration_intercept']:.4f}, "
            f"slope={m['calibration_slope']:.4f}, ECE={m['ece']:.4f}, MCE={m['mce']:.4f}, AUC={m['auc']:.4f}"
        )


if __name__ == "__main__":
    main()
