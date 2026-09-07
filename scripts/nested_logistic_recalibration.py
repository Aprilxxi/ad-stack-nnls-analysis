"""Nested development-set logistic recalibration for the canonical Stack-NNLS model.

This is a probability-calibration layer only.  It does not refit the primary
Stack-NNLS model and it never uses external-cohort outcomes to fit or select a
calibrator.  The canonical development OOF probabilities are split into
outer folds.  Within each outer-training partition, an inner cross-validation
chooses between the identity mapping and standard two-parameter logistic
recalibration by mean Brier score.  The selected mapping is then fitted on the
complete outer-training partition and evaluated on the untouched outer fold.

After the nested development assessment, the selected mapping is refitted on
all development OOF probabilities and applied once to the locked external
Stack-NNLS probabilities.  Aggregate metrics, bootstrap confidence intervals,
fold-level coefficients, calibration tables, and a raw-versus-recalibrated
reliability plot are written to the output directory.  Patient-level
predictions are not written unless --write-predictions is explicitly supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold


SCRIPT_VERSION = "1.1.0-nnls-pooled-oof"
DEFAULT_OUTER_FOLDS = 5
DEFAULT_INNER_FOLDS = 5
DEFAULT_BOOTSTRAP_ROUNDS = 2000
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Validated analysis directory containing the Stack-NNLS artefacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <analysis-dir>/nested_logistic_recalibration_results.",
    )
    parser.add_argument("--internal-cache", type=Path, default=None)
    parser.add_argument("--external-predictions", type=Path, default=None)
    parser.add_argument("--outer-folds", type=int, default=DEFAULT_OUTER_FOLDS)
    parser.add_argument("--inner-folds", type=int, default=DEFAULT_INNER_FOLDS)
    parser.add_argument("--bootstrap-rounds", type=int, default=DEFAULT_BOOTSTRAP_ROUNDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--write-predictions",
        action="store_true",
        help="Write patient-level raw and recalibrated probabilities to the workbook.",
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    out[~positive] = ez / (1.0 + ez)
    return out


def logit(prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=float), 1e-8, 1.0 - 1e-8)
    return np.log(p / (1.0 - p))


def fit_logistic_calibrator(y: np.ndarray, p: np.ndarray, cal) -> Tuple[float, float]:
    """Fit standard logistic calibration: logit(y) ~ intercept + slope*logit(p)."""
    intercept, slope = cal.fit_calibration_slope(np.asarray(y, dtype=int), np.asarray(p, dtype=float))
    return float(intercept), float(slope)


def apply_calibrator(p: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    return sigmoid(float(intercept) + float(slope) * logit(np.asarray(p, dtype=float)))


def evaluate_inner_selection(
    y_train: np.ndarray,
    p_train: np.ndarray,
    inner_folds: int,
    seed: int,
) -> Dict[str, Any]:
    """Choose identity versus logistic recalibration using only outer training data."""
    splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    identity_scores: List[float] = []
    logistic_scores: List[float] = []
    for inner_train_idx, inner_valid_idx in splitter.split(p_train, y_train):
        inner_y = y_train[inner_train_idx]
        inner_p = p_train[inner_train_idx]
        intercept, slope = fit_logistic_calibrator(inner_y, inner_p, cal=_CAL)
        valid_y = y_train[inner_valid_idx]
        valid_p = p_train[inner_valid_idx]
        identity_scores.append(float(brier_score_loss(valid_y, valid_p)))
        logistic_scores.append(
            float(brier_score_loss(valid_y, apply_calibrator(valid_p, intercept, slope)))
        )
    identity_mean = float(np.mean(identity_scores))
    logistic_mean = float(np.mean(logistic_scores))
    selected = "logistic" if logistic_mean <= identity_mean else "identity"
    return {
        "selected_method": selected,
        "inner_identity_brier_mean": identity_mean,
        "inner_logistic_brier_mean": logistic_mean,
        "inner_identity_brier_sd": float(np.std(identity_scores, ddof=1)),
        "inner_logistic_brier_sd": float(np.std(logistic_scores, ddof=1)),
    }


def nested_development_recalibration(
    y: np.ndarray,
    p: np.ndarray,
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Generate outer-fold recalibrated development probabilities."""
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(p, dtype=float)
    if outer_folds < 3 or inner_folds < 3:
        raise ValueError("Use at least 3 outer and inner folds.")
    if min(np.bincount(y_arr)) < outer_folds:
        raise ValueError("Each class must have at least outer_folds observations.")

    calibrated = np.full_like(p_arr, np.nan, dtype=float)
    rows: List[Dict[str, Any]] = []
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    for fold, (train_idx, valid_idx) in enumerate(outer.split(p_arr, y_arr), start=1):
        y_train, p_train = y_arr[train_idx], p_arr[train_idx]
        selection = evaluate_inner_selection(
            y_train=y_train,
            p_train=p_train,
            inner_folds=inner_folds,
            seed=seed + fold,
        )
        method = selection["selected_method"]
        if method == "logistic":
            intercept, slope = fit_logistic_calibrator(y_train, p_train, cal=_CAL)
            p_valid_calibrated = apply_calibrator(p_arr[valid_idx], intercept, slope)
        else:
            intercept, slope = 0.0, 1.0
            p_valid_calibrated = p_arr[valid_idx].copy()
        calibrated[valid_idx] = p_valid_calibrated
        rows.append(
            {
                "outer_fold": fold,
                "train_n": int(len(train_idx)),
                "validation_n": int(len(valid_idx)),
                "train_events": int(y_train.sum()),
                "validation_events": int(y_arr[valid_idx].sum()),
                "selected_method": method,
                "calibration_intercept": float(intercept),
                "calibration_slope": float(slope),
                **selection,
            }
        )
    if np.isnan(calibrated).any():
        raise RuntimeError("Nested outer-fold calibration did not produce all probabilities.")
    return calibrated, rows


def full_development_selection(
    y: np.ndarray,
    p: np.ndarray,
    inner_folds: int,
    seed: int,
) -> Dict[str, Any]:
    selection = evaluate_inner_selection(
        y_train=np.asarray(y, dtype=int),
        p_train=np.asarray(p, dtype=float),
        inner_folds=inner_folds,
        seed=seed,
    )
    if selection["selected_method"] == "logistic":
        intercept, slope = fit_logistic_calibrator(y, p, cal=_CAL)
    else:
        intercept, slope = 0.0, 1.0
    return {
        **selection,
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
    }


def bootstrap_metrics(y: np.ndarray, p: np.ndarray, cal, rounds: int, seed: int) -> Dict[str, Dict[str, float]]:
    result = cal.bootstrap_calibration(
        np.asarray(y, dtype=int),
        np.asarray(p, dtype=float),
        rounds=rounds,
        seed=seed,
    )
    return result


def build_summary_row(
    dataset: str,
    method: str,
    y: np.ndarray,
    p: np.ndarray,
    cal,
    bootstrap_rounds: int,
    seed: int,
    source: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    metrics = cal.calibration_metrics(np.asarray(y, dtype=int), np.asarray(p, dtype=float))
    ci = bootstrap_metrics(y, p, cal, rounds=bootstrap_rounds, seed=seed)
    row: Dict[str, Any] = {
        "dataset": dataset,
        "method": method,
        **metrics,
        "source": source,
    }
    for metric, interval in ci.items():
        row[f"{metric}_ci_low"] = interval["ci_low"]
        row[f"{metric}_ci_high"] = interval["ci_high"]
    return row, {"metrics": metrics, "bootstrap_ci": ci}, cal.calibration_bins(y, p)


def plot_recalibration(
    plot_data: Dict[str, Dict[str, Dict[str, Any]]],
    out_path: Path,
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7), sharex=True, sharey=True)
    for ax, (dataset, methods) in zip(axes, plot_data.items()):
        ax.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", lw=1.2, label="Perfect calibration")
        raw_bins = methods["Raw Stack-NNLS"]["bins"]
        cal_bins = methods["Nested logistic recalibration"]["bins"]
        ax.plot(
            raw_bins["mean_predicted"],
            raw_bins["observed_fraction"],
            marker="o",
            color="#9ca3af",
            lw=1.6,
            linestyle=":",
            label="Raw Stack-NNLS",
        )
        color = "#1f77b4" if dataset == "Development (outer CV)" else "#d62728"
        ax.plot(
            cal_bins["mean_predicted"],
            cal_bins["observed_fraction"],
            marker="o",
            color=color,
            lw=2.0,
            label="Logistic recalibration",
        )
        ax.set_title(dataset)
        ax.set_xlabel("Mean predicted probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.22)
        m = methods["Nested logistic recalibration"]["metrics"]
        ax.text(
            0.04,
            0.95,
            f"Brier = {m['brier_score']:.3f}\nSlope = {m['calibration_slope']:.2f}\nIntercept = {m['calibration_intercept']:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#cbd5e1", "alpha": 0.9},
        )
    axes[0].set_ylabel("Observed event fraction")
    axes[1].legend(loc="lower right", frameon=False)
    fig.suptitle("Nested logistic recalibration of the Stack-NNLS model", y=1.02, fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


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


def main() -> None:
    global _CAL
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    output_dir = (args.output_dir or analysis_dir / "nested_logistic_recalibration_results").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.bootstrap_rounds < 100:
        raise ValueError("Use at least 100 bootstrap rounds for a useful CI.")

    helper_path = Path(__file__).resolve().with_name("calibration_analysis.py")
    _CAL = load_module("jhir_calibration_helper", helper_path)
    cache_path = (args.internal_cache or analysis_dir / "prediction_cache.pkl").resolve()
    ext_path = (
        args.external_predictions
        or analysis_dir / "external_stack_nnls_predictions_bootstrap_bca.xlsx"
    ).resolve()
    y_int, p_int, internal_source, _ = _CAL.load_internal_predictions(
        analysis_dir=analysis_dir,
        cache_path=cache_path,
        force_recompute=False,
    )
    y_ext, p_ext, external_source = _CAL.load_external_predictions(
        analysis_dir=analysis_dir,
        pred_path=ext_path,
        force_recompute=False,
        sv=sys.modules["jhir_stack_views_nnls"],
    )

    nested_p_int, fold_rows = nested_development_recalibration(
        y=np.asarray(y_int, dtype=int),
        p=np.asarray(p_int, dtype=float),
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        seed=args.seed,
    )
    final_cal = full_development_selection(
        y=np.asarray(y_int, dtype=int),
        p=np.asarray(p_int, dtype=float),
        inner_folds=args.inner_folds,
        seed=args.seed + 1000,
    )
    p_ext_cal = apply_calibrator(
        np.asarray(p_ext, dtype=float),
        final_cal["calibration_intercept"],
        final_cal["calibration_slope"],
    )

    datasets = {
        "Development (outer CV)": {
            "Raw Stack-NNLS": (np.asarray(y_int, dtype=int), np.asarray(p_int, dtype=float), internal_source),
            "Nested logistic recalibration": (np.asarray(y_int, dtype=int), nested_p_int, "outer-fold fitted calibrators"),
        },
        "External validation": {
            "Raw Stack-NNLS": (np.asarray(y_ext, dtype=int), np.asarray(p_ext, dtype=float), external_source),
            "Nested logistic recalibration": (np.asarray(y_ext, dtype=int), p_ext_cal, "final development calibrator applied to external probabilities"),
        },
    }
    summary_rows: List[Dict[str, Any]] = []
    bin_rows: List[Dict[str, Any]] = []
    result_json: Dict[str, Any] = {}
    plot_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    prediction_rows: List[Dict[str, Any]] = []
    for dataset, methods in datasets.items():
        result_json[dataset] = {}
        plot_data[dataset] = {}
        for method, (y, p, source) in methods.items():
            row, result, bins = build_summary_row(
                dataset=dataset,
                method=method,
                y=y,
                p=p,
                cal=_CAL,
                bootstrap_rounds=args.bootstrap_rounds,
                seed=args.seed,
                source=source,
            )
            summary_rows.append(row)
            for bin_row in bins.to_dict(orient="records"):
                bin_rows.append({"dataset": dataset, "method": method, **bin_row})
            result_json[dataset][method] = {**result, "source": source, "bins": bins.to_dict(orient="records")}
            plot_data[dataset][method] = {"metrics": result["metrics"], "bins": bins}
            if args.write_predictions:
                prediction_rows.extend(
                    {
                        "dataset": dataset,
                        "method": method,
                        "row": int(i),
                        "outcome": int(yy),
                        "predicted_probability": float(pp),
                    }
                    for i, (yy, pp) in enumerate(zip(y, p))
                )

    plot_path = output_dir / "nested_logistic_recalibration_plot.png"
    plot_recalibration(plot_data, plot_path)
    workbook_path = output_dir / "nested_logistic_recalibration_metrics.xlsx"
    with pd.ExcelWriter(workbook_path) as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(bin_rows).to_excel(writer, sheet_name="calibration_bins", index=False)
        pd.DataFrame(fold_rows).to_excel(writer, sheet_name="outer_fold_coefficients", index=False)
        pd.DataFrame([final_cal]).to_excel(writer, sheet_name="final_calibrator", index=False)
        if prediction_rows:
            pd.DataFrame(prediction_rows).to_excel(writer, sheet_name="predictions", index=False)

    script_paths = [helper_path, Path(__file__).resolve(), analysis_dir / "stack_views_nnls.py", analysis_dir / "external_validate_stack_views_nnls_delong.py"]
    script_hashes = {p.name: sha256_file(p) for p in script_paths if p.exists()}
    report_lines = [
        "# Nested logistic recalibration of Stack-NNLS",
        "",
        f"Script version: {SCRIPT_VERSION}",
        f"Outer folds: {args.outer_folds}; inner folds: {args.inner_folds}; stratified bootstrap: {args.bootstrap_rounds} resamples; seed: {args.seed}",
        "",
        "Development calibration used pooled NNLS-layer OOF probabilities and outcomes. Upstream base OOF features were fixed; neither NNLS-layer cross-validation nor calibration-stage cross-validation constitutes outer validation of the entire pipeline. External outcomes were used only for evaluation, not for fitting or selecting the calibrator.",
        "",
        "The inner loop selected identity versus standard two-parameter logistic recalibration by mean Brier score.  The selected mapping was fitted within each outer development-training partition and evaluated on its untouched outer fold.  A final mapping was then fitted on all development OOF probabilities and applied once to the external probabilities.",
        "",
        "| Dataset | Method | Brier score | Calibration intercept | Calibration slope | ECE | MCE | AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            f"| {row['dataset']} | {row['method']} | {row['brier_score']:.3f} ({row['brier_score_ci_low']:.3f}–{row['brier_score_ci_high']:.3f}) | "
            f"{row['calibration_intercept']:.3f} ({row['calibration_intercept_ci_low']:.3f}–{row['calibration_intercept_ci_high']:.3f}) | "
            f"{row['calibration_slope']:.3f} ({row['calibration_slope_ci_low']:.3f}–{row['calibration_slope_ci_high']:.3f}) | "
            f"{row['ece']:.3f} ({row['ece_ci_low']:.3f}–{row['ece_ci_high']:.3f}) | "
            f"{row['mce']:.3f} ({row['mce_ci_low']:.3f}–{row['mce_ci_high']:.3f}) | {row['auc']:.3f} |"
        )
    report_lines.extend(
        [
            "",
            "## Locked final calibrator",
            "",
            f"- Selected method: `{final_cal['selected_method']}`",
            f"- Development calibration intercept: `{final_cal['calibration_intercept']:.6f}`",
            f"- Development calibration slope: `{final_cal['calibration_slope']:.6f}`",
            f"- Inner-CV identity Brier mean: `{final_cal['inner_identity_brier_mean']:.6f}`",
            f"- Inner-CV logistic Brier mean: `{final_cal['inner_logistic_brier_mean']:.6f}`",
            "",
            "A single increasing logistic mapping preserves AUC on external scores. Pooled development scores use different outer-fold calibrators, so their pooled AUC may change slightly. This is post-hoc probability recalibration of the fixed scores.",
            "",
            "## Provenance",
            "",
            f"- Internal source: `{internal_source}`",
            f"- External source: `{external_source}`",
            "- Script SHA-256:",
        ]
    )
    report_lines.extend(f"  - `{name}`: `{digest}`" for name, digest in script_hashes.items())
    report_path = output_dir / "nested_logistic_recalibration_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    payload = {
        "script_version": SCRIPT_VERSION,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "bootstrap_rounds": args.bootstrap_rounds,
        "seed": args.seed,
        "internal_source": internal_source,
        "external_source": external_source,
        "final_calibrator": final_cal,
        "outer_fold_coefficients": fold_rows,
        "script_sha256": script_hashes,
        "datasets": result_json,
    }
    json_path = output_dir / "nested_logistic_recalibration_results.json"
    json_path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Nested recalibration summary saved to {workbook_path}")
    print(f"Nested recalibration report saved to {report_path}")
    print(f"Nested recalibration plot saved to {plot_path} and {plot_path.with_suffix('.pdf')}")
    print(
        f"Final calibrator: method={final_cal['selected_method']}, intercept={final_cal['calibration_intercept']:.6f}, slope={final_cal['calibration_slope']:.6f}"
    )
    for row in summary_rows:
        print(
            f"{row['dataset']} / {row['method']}: Brier={row['brier_score']:.4f}, "
            f"intercept={row['calibration_intercept']:.4f}, slope={row['calibration_slope']:.4f}, "
            f"ECE={row['ece']:.4f}, MCE={row['mce']:.4f}, AUC={row['auc']:.4f}"
        )


_CAL = None


if __name__ == "__main__":
    main()
