from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu


DEFAULT_TRAIN_FILE = "train_pruned_ratio_0p04.xlsx"
DEFAULT_EXTERNAL_FILE = "External_2.0.xlsx"
DEFAULT_MODEL_SCRIPT = "stack_views_nnls.py"
DEFAULT_OUTPUT_PREFIX = "table1_train_vs_external"
DEFAULT_AGE_CUTOFF = 8.0
DEFAULT_RECORD_COUNT_COL = "\u75c5\u5386\u8bb0\u5f55\u6570\u91cf"
DEFAULT_RECORD_COUNT_MIN = 5
OUTCOME_COL = "Asthma"


def dedupe(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(items))


def parse_model_constants(model_script: Path) -> Dict[str, object]:
    wanted = {
        "DEMO_COLS",
        "BLOOD_VIEW",
        "MERGED_ALLERGEN_MAPPING",
        "ALLERGEN_RAW_NUMERIC",
        "ALLERGEN_ENGINEERED_NUMERIC",
        "IGNORED_ALLERGEN_FEATURES",
        "USE_ENGINEERED_ALLERGEN_FEATURES",
        "HISTORY_BASE_COLS",
        "DEMO_POLICY",
        "RECORD_COUNT_COL",
    }
    text = model_script.read_text(encoding="utf-8-sig")
    tree = ast.parse(text, filename=str(model_script))

    values: Dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            for target in targets:
                name = target.id
                if name not in wanted:
                    continue
                try:
                    values[name] = ast.literal_eval(node.value)
                except Exception:
                    continue
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if name not in wanted or node.value is None:
                continue
            try:
                values[name] = ast.literal_eval(node.value)
            except Exception:
                continue

    required = {"DEMO_COLS", "BLOOD_VIEW", "MERGED_ALLERGEN_MAPPING", "HISTORY_BASE_COLS", "DEMO_POLICY"}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"Missing constants in {model_script.name}: {missing}")
    return values


@dataclass
class ModelFeatures:
    ordered_features: List[str]
    categorical_features: set[str]
    feature_group: Dict[str, str]
    record_count_col: str


def build_model_features(constants: Dict[str, object]) -> ModelFeatures:
    demo_cols = list(constants.get("DEMO_COLS", []))
    blood_view = list(constants.get("BLOOD_VIEW", []))
    history_base = list(constants.get("HISTORY_BASE_COLS", []))
    merged_mapping = dict(constants.get("MERGED_ALLERGEN_MAPPING", {}))
    allergen_raw = list(constants.get("ALLERGEN_RAW_NUMERIC", []))
    allergen_engineered = list(constants.get("ALLERGEN_ENGINEERED_NUMERIC", []))
    ignored_allergen = {str(x).lower() for x in constants.get("IGNORED_ALLERGEN_FEATURES", set())}
    use_engineered = bool(constants.get("USE_ENGINEERED_ALLERGEN_FEATURES", False))
    demo_policy = str(constants.get("DEMO_POLICY", "all"))
    record_count_col = str(constants.get("RECORD_COUNT_COL", DEFAULT_RECORD_COUNT_COL))

    merged_allergen_cols = list(merged_mapping.keys())
    allergen_view = dedupe(merged_allergen_cols + allergen_raw + demo_cols)
    allergen_view = [c for c in allergen_view if c.lower() not in ignored_allergen]
    if use_engineered:
        allergen_view = dedupe(allergen_view + allergen_engineered)

    history_view = history_base + demo_cols

    def apply_demo_policy(view_cols: Sequence[str], view_name: str) -> List[str]:
        if demo_policy == "all":
            return list(view_cols)
        if demo_policy in {"none", "separate"}:
            return [c for c in view_cols if c not in demo_cols]
        if demo_policy == "history_only" and view_name != "history":
            return [c for c in view_cols if c not in demo_cols]
        return list(view_cols)

    blood_view = apply_demo_policy(blood_view, "blood")
    allergen_view = apply_demo_policy(allergen_view, "allergen")
    history_view = apply_demo_policy(history_view, "history")

    ordered_features = dedupe(
        demo_cols
        + [c for c in blood_view if c not in demo_cols]
        + [c for c in allergen_view if c not in demo_cols]
        + [c for c in history_view if c not in demo_cols]
    )
    if demo_policy == "separate":
        ordered_features = dedupe(ordered_features + demo_cols)

    feature_group: Dict[str, str] = {}
    for col in demo_cols:
        feature_group[col] = "Demographics"
    for col in blood_view:
        if col not in feature_group:
            feature_group[col] = "Blood tests"
    for col in allergen_view:
        if col not in feature_group:
            feature_group[col] = "Allergen profile"
    for col in history_base:
        feature_group[col] = "Past/family history"

    feature_group.setdefault(OUTCOME_COL, "Outcome")

    categorical_features = set(merged_allergen_cols + history_base + ["Gender", OUTCOME_COL])
    return ModelFeatures(
        ordered_features=ordered_features,
        categorical_features=categorical_features,
        feature_group=feature_group,
        record_count_col=record_count_col,
    )


def format_pvalue(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def format_continuous(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "NA"
    q1 = values.quantile(0.25)
    q2 = values.quantile(0.50)
    q3 = values.quantile(0.75)
    return f"{q2:.2f} [{q1:.2f}, {q3:.2f}]"


def to_binary(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return values.astype(int)
    return (values > 0).astype(int)


def format_binary(series: pd.Series) -> str:
    values = to_binary(series)
    n = int(values.shape[0])
    if n == 0:
        return "NA"
    pos = int(values.sum())
    pct = 100.0 * pos / n
    return f"{pos}/{n} ({pct:.1f}%)"


def missing_info(series: pd.Series) -> str:
    n_total = int(series.shape[0])
    if n_total == 0:
        return "0/0 (0.0%)"
    n_missing = int(series.isna().sum())
    pct = 100.0 * n_missing / n_total
    return f"{n_missing}/{n_total} ({pct:.1f}%)"


def continuous_test(train: pd.Series, external: pd.Series) -> Tuple[str, float]:
    x = pd.to_numeric(train, errors="coerce").dropna()
    y = pd.to_numeric(external, errors="coerce").dropna()
    if x.empty or y.empty:
        return "Mann-Whitney U", np.nan
    try:
        p = mannwhitneyu(x, y, alternative="two-sided").pvalue
    except ValueError:
        p = np.nan
    return "Mann-Whitney U", float(p)


def binary_test(train: pd.Series, external: pd.Series) -> Tuple[str, float]:
    x = to_binary(train)
    y = to_binary(external)
    if x.empty or y.empty:
        return "Fisher/Chi-square", np.nan

    x_pos = int(x.sum())
    x_neg = int(x.shape[0] - x_pos)
    y_pos = int(y.sum())
    y_neg = int(y.shape[0] - y_pos)
    table = np.array([[x_pos, x_neg], [y_pos, y_neg]], dtype=int)

    if (table.sum(axis=1) == 0).any() or (table.sum(axis=0) == 0).any():
        return "Fisher/Chi-square", np.nan

    try:
        _, _, _, expected = chi2_contingency(table, correction=False)
        if (expected < 5).any():
            p = fisher_exact(table).pvalue
            return "Fisher exact", float(p)
        p = chi2_contingency(table, correction=False).pvalue
        return "Chi-square", float(p)
    except ValueError:
        return "Fisher/Chi-square", np.nan


def continuous_smd(train: pd.Series, external: pd.Series) -> float:
    x = pd.to_numeric(train, errors="coerce").dropna()
    y = pd.to_numeric(external, errors="coerce").dropna()
    if x.empty or y.empty:
        return np.nan
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled_sd = np.sqrt((vx + vy) / 2.0)
    if pooled_sd == 0 or np.isnan(pooled_sd):
        return np.nan
    return float((x.mean() - y.mean()) / pooled_sd)


def binary_smd(train: pd.Series, external: pd.Series) -> float:
    x = to_binary(train)
    y = to_binary(external)
    if x.empty or y.empty:
        return np.nan
    p1 = x.mean()
    p2 = y.mean()
    denom = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2.0)
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float((p1 - p2) / denom)


def looks_binary(train: pd.Series, external: pd.Series) -> bool:
    values = pd.concat([train, external], ignore_index=True)
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return False
    uniq = set(values.unique().tolist())
    return uniq.issubset({0, 1})


def filter_train(
    df: pd.DataFrame, age_cutoff: float, record_count_col: str, record_count_min: int
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    flow = {"n_raw": int(df.shape[0])}
    work = df.copy()

    if "age" in work.columns:
        age = pd.to_numeric(work["age"], errors="coerce")
        work = work[age <= age_cutoff].reset_index(drop=True)
    flow["n_age_le_cutoff"] = int(work.shape[0])

    if record_count_col in work.columns:
        rec = pd.to_numeric(work[record_count_col], errors="coerce")
        work = work[rec > record_count_min].reset_index(drop=True)
        flow["n_record_gt_min"] = int(work.shape[0])
    else:
        flow["n_record_gt_min"] = int(work.shape[0])

    flow["n_final"] = int(work.shape[0])
    return work, flow


def filter_external(df: pd.DataFrame, age_cutoff: float) -> Tuple[pd.DataFrame, Dict[str, int]]:
    flow = {"n_raw": int(df.shape[0])}
    work = df.copy()

    if "age" in work.columns:
        age = pd.to_numeric(work["age"], errors="coerce")
        work = work[age <= age_cutoff].reset_index(drop=True)
    flow["n_age_le_cutoff"] = int(work.shape[0])
    flow["n_final"] = int(work.shape[0])
    return work, flow


def build_comparison_table(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    model_features: ModelFeatures,
    left_summary_col: str,
    right_summary_col: str,
    left_missing_col: str,
    right_missing_col: str,
    include_outcome: bool = True,
) -> pd.DataFrame:
    feature_cols = [c for c in model_features.ordered_features if (c in left_df.columns or c in right_df.columns)]
    if include_outcome and (OUTCOME_COL in left_df.columns or OUTCOME_COL in right_df.columns):
        feature_cols = dedupe(feature_cols + [OUTCOME_COL])
    if not include_outcome:
        feature_cols = [c for c in feature_cols if c != OUTCOME_COL]

    rows = []
    for col in feature_cols:
        left_series = left_df[col] if col in left_df.columns else pd.Series(dtype=float)
        right_series = right_df[col] if col in right_df.columns else pd.Series(dtype=float)

        is_binary = col in model_features.categorical_features or looks_binary(left_series, right_series)
        if is_binary:
            summary_left = format_binary(left_series)
            summary_right = format_binary(right_series)
            test_name, pvalue = binary_test(left_series, right_series)
            smd = binary_smd(left_series, right_series)
            var_type = "Binary"
        else:
            summary_left = format_continuous(left_series)
            summary_right = format_continuous(right_series)
            test_name, pvalue = continuous_test(left_series, right_series)
            smd = continuous_smd(left_series, right_series)
            var_type = "Continuous"

        rows.append(
            {
                "group": model_features.feature_group.get(col, "Other model feature"),
                "variable": col,
                "type": var_type,
                left_summary_col: summary_left,
                right_summary_col: summary_right,
                left_missing_col: missing_info(left_series),
                right_missing_col: missing_info(right_series),
                "test": test_name,
                "p_value": format_pvalue(pvalue),
                "smd": "" if np.isnan(smd) else f"{smd:.3f}",
            }
        )

    return pd.DataFrame(rows)


def build_table(
    train_df: pd.DataFrame,
    external_df: pd.DataFrame,
    model_features: ModelFeatures,
) -> pd.DataFrame:
    return build_comparison_table(
        left_df=train_df,
        right_df=external_df,
        model_features=model_features,
        left_summary_col="train_summary",
        right_summary_col="external_summary",
        left_missing_col="train_missing",
        right_missing_col="external_missing",
        include_outcome=True,
    )


def split_by_outcome(df: pd.DataFrame, outcome_col: str = OUTCOME_COL) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    if outcome_col not in df.columns:
        raise KeyError(f"Outcome column not found: {outcome_col}")

    outcome_num = pd.to_numeric(df[outcome_col], errors="coerce")
    asthma_mask = outcome_num > 0
    non_asthma_mask = outcome_num == 0

    asthma_df = df.loc[asthma_mask].reset_index(drop=True)
    non_asthma_df = df.loc[non_asthma_mask].reset_index(drop=True)
    split = {
        "n_total": int(df.shape[0]),
        "n_asthma": int(asthma_mask.sum()),
        "n_non_asthma": int(non_asthma_mask.sum()),
        "n_outcome_missing": int(outcome_num.isna().sum()),
    }
    return asthma_df, non_asthma_df, split


def build_dataset_asthma_table(
    df: pd.DataFrame,
    model_features: ModelFeatures,
    dataset_label: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    asthma_df, non_asthma_df, split = split_by_outcome(df, outcome_col=OUTCOME_COL)
    table = build_comparison_table(
        left_df=asthma_df,
        right_df=non_asthma_df,
        model_features=model_features,
        left_summary_col=f"{dataset_label} asthma (n={split['n_asthma']})",
        right_summary_col=f"{dataset_label} non-asthma (n={split['n_non_asthma']})",
        left_missing_col=f"{dataset_label} asthma missing",
        right_missing_col=f"{dataset_label} non-asthma missing",
        include_outcome=False,
    )
    return table, split


def save_outputs(
    flow_df: pd.DataFrame,
    table_df: pd.DataFrame,
    output_dir: Path,
    output_prefix: str,
    extra_sheets: Dict[str, pd.DataFrame] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / f"{output_prefix}.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        flow_df.to_excel(writer, sheet_name="sample_flow", index=False)
        table_df.to_excel(writer, sheet_name="table1", index=False)
        if extra_sheets:
            for sheet_name, df in extra_sheets.items():
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    return excel_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate eClinicalMedicine-style Table 1 for train vs external cohorts."
    )
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--train-file", type=str, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--external-file", type=str, default=DEFAULT_EXTERNAL_FILE)
    parser.add_argument("--model-script", type=str, default=DEFAULT_MODEL_SCRIPT)
    parser.add_argument("--age-cutoff", type=float, default=DEFAULT_AGE_CUTOFF)
    parser.add_argument("--record-count-min", type=int, default=DEFAULT_RECORD_COUNT_MIN)
    parser.add_argument("--output-prefix", type=str, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    train_path = base_dir / args.train_file
    external_path = base_dir / args.external_file
    model_script_path = base_dir / args.model_script
    output_dir = args.output_dir.resolve() if args.output_dir else base_dir

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not external_path.exists():
        raise FileNotFoundError(f"External file not found: {external_path}")
    if not model_script_path.exists():
        raise FileNotFoundError(f"Model script not found: {model_script_path}")

    constants = parse_model_constants(model_script_path)
    model_features = build_model_features(constants)

    train_raw = pd.read_excel(train_path)
    external_raw = pd.read_excel(external_path)

    train_filtered, train_flow = filter_train(
        train_raw,
        age_cutoff=args.age_cutoff,
        record_count_col=model_features.record_count_col,
        record_count_min=args.record_count_min,
    )
    external_filtered, external_flow = filter_external(external_raw, age_cutoff=args.age_cutoff)
    train_asthma_table, train_split = build_dataset_asthma_table(
        train_filtered, model_features=model_features, dataset_label="train"
    )
    external_asthma_table, external_split = build_dataset_asthma_table(
        external_filtered, model_features=model_features, dataset_label="external"
    )

    flow_df = pd.DataFrame(
        [
            {
                "dataset": "train",
                "n_raw": train_flow["n_raw"],
                f"n_age_le_{args.age_cutoff:g}": train_flow["n_age_le_cutoff"],
                f"n_record_gt_{args.record_count_min}": train_flow["n_record_gt_min"],
                "n_final": train_flow["n_final"],
                "n_asthma": train_split["n_asthma"],
                "n_non_asthma": train_split["n_non_asthma"],
                "n_outcome_missing": train_split["n_outcome_missing"],
            },
            {
                "dataset": "external",
                "n_raw": external_flow["n_raw"],
                f"n_age_le_{args.age_cutoff:g}": external_flow["n_age_le_cutoff"],
                f"n_record_gt_{args.record_count_min}": np.nan,
                "n_final": external_flow["n_final"],
                "n_asthma": external_split["n_asthma"],
                "n_non_asthma": external_split["n_non_asthma"],
                "n_outcome_missing": external_split["n_outcome_missing"],
            },
        ]
    )

    table_df = build_table(train_filtered, external_filtered, model_features)
    table_df = table_df.rename(
        columns={
            "train_summary": f"train (n={train_flow['n_final']})",
            "external_summary": f"external (n={external_flow['n_final']})",
        }
    )

    excel_path = save_outputs(
        flow_df=flow_df,
        table_df=table_df,
        output_dir=output_dir,
        output_prefix=args.output_prefix,
        extra_sheets={
            "train_asthma_vs_non": train_asthma_table,
            "external_asthma_vs_non": external_asthma_table,
        },
    )

    print(f"train final n: {train_flow['n_final']}")
    print(f"external final n: {external_flow['n_final']}")
    print(f"train asthma/non-asthma: {train_split['n_asthma']}/{train_split['n_non_asthma']}")
    print(f"external asthma/non-asthma: {external_split['n_asthma']}/{external_split['n_non_asthma']}")
    print(f"excel file: {excel_path}")


if __name__ == "__main__":
    main()
