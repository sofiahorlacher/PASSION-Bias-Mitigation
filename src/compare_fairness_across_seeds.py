import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


PRIMARY_GROUP = "fitzpatrick"
REPORTING_GROUPS = ["fitzpatrick", "sex", "ageGroup", "country"]
OPTIONAL_VARIANT_COLUMNS = ["MitigationStrength", "StrengthLabel", "SubgroupLabel"]
FAIRNESS_COLUMNS = {
    "overall_eod_mean_to_overall_mean": "eod_mean_to_overall_mean",
    "overall_eod_mean_to_overall_worst": "eod_mean_to_overall_worst",
    "overall_eor_mean_to_overall_mean": "eor_mean_to_overall_mean",
    "overall_eor_mean_to_overall_worst": "eor_mean_to_overall_worst",
}
SUBGROUP_VALUE_COLUMNS = ["fitzpatrick", "sex", "ageGroup", "country"]


def get_experiment_configs():
    return {
        "exp1": {
            "name": "Standard Split - Conditions",
            "pattern": "experiment_standard_split_conditions__passion__{model}",
        },
        "exp2": {
            "name": "Standard Split - Impetigo",
            "pattern": "experiment_standard_split_impetigo__passion__{model}",
        },
        "exp3": {
            "name": "Center Generalization",
            "pattern": "experiment_center__passion__{model}",
            "supported": False,
            "skip_reason": (
                "Fairness sidecar files are overwritten across center splits and "
                "cannot be compared reliably across seeds."
            ),
        },
        "exp4": {
            "name": "Age Group Generalization",
            "pattern": "experiment_age__passion__{model}",
            "supported": False,
            "skip_reason": (
                "Fairness sidecar files are overwritten across age-group splits and "
                "cannot be compared reliably across seeds."
            ),
        },
        "exp5": {
            "name": "Stratified Validation Split (No Folds)",
            "pattern": (
                "experiment_stratified_validation_split_conditions"
                "__{split}__passion__{model}"
            ),
            "stratified": True,
        },
        "exp6": {
            "name": "Stratified Validation Split (5-Fold CV)",
            "pattern": (
                "experiment_stratified_validation_split_conditions_5folds"
                "__{split}__passion__{model}"
            ),
            "stratified": True,
        },
        "exp7": {
            "name": "Standard Test Evaluation of Models Trained on Stratified Splits",
            "pattern": (
                "experiment_standard_split_conditions"
                "__model_trained_with_{split}__passion__{model}"
            ),
            "stratified": True,
        },
        "exp8": {
            "name": "Plain Color Jitter + Oversampling",
            "pattern": (
                "experiment_stratified_validation_split_conditions_color_jitter_oversampled_{fold_tag}"
                "__{subgroup_label}__strength_{strength_label}__{split}__passion__{model}"
            ),
            "underrepresented_group_columns_options": ["fitzpatrick"],
            "mitigation_strengths": [1 / 3, 2 / 3, 1.0],
            "stratified": True,
        },
        "exp9": {
            "name": "Instance Reweighting",
            "pattern": (
                "experiment_stratified_validation_split_conditions_instance_reweighting_{fold_tag}"
                "__{subgroup_label}__strength_{strength_label}__{split}__passion__{model}"
            ),
            "underrepresented_group_columns_options": [["fitzpatrick"]],
            "mitigation_strengths": [1 / 3, 2 / 3, 1.0],
            "stratified": True,
        },
    }


def get_stratified_splits(seed: int):
    return {
        "Split 1": f"none__seed_{seed}",
        "Split 2": f"conditions_PASSION_impetig__seed_{seed}",
        "Split 3": f"conditions_PASSION_impetig_country__seed_{seed}",
        "Split 4": f"conditions_PASSION_impetig_fitzpatrick__seed_{seed}",
        "Split 5": f"conditions_PASSION_impetig_country_fitzpatrick__seed_{seed}",
        "Split 6": f"conditions_PASSION_impetig_country_fitzpatrick_sex__seed_{seed}",
    }


def build_subgroup_label(group_columns) -> str:
    if isinstance(group_columns, str):
        return group_columns
    return "_".join(group_columns)


def format_strength_label(strength: float) -> str:
    return f"{float(strength):.2f}".replace(".", "p")


def get_variant_group_cols(df: pd.DataFrame) -> List[str]:
    return [col for col in OPTIONAL_VARIANT_COLUMNS if col in df.columns]


def get_patterns_for_experiment(
    exp_config: dict,
    model: str,
    split_str: Optional[str] = None,
) -> List[dict]:
    group_options = exp_config.get("underrepresented_group_columns_options", [None])
    strength_options = exp_config.get("mitigation_strengths", [None])
    patterns = []

    for cols in group_options:
        subgroup_label = build_subgroup_label(cols) if cols is not None else None
        for strength in strength_options:
            pattern_kwargs = {"model": model}
            variant = {}
            if subgroup_label is not None:
                pattern_kwargs["subgroup_label"] = subgroup_label
                variant["SubgroupLabel"] = subgroup_label
            if split_str is not None:
                pattern_kwargs["split"] = split_str
            if "fold_tag" in exp_config:
                pattern_kwargs["fold_tag"] = exp_config["fold_tag"]
            if strength is not None:
                strength = float(strength)
                pattern_kwargs["strength_label"] = format_strength_label(strength)
                variant["MitigationStrength"] = strength
                variant["StrengthLabel"] = pattern_kwargs["strength_label"]

            variant["pattern"] = exp_config["pattern"].format(**pattern_kwargs)
            patterns.append(variant)

    return patterns


def get_experiment_csv(seed_path: Path, pattern: str) -> Optional[Path]:
    direct_match = seed_path / f"{pattern}.csv"
    if direct_match.exists():
        return direct_match

    versioned_matches = sorted(seed_path.glob(f"{pattern}*.csv"))
    if versioned_matches:
        return versioned_matches[-1]
    return None


def get_sidecar_file(
    exp_folder: Path,
    prefix: str,
    pattern: str,
    run_label: str = "Test",
) -> Optional[Path]:
    sidecar_path = exp_folder / f"{prefix}_{pattern}__{run_label}.csv"
    if sidecar_path.exists():
        return sidecar_path
    return None


def discover_sidecar_runs(
    exp_folder: Path,
    prefix: str,
    pattern: str,
    run_mode: str,
) -> List[tuple[str, Path]]:
    results = []
    prefix_str = f"{prefix}_{pattern}__"
    for path in sorted(exp_folder.glob(f"{prefix}_{pattern}__*.csv")):
        run_label = path.stem.removeprefix(prefix_str)
        if "__" in run_label:
            continue
        if run_mode == "test" and run_label != "Test":
            continue
        if run_mode == "folds" and not str(run_label).startswith("Fold-"):
            continue
        results.append((run_label, path))
    return results


def select_runs(df: pd.DataFrame, run_mode: str) -> pd.DataFrame:
    if run_mode == "test":
        return df[df["AdditionalRunInfo"] == "Test"].copy()
    if run_mode == "folds":
        return df[df["AdditionalRunInfo"].astype(str).str.startswith("Fold-")].copy()
    raise ValueError(f"Unsupported run_mode: {run_mode}")


def aggregate_mean_std(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    numeric_cols = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != "Seed"
    ]
    if not numeric_cols:
        return df.copy()

    aggregated = (
        df.groupby(group_cols, dropna=False)[numeric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregated.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in aggregated.columns
    ]

    seed_counts = (
        df.groupby(group_cols, dropna=False)["Seed"]
        .nunique()
        .reset_index(name="num_seeds")
    )
    return aggregated.merge(seed_counts, on=group_cols, how="left")


def collapse_folds_within_seed(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    numeric_cols = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != "Seed"
    ]
    key_cols = ["Seed", *group_cols]
    rows = []

    for keys, group_df in df.groupby(key_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(key_cols, keys))
        for col in numeric_cols:
            row[col] = group_df[col].mean()
            row[f"{col}_fold_std"] = group_df[col].std()
        row["num_folds"] = (
            group_df["RunLabel"].nunique() if "RunLabel" in group_df.columns else len(group_df)
        )
        rows.append(row)

    return pd.DataFrame(rows)


def build_subgroup_value(row: pd.Series) -> str:
    parts = []
    for col in SUBGROUP_VALUE_COLUMNS:
        value = row.get(col)
        if pd.notna(value):
            parts.append(f"{col}={value}")
    return " | ".join(parts)


def collect_performance_for_experiment(
    exp_key: str,
    exp_config: dict,
    seeds: List[int],
    eval_path: Path,
    model: str,
    eval_type: str,
    run_mode: str = "test",
) -> pd.DataFrame:
    rows = []

    for seed in seeds:
        seed_path = eval_path / f"seed_{seed}"
        if not seed_path.exists():
            continue

        split_items = (
            get_stratified_splits(seed).items()
            if exp_config.get("stratified", False)
            else [("Standard", None)]
        )

        for split_name, split_str in split_items:
            for variant in get_patterns_for_experiment(exp_config, model, split_str):
                experiment_csv = get_experiment_csv(seed_path, variant["pattern"])
                if experiment_csv is None:
                    continue

                df = pd.read_csv(experiment_csv)
                selected = df[df["EvalType"] == eval_type].copy()
                selected = select_runs(selected, run_mode)
                if selected.empty:
                    continue

                for _, row in selected.iterrows():
                    result_row = {
                        "Experiment": exp_key,
                        "Split": split_name,
                        "Seed": seed,
                        "RunLabel": row["AdditionalRunInfo"],
                        "score": float(row["Score"]),
                    }
                    for col in OPTIONAL_VARIANT_COLUMNS:
                        if col in variant:
                            result_row[col] = variant[col]
                    if "AUROC" in row.index and pd.notna(row["AUROC"]):
                        result_row["auroc"] = float(row["AUROC"])
                    rows.append(result_row)

    return pd.DataFrame(rows)


def collect_fairness_for_experiment(
    exp_key: str,
    exp_config: dict,
    seeds: List[int],
    eval_path: Path,
    model: str,
    run_mode: str = "test",
) -> pd.DataFrame:
    rows = []

    for seed in seeds:
        seed_path = eval_path / f"seed_{seed}"
        if not seed_path.exists():
            continue

        split_items = (
            get_stratified_splits(seed).items()
            if exp_config.get("stratified", False)
            else [("Standard", None)]
        )

        for split_name, split_str in split_items:
            for variant in get_patterns_for_experiment(exp_config, model, split_str):
                exp_folder = seed_path / variant["pattern"]
                if not exp_folder.exists():
                    continue

                run_files = discover_sidecar_runs(
                    exp_folder,
                    "fairness_metric_results",
                    variant["pattern"],
                    run_mode=run_mode,
                )
                for run_label, fairness_file in run_files:
                    df = pd.read_csv(fairness_file)
                    if df.empty or "GroupBy" not in df.columns:
                        continue

                    df = df[df["GroupBy"].isin(REPORTING_GROUPS)].copy()
                    if df.empty:
                        continue

                    keep_cols = ["GroupBy", *[col for col in FAIRNESS_COLUMNS if col in df.columns]]
                    df = df[keep_cols].copy()
                    df["Experiment"] = exp_key
                    df["Split"] = split_name
                    df["Seed"] = seed
                    df["RunLabel"] = run_label
                    for col in OPTIONAL_VARIANT_COLUMNS:
                        if col in variant:
                            df[col] = variant[col]
                    rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def collect_subgroups_for_experiment(
    exp_key: str,
    exp_config: dict,
    seeds: List[int],
    eval_path: Path,
    model: str,
    run_mode: str = "test",
) -> pd.DataFrame:
    rows = []

    for seed in seeds:
        seed_path = eval_path / f"seed_{seed}"
        if not seed_path.exists():
            continue

        split_items = (
            get_stratified_splits(seed).items()
            if exp_config.get("stratified", False)
            else [("Standard", None)]
        )

        for split_name, split_str in split_items:
            for variant in get_patterns_for_experiment(exp_config, model, split_str):
                exp_folder = seed_path / variant["pattern"]
                if not exp_folder.exists():
                    continue

                run_files = discover_sidecar_runs(
                    exp_folder,
                    "subgroup_metric_results",
                    variant["pattern"],
                    run_mode=run_mode,
                )
                for run_label, subgroup_file in run_files:
                    df = pd.read_csv(subgroup_file)
                    if df.empty or "GroupBy" not in df.columns:
                        continue

                    df = df[df["GroupBy"].isin(REPORTING_GROUPS)].copy()
                    if df.empty:
                        continue

                    keep_cols = [
                        "GroupBy",
                        *[col for col in SUBGROUP_VALUE_COLUMNS if col in df.columns],
                        *[col for col in ["Support", "balancedAcc"] if col in df.columns],
                    ]
                    df = df[keep_cols].copy()
                    df["Experiment"] = exp_key
                    df["Split"] = split_name
                    df["Seed"] = seed
                    df["RunLabel"] = run_label
                    for col in OPTIONAL_VARIANT_COLUMNS:
                        if col in variant:
                            df[col] = variant[col]
                    rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_fairness(fairness_df: pd.DataFrame) -> pd.DataFrame:
    if fairness_df.empty:
        return pd.DataFrame()

    summary = aggregate_mean_std(
        fairness_df,
        group_cols=["Experiment", "Split", "GroupBy", *get_variant_group_cols(fairness_df)],
    )
    return summary.rename(columns=FAIRNESS_COLUMNS)


def summarize_subgroups(subgroup_df: pd.DataFrame) -> pd.DataFrame:
    if subgroup_df.empty:
        return pd.DataFrame()

    group_cols = [
        "Experiment",
        "Split",
        "GroupBy",
        *get_variant_group_cols(subgroup_df),
        *[col for col in SUBGROUP_VALUE_COLUMNS if col in subgroup_df.columns],
    ]
    summary = aggregate_mean_std(subgroup_df, group_cols=group_cols)
    summary["subgroup"] = summary.apply(build_subgroup_value, axis=1)

    ordered_cols = [
        "Experiment",
        "Split",
        "GroupBy",
        *get_variant_group_cols(summary),
        "subgroup",
        *[col for col in SUBGROUP_VALUE_COLUMNS if col in summary.columns],
        *[col for col in summary.columns if col.startswith("Support_")],
        *[col for col in summary.columns if col.startswith("balancedAcc_")],
        "num_seeds",
    ]
    ordered_cols = [col for col in ordered_cols if col in summary.columns]
    return summary[ordered_cols].copy()


def summarize_worst_subgroups(subgroup_summary: pd.DataFrame) -> pd.DataFrame:
    if subgroup_summary.empty:
        return pd.DataFrame()

    if "balancedAcc_mean" not in subgroup_summary.columns:
        return pd.DataFrame()

    group_cols = ["Experiment", "Split", "GroupBy", *get_variant_group_cols(subgroup_summary)]
    rows = []
    for _, group_df in subgroup_summary.groupby(group_cols, dropna=False):
        worst_row = group_df.nsmallest(1, "balancedAcc_mean").iloc[0]
        result_row = {
            "Experiment": worst_row["Experiment"],
            "Split": worst_row["Split"],
            "GroupBy": worst_row["GroupBy"],
            "worst_subgroup": worst_row["subgroup"],
            "worst_balancedAcc_mean": worst_row["balancedAcc_mean"],
            "worst_balancedAcc_std": worst_row.get("balancedAcc_std", np.nan),
            "worst_support_mean": worst_row.get("Support_mean", np.nan),
            "worst_support_std": worst_row.get("Support_std", np.nan),
            "num_seeds": worst_row.get("num_seeds", np.nan),
        }
        for col in get_variant_group_cols(subgroup_summary):
            result_row[col] = worst_row.get(col)
        rows.append(result_row)
    return pd.DataFrame(rows)


def build_primary_comparison_summary(
    performance_summary: pd.DataFrame,
    fairness_summary: pd.DataFrame,
    worst_subgroup_summary: pd.DataFrame,
) -> pd.DataFrame:
    if performance_summary.empty:
        return pd.DataFrame()

    summary = performance_summary.copy()

    if not fairness_summary.empty:
        fitzpatrick_fairness = fairness_summary[
            fairness_summary["GroupBy"] == PRIMARY_GROUP
        ].copy()
        if not fitzpatrick_fairness.empty:
            fairness_merge_cols = [
                "Experiment",
                "Split",
                *[
                    col
                    for col in OPTIONAL_VARIANT_COLUMNS
                    if col in summary.columns and col in fitzpatrick_fairness.columns
                ],
            ]
            fitzpatrick_fairness = fitzpatrick_fairness.drop(columns=["GroupBy"])
            fitzpatrick_fairness = fitzpatrick_fairness.rename(
                columns={
                    "eod_mean_to_overall_mean_mean": "fitzpatrick_eod_mean",
                    "eod_mean_to_overall_mean_std": "fitzpatrick_eod_std",
                    "eod_mean_to_overall_worst_mean": "fitzpatrick_worst_eod_mean",
                    "eod_mean_to_overall_worst_std": "fitzpatrick_worst_eod_std",
                    "eor_mean_to_overall_mean_mean": "fitzpatrick_eor_mean",
                    "eor_mean_to_overall_mean_std": "fitzpatrick_eor_std",
                    "eor_mean_to_overall_worst_mean": "fitzpatrick_worst_eor_mean",
                    "eor_mean_to_overall_worst_std": "fitzpatrick_worst_eor_std",
                }
            )
            summary = summary.merge(
                fitzpatrick_fairness,
                on=fairness_merge_cols,
                how="left",
                suffixes=("", "__fairness"),
            )

    if not worst_subgroup_summary.empty:
        fitzpatrick_worst = worst_subgroup_summary[
            worst_subgroup_summary["GroupBy"] == PRIMARY_GROUP
        ].copy()
        if not fitzpatrick_worst.empty:
            subgroup_merge_cols = [
                "Experiment",
                "Split",
                *[
                    col
                    for col in OPTIONAL_VARIANT_COLUMNS
                    if col in summary.columns and col in fitzpatrick_worst.columns
                ],
            ]
            fitzpatrick_worst = fitzpatrick_worst.drop(columns=["GroupBy"])
            fitzpatrick_worst = fitzpatrick_worst.rename(
                columns={
                    "worst_subgroup": "fitzpatrick_worst_subgroup",
                    "worst_balancedAcc_mean": "fitzpatrick_worst_balancedAcc_mean",
                    "worst_balancedAcc_std": "fitzpatrick_worst_balancedAcc_std",
                    "worst_support_mean": "fitzpatrick_worst_support_mean",
                    "worst_support_std": "fitzpatrick_worst_support_std",
                }
            )
            summary = summary.merge(
                fitzpatrick_worst,
                on=subgroup_merge_cols,
                how="left",
                suffixes=("", "__subgroup"),
            )

    duplicate_cols = [col for col in summary.columns if col.endswith("__fairness") or col.endswith("__subgroup")]
    if duplicate_cols:
        summary = summary.drop(columns=duplicate_cols)
    return summary


def save_dataframe(output_dir: Path, filename: str, df: pd.DataFrame):
    if df.empty:
        print(f"No rows for {filename}")
        return
    path = output_dir / filename
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate PASSION performance and fairness metrics across seeds."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 32],
        help="Seeds to compare.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        help="Specific experiments to analyze, e.g. exp1 exp5 exp8.",
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default="assets/evaluation",
        help="Path to evaluation results.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="imagenet_tiny",
        help="Model name used in the experiment file names.",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        default="finetuning",
        help="EvalType to use for overall performance aggregation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/fairness_comparison",
        help="Output directory for aggregated CSV files.",
    )
    parser.add_argument(
        "--mitigation_n_folds",
        type=int,
        default=5,
        help="Fold count used in mitigation experiment names.",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_configs = get_experiment_configs()
    fold_tag = f"{args.mitigation_n_folds}folds"
    for exp_key in ("exp8", "exp9"):
        if exp_key in exp_configs:
            exp_configs[exp_key]["fold_tag"] = fold_tag
    if args.experiments:
        exp_configs = {k: v for k, v in exp_configs.items() if k in args.experiments}

    print(f"Comparing seeds: {args.seeds}")
    print(f"Using reporting groups: {', '.join(REPORTING_GROUPS)}")

    for exp_key, exp_config in exp_configs.items():
        print(f"\n--- {exp_key.upper()}: {exp_config['name']} ---")

        if not exp_config.get("supported", True):
            print(f"Skipping: {exp_config['skip_reason']}")
            continue

        performance_by_seed = collect_performance_for_experiment(
            exp_key=exp_key,
            exp_config=exp_config,
            seeds=args.seeds,
            eval_path=eval_path,
            model=args.model,
            eval_type=args.eval_type,
            run_mode="test",
        )
        performance_summary = aggregate_mean_std(
            performance_by_seed,
            group_cols=["Experiment", "Split", *get_variant_group_cols(performance_by_seed)],
        )
        performance_folds = collect_performance_for_experiment(
            exp_key=exp_key,
            exp_config=exp_config,
            seeds=args.seeds,
            eval_path=eval_path,
            model=args.model,
            eval_type=args.eval_type,
            run_mode="folds",
        )
        performance_fold_by_seed = collapse_folds_within_seed(
            performance_folds,
            group_cols=["Experiment", "Split", *get_variant_group_cols(performance_folds)],
        )
        performance_fold_summary = aggregate_mean_std(
            performance_fold_by_seed,
            group_cols=["Experiment", "Split", *get_variant_group_cols(performance_fold_by_seed)],
        )

        fairness_by_seed = collect_fairness_for_experiment(
            exp_key=exp_key,
            exp_config=exp_config,
            seeds=args.seeds,
            eval_path=eval_path,
            model=args.model,
            run_mode="test",
        )
        fairness_summary = summarize_fairness(fairness_by_seed)
        fairness_folds = collect_fairness_for_experiment(
            exp_key=exp_key,
            exp_config=exp_config,
            seeds=args.seeds,
            eval_path=eval_path,
            model=args.model,
            run_mode="folds",
        )
        fairness_fold_by_seed = collapse_folds_within_seed(
            fairness_folds,
            group_cols=[
                "Experiment",
                "Split",
                "GroupBy",
                *get_variant_group_cols(fairness_folds),
            ],
        )
        fairness_fold_summary = summarize_fairness(fairness_fold_by_seed)

        subgroup_by_seed = collect_subgroups_for_experiment(
            exp_key=exp_key,
            exp_config=exp_config,
            seeds=args.seeds,
            eval_path=eval_path,
            model=args.model,
            run_mode="test",
        )
        subgroup_summary = summarize_subgroups(subgroup_by_seed)
        worst_subgroup_summary = summarize_worst_subgroups(subgroup_summary)
        subgroup_folds = collect_subgroups_for_experiment(
            exp_key=exp_key,
            exp_config=exp_config,
            seeds=args.seeds,
            eval_path=eval_path,
            model=args.model,
            run_mode="folds",
        )
        subgroup_fold_by_seed = collapse_folds_within_seed(
            subgroup_folds,
            group_cols=[
                "Experiment",
                "Split",
                "GroupBy",
                *get_variant_group_cols(subgroup_folds),
                *[col for col in SUBGROUP_VALUE_COLUMNS if col in subgroup_folds.columns],
            ],
        )
        subgroup_fold_summary = summarize_subgroups(subgroup_fold_by_seed)
        worst_subgroup_fold_summary = summarize_worst_subgroups(subgroup_fold_summary)

        comparison_summary = build_primary_comparison_summary(
            performance_summary=performance_summary,
            fairness_summary=fairness_summary,
            worst_subgroup_summary=worst_subgroup_summary,
        )
        comparison_fold_summary = build_primary_comparison_summary(
            performance_summary=performance_fold_summary,
            fairness_summary=fairness_fold_summary,
            worst_subgroup_summary=worst_subgroup_fold_summary,
        )

        save_dataframe(output_dir, f"{exp_key}_performance_by_seed.csv", performance_by_seed)
        save_dataframe(output_dir, f"{exp_key}_performance_summary.csv", performance_summary)
        save_dataframe(output_dir, f"{exp_key}_fold_performance_raw.csv", performance_folds)
        save_dataframe(output_dir, f"{exp_key}_fold_performance_by_seed.csv", performance_fold_by_seed)
        save_dataframe(output_dir, f"{exp_key}_fold_performance_summary.csv", performance_fold_summary)
        save_dataframe(output_dir, f"{exp_key}_fairness_by_seed.csv", fairness_by_seed)
        save_dataframe(output_dir, f"{exp_key}_fairness_summary.csv", fairness_summary)
        save_dataframe(output_dir, f"{exp_key}_fold_fairness_raw.csv", fairness_folds)
        save_dataframe(output_dir, f"{exp_key}_fold_fairness_by_seed.csv", fairness_fold_by_seed)
        save_dataframe(output_dir, f"{exp_key}_fold_fairness_summary.csv", fairness_fold_summary)
        save_dataframe(output_dir, f"{exp_key}_subgroups_by_seed.csv", subgroup_by_seed)
        save_dataframe(output_dir, f"{exp_key}_subgroups_summary.csv", subgroup_summary)
        save_dataframe(output_dir, f"{exp_key}_fold_subgroups_raw.csv", subgroup_folds)
        save_dataframe(output_dir, f"{exp_key}_fold_subgroups_by_seed.csv", subgroup_fold_by_seed)
        save_dataframe(output_dir, f"{exp_key}_fold_subgroups_summary.csv", subgroup_fold_summary)
        save_dataframe(output_dir, f"{exp_key}_worst_subgroup_summary.csv", worst_subgroup_summary)
        save_dataframe(output_dir, f"{exp_key}_fold_worst_subgroup_summary.csv", worst_subgroup_fold_summary)
        save_dataframe(output_dir, f"{exp_key}_comparison_summary.csv", comparison_summary)
        save_dataframe(output_dir, f"{exp_key}_comparison_fold_summary.csv", comparison_fold_summary)


if __name__ == "__main__":
    main()
