import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse


FITZPATRICK_GROUP = "fitzpatrick"
OPTIONAL_VARIANT_COLUMNS = ["MitigationStrength", "StrengthLabel", "SubgroupLabel"]
REQUIRED_BASE_COLUMNS = [
    "Experiment",
    "Split",
    "auroc_mean",
    "auroc_std",
    "balancedAcc_mean",
    "balancedAcc_std",
    "fitzpatrick_worst_balancedAcc_mean",
    "fitzpatrick_worst_balancedAcc_std",
]
METHOD_STYLES = {
    "exp6": {
        "label": "Baseline (CV)",
        "color": "#7f7f7f",
        "marker": "o",
    },
    "exp8": {
        "label": "Color Jitter + Oversampling",
        "color": "#1f77b4",
        "marker": "s",
    },
    "exp9": {
        "label": "Instance Reweighting",
        "color": "#d62728",
        "marker": "^",
    },
    "exp10": {
        "label": "Group DRO",
        "color": "#2ca02c",
        "marker": "D",
    },
    "exp11": {
        "label": "Hardt Post-Processing",
        "color": "#ff7f0e",
        "marker": "P",
    },
    "exp12": {
        "label": "MIFair (OAE)",
        "color": "#8c564b",
        "marker": "X",
    },
}
MAIN_EXPERIMENT_KEYS = ["exp6", "exp8", "exp9", "exp10", "exp12"]
HARDT_EXPERIMENT_KEYS = ["exp11"]
LEGEND_LOCATION = "lower left"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Skipping missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def format_strength_label(strength: float) -> str:
    if np.isclose(strength, 0.0):
        return "base"
    if np.isclose(strength, 1 / 3, atol=0.02):
        return "low"
    if np.isclose(strength, 2 / 3, atol=0.02):
        return "medium"
    if np.isclose(strength, 1.0):
        return "full"
    return f"{strength:.2f}"


def fill_column_from_aliases(
    df: pd.DataFrame,
    target_col: str,
    aliases: list[str],
) -> pd.DataFrame:
    if target_col in df.columns:
        return df
    for alias in aliases:
        if alias in df.columns:
            df[target_col] = df[alias]
            break
    return df


def get_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def merge_fitzpatrick_fairness_summary(
    summary: pd.DataFrame,
    exp_key: str,
    input_dir: Path,
) -> pd.DataFrame:
    fairness_summary = load_csv(input_dir / f"{exp_key}_fold_fairness_summary.csv")
    if fairness_summary.empty or "GroupBy" not in fairness_summary.columns:
        return summary

    fairness_summary = fairness_summary[
        fairness_summary["GroupBy"] == FITZPATRICK_GROUP
    ].copy()
    if fairness_summary.empty:
        return summary

    fairness_summary = fill_column_from_aliases(
        fairness_summary,
        "fitzpatrick_eod_mean",
        ["eod_mean_to_overall_mean_mean", "overall_eod_mean_to_overall_mean_mean"],
    )
    fairness_summary = fill_column_from_aliases(
        fairness_summary,
        "fitzpatrick_eod_std",
        ["eod_mean_to_overall_mean_std", "overall_eod_mean_to_overall_mean_std"],
    )

    required_cols = ["fitzpatrick_eod_mean", "fitzpatrick_eod_std"]
    if any(col not in fairness_summary.columns for col in required_cols):
        return summary

    merge_cols = ["Experiment", "Split"]
    for col in OPTIONAL_VARIANT_COLUMNS:
        if col in summary.columns and col in fairness_summary.columns:
            merge_cols.append(col)

    fairness_summary = fairness_summary[
        [*merge_cols, *required_cols]
    ].drop_duplicates()
    return summary.merge(
        fairness_summary,
        on=merge_cols,
        how="left",
        suffixes=("", "__fairness"),
    )


def merge_worst_subgroup_fold_std(
    summary: pd.DataFrame,
    exp_key: str,
    input_dir: Path,
) -> pd.DataFrame:
    if "fitzpatrick_worst_subgroup" not in summary.columns:
        return summary

    subgroup_summary = load_csv(input_dir / f"{exp_key}_fold_subgroups_summary.csv")
    if subgroup_summary.empty or "GroupBy" not in subgroup_summary.columns:
        return summary

    subgroup_summary = subgroup_summary[
        subgroup_summary["GroupBy"] == FITZPATRICK_GROUP
    ].copy()
    if subgroup_summary.empty or "subgroup" not in subgroup_summary.columns:
        return summary

    required_cols = ["balancedAcc_fold_std_mean", "balancedAcc_fold_std_std"]
    if any(col not in subgroup_summary.columns for col in required_cols):
        return summary

    merge_cols = ["Experiment", "Split"]
    for col in OPTIONAL_VARIANT_COLUMNS:
        if col in summary.columns and col in subgroup_summary.columns:
            merge_cols.append(col)

    subgroup_summary = subgroup_summary[
        [*merge_cols, "subgroup", *required_cols]
    ].rename(
        columns={
            "subgroup": "fitzpatrick_worst_subgroup",
            "balancedAcc_fold_std_mean": "fitzpatrick_worst_balancedAcc_fold_std_mean",
            "balancedAcc_fold_std_std": "fitzpatrick_worst_balancedAcc_fold_std_std",
        }
    )
    return summary.merge(
        subgroup_summary.drop_duplicates(),
        on=[*merge_cols, "fitzpatrick_worst_subgroup"],
        how="left",
        suffixes=("", "__worst_subgroup"),
    )


def normalize_fitzpatrick_eod_columns(
    summary: pd.DataFrame,
    exp_key: str,
    input_dir: Path,
) -> pd.DataFrame:
    summary = fill_column_from_aliases(
        summary,
        "fitzpatrick_eod_mean",
        ["eod_mean_to_overall_mean_mean", "overall_eod_mean_to_overall_mean_mean"],
    )
    summary = fill_column_from_aliases(
        summary,
        "fitzpatrick_eod_std",
        ["eod_mean_to_overall_mean_std", "overall_eod_mean_to_overall_mean_std"],
    )
    if (
        "fitzpatrick_eod_mean" not in summary.columns
        or "fitzpatrick_eod_std" not in summary.columns
    ):
        summary = merge_fitzpatrick_fairness_summary(summary, exp_key, input_dir)
    return summary


def prepare_hardt_threshold_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "MitigationStrength" not in summary.columns:
        return summary

    summary = summary.copy()
    zero_strength_mask = np.isclose(summary["MitigationStrength"], 0.0)
    if zero_strength_mask.any():
        summary.loc[zero_strength_mask, "StrengthLabel"] = "baseline"
    else:
        print(
            "Warning: no exp11 strength-0.0 row found. "
            "Threshold plots will be generated without the calibration baseline."
        )
    return summary.sort_values("MitigationStrength")


def load_experiment_summary(
    exp_key: str,
    input_dir: Path,
    split_name: str,
) -> pd.DataFrame:
    # Reuse the comparison summary so Pareto plots follow the exact same
    # fold-within-seed collapse and cross-seed aggregation order.
    summary = load_csv(input_dir / f"{exp_key}_comparison_fold_summary.csv")
    if summary.empty:
        return pd.DataFrame()

    summary = summary[summary["Split"] == split_name].copy()
    if summary.empty:
        return pd.DataFrame()

    summary = normalize_fitzpatrick_eod_columns(summary, exp_key, input_dir)
    summary = merge_worst_subgroup_fold_std(summary, exp_key, input_dir)

    missing_cols = [col for col in REQUIRED_BASE_COLUMNS if col not in summary.columns]
    if missing_cols:
        print(
            f"Skipping {exp_key} because required columns are missing. "
            f"summary={missing_cols}"
        )
        return pd.DataFrame()

    if "MitigationStrength" not in summary.columns:
        summary["MitigationStrength"] = 0.0
    summary["MitigationStrength"] = summary["MitigationStrength"].astype(float)
    summary["StrengthLabel"] = summary["MitigationStrength"].apply(format_strength_label)

    style = METHOD_STYLES.get(
        exp_key,
        {"label": exp_key, "color": "#333333", "marker": "o"},
    )
    summary["ExperimentKey"] = exp_key
    summary["MethodLabel"] = style["label"]
    summary["Color"] = style["color"]
    summary["Marker"] = style["marker"]
    return summary.sort_values(["ExperimentKey", "MitigationStrength"])


def build_plot_summary(
    comparison_summary: pd.DataFrame,
    fairness_mean_col: str,
    fairness_std_cols: list[str],
    invert_fairness: bool = False,
) -> pd.DataFrame:
    if comparison_summary.empty:
        return pd.DataFrame()
    fairness_std_col = get_first_existing_column(comparison_summary, fairness_std_cols)
    required_cols = [
        "ExperimentKey",
        "MethodLabel",
        "Color",
        "Marker",
        "MitigationStrength",
        "StrengthLabel",
        fairness_mean_col,
    ]
    if fairness_std_col is not None:
        required_cols.append(fairness_std_col)
    missing_cols = [col for col in required_cols if col not in comparison_summary.columns]
    if missing_cols or fairness_std_col is None:
        if fairness_std_col is None:
            missing_cols = [*missing_cols, f"one of {fairness_std_cols}"]
        print(
            "Skipping plot summary because required columns are missing: "
            f"{missing_cols}"
        )
        return pd.DataFrame()

    summary = comparison_summary.copy()
    summary["fairness_mean"] = summary[fairness_mean_col]
    if invert_fairness:
        summary["fairness_mean"] = 1.0 - summary["fairness_mean"]
    summary["fairness_std"] = summary[fairness_std_col]
    return summary


def get_std_column_name(mean_col: str) -> str:
    if mean_col.endswith("_mean"):
        return mean_col[:-5] + "_std"
    return f"{mean_col}_std"


def compute_pareto_frontier(
    summary: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    summary = summary[summary[x_col].notna() & summary[y_col].notna()].copy()
    if summary.empty:
        return pd.DataFrame()

    rows = []
    for idx, row in summary.iterrows():
        dominated = False
        for other_idx, other in summary.iterrows():
            if idx == other_idx:
                continue
            if (
                other[x_col] >= row[x_col]
                and other[y_col] >= row[y_col]
                and (
                    other[x_col] > row[x_col]
                    or other[y_col] > row[y_col]
                )
            ):
                dominated = True
                break
        if not dominated:
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(x_col)


def print_tradeoff_values(
    summary: pd.DataFrame,
    x_col: str,
    y_col: str,
    y_std_col: str | None,
    title: str,
    split_name: str,
) -> None:
    print(f"\n=== {title} ({split_name}) ===")

    if summary.empty:
        print("No CV summary points available.")
        return

    y_std_col = y_std_col or get_std_column_name(y_col)
    cv_cols = ["MethodLabel", "StrengthLabel", x_col, "fairness_std", y_col, y_std_col]
    cv_table = summary[cv_cols].copy().rename(
        columns={
            "MethodLabel": "method",
            "StrengthLabel": "strength",
            x_col: "x_value",
            "fairness_std": "x_std",
            y_col: "y_value",
            y_std_col: "y_std",
        }
    )
    print("CV mean points:")
    print(cv_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    frontier = compute_pareto_frontier(summary, x_col=x_col, y_col=y_col)
    if frontier.empty:
        print("\nPareto frontier: none")
    else:
        frontier_table = frontier[cv_cols].copy().rename(
            columns={
                "MethodLabel": "method",
                "StrengthLabel": "strength",
                x_col: "x_value",
                "fairness_std": "x_std",
                y_col: "y_value",
                y_std_col: "y_std",
            }
        )
        print("\nPareto frontier points:")
        print(frontier_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def plot_tradeoff(
    summary: pd.DataFrame,
    experiment_keys: list[str],
    x_col: str,
    x_label: str,
    y_col: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    if summary.empty:
        print(f"No data available for {title}.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for exp_key in experiment_keys:
        method_summary = summary[summary["ExperimentKey"] == exp_key].copy()
        method_summary = method_summary[
            method_summary[x_col].notna() & method_summary[y_col].notna()
        ]
        if method_summary.empty:
            continue
        color = method_summary["Color"].iloc[0]
        marker = method_summary["Marker"].iloc[0]
        label = method_summary["MethodLabel"].iloc[0]

        ax.scatter(
            method_summary[x_col],
            method_summary[y_col],
            color=color,
            marker=marker,
            s=90,
            label=label,
        )

        for _, row in method_summary.iterrows():
            ax.annotate(
                row["StrengthLabel"],
                (row[x_col], row[y_col]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
                color=color,
            )

    frontier = compute_pareto_frontier(summary, x_col=x_col, y_col=y_col)
    if not frontier.empty:
        if len(frontier) == 1:
            ax.plot(
                [],
                [],
                color="black",
                linewidth=2.0,
                linestyle="--",
                label="Pareto Frontier",
            )
            x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
            y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
            ax.add_patch(
                Ellipse(
                    (frontier[x_col].iloc[0], frontier[y_col].iloc[0]),
                    width=0.035 * x_span,
                    height=0.05 * y_span,
                    fill=False,
                    edgecolor="black",
                    linewidth=2.2,
                    linestyle="--",
                    zorder=5,
                )
            )
        else:
            ax.plot(
                frontier[x_col],
                frontier[y_col],
                color="black",
                linewidth=2.0,
                linestyle="--",
                label="Pareto Frontier",
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc=LEGEND_LOCATION, frameon=False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path.with_suffix('.png')}")


def plot_tradeoff_with_std(
    summary: pd.DataFrame,
    experiment_keys: list[str],
    x_col: str,
    x_label: str,
    y_col: str,
    y_std_col: str | None,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    if summary.empty:
        print(f"No data available for {title} (with std).")
        return

    y_std_col = y_std_col or get_std_column_name(y_col)
    fig, ax = plt.subplots(figsize=(8, 6))

    for exp_key in experiment_keys:
        method_summary = summary[summary["ExperimentKey"] == exp_key].copy()
        method_summary = method_summary[
            method_summary[x_col].notna() & method_summary[y_col].notna()
        ]
        if method_summary.empty:
            continue

        color = method_summary["Color"].iloc[0]
        marker = method_summary["Marker"].iloc[0]
        label = method_summary["MethodLabel"].iloc[0]

        ax.scatter(
            method_summary[x_col],
            method_summary[y_col],
            color=color,
            marker=marker,
            s=90,
            label=label,
            zorder=2,
        )

        ax.errorbar(
            method_summary[x_col],
            method_summary[y_col],
            xerr=method_summary["fairness_std"].fillna(0.0),
            yerr=method_summary[y_std_col].fillna(0.0),
            fmt="none",
            ecolor=color,
            elinewidth=1.2,
            capsize=3,
            capthick=1.2,
            alpha=0.55,
            zorder=3,
        )

        for _, row in method_summary.iterrows():
            ax.annotate(
                row["StrengthLabel"],
                (row[x_col], row[y_col]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
                color=color,
            )

    frontier = compute_pareto_frontier(summary, x_col=x_col, y_col=y_col)
    if not frontier.empty:
        if len(frontier) == 1:
            ax.plot(
                [],
                [],
                color="black",
                linewidth=2.0,
                linestyle="--",
                label="Pareto Frontier",
            )
            x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
            y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
            ax.add_patch(
                Ellipse(
                    (frontier[x_col].iloc[0], frontier[y_col].iloc[0]),
                    width=0.035 * x_span,
                    height=0.05 * y_span,
                    fill=False,
                    edgecolor="black",
                    linewidth=2.2,
                    linestyle="--",
                    zorder=5,
                )
            )
        else:
            ax.plot(
                frontier[x_col],
                frontier[y_col],
                color="black",
                linewidth=2.0,
                linestyle="--",
                label="Pareto Frontier",
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title} with CV Std")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc=LEGEND_LOCATION, frameon=False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved std plot: {output_path.with_suffix('.png')}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Pareto-style AUROC/fairness trade-off figures from aggregated seed results."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="results/fairness_comparison",
        help="Directory containing the compare_fairness_across_seeds CSV outputs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pareto_plots",
        help="Directory for the generated Pareto plots.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="Split 1",
        help="Split label to visualize, e.g. 'Split 1'.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    all_summaries = []
    for exp_key in MAIN_EXPERIMENT_KEYS + HARDT_EXPERIMENT_KEYS:
        exp_summary = load_experiment_summary(
            exp_key,
            input_dir,
            args.split,
        )
        if not exp_summary.empty:
            all_summaries.append(exp_summary)

    if not all_summaries:
        print(
            "No experiment summary rows were found. "
            "Did you run compare_fairness_across_seeds first?"
        )
        return

    comparison_summary = pd.concat(all_summaries, ignore_index=True)
    summary_worst = build_plot_summary(
        comparison_summary=comparison_summary,
        fairness_mean_col="fitzpatrick_worst_balancedAcc_mean",
        fairness_std_cols=[
            "fitzpatrick_worst_balancedAcc_fold_std_mean",
            "fitzpatrick_worst_balancedAcc_std",
        ],
    )
    summary_eod = build_plot_summary(
        comparison_summary=comparison_summary,
        fairness_mean_col="fitzpatrick_eod_mean",
        fairness_std_cols=[
            "fitzpatrick_eod_fold_std_mean",
            "overall_eod_mean_to_overall_mean_fold_std_mean",
            "eod_mean_to_overall_mean_fold_std_mean",
            "fitzpatrick_eod_std",
        ],
        invert_fairness=True,
    )
    balancedacc_std_col = get_first_existing_column(
        summary_eod,
        ["balancedAcc_fold_std_mean", "balancedAcc_std"],
    )
    auroc_std_col = get_first_existing_column(
        comparison_summary,
        ["auroc_fold_std_mean", "auroc_std"],
    )
    plot_tradeoff(
        summary=summary_eod,
        experiment_keys=MAIN_EXPERIMENT_KEYS,
        x_col="fairness_mean",
        x_label="Fairness Score (1 - Fitzpatrick EOD)",
        y_col="balancedAcc_mean",
        y_label="Balanced Accuracy",
        title="Balanced Accuracy vs Fitzpatrick Fairness Score",
        output_path=output_dir / "pareto_balanced_accuracy_vs_eod",
    )
    plot_tradeoff_with_std(
        summary=summary_eod,
        experiment_keys=MAIN_EXPERIMENT_KEYS,
        x_col="fairness_mean",
        x_label="Fairness Score (1 - Fitzpatrick EOD)",
        y_col="balancedAcc_mean",
        y_std_col=balancedacc_std_col,
        y_label="Balanced Accuracy",
        title="Balanced Accuracy vs Fitzpatrick Fairness Score",
        output_path=output_dir / "pareto_balanced_accuracy_vs_eod_with_std",
    )
    print_tradeoff_values(
        summary=summary_eod,
        x_col="fairness_mean",
        y_col="balancedAcc_mean",
        y_std_col=balancedacc_std_col,
        title="Balanced Accuracy vs Fitzpatrick Fairness Score",
        split_name=args.split,
    )
    plot_tradeoff(
        summary=summary_worst,
        experiment_keys=MAIN_EXPERIMENT_KEYS,
        x_col="fairness_mean",
        x_label="Worst Fitzpatrick Subgroup Balanced Accuracy",
        y_col="auroc_mean",
        y_label="AUROC",
        title="AUROC vs Worst Fitzpatrick Subgroup Balanced Accuracy",
        output_path=output_dir / "pareto_auroc_vs_worst_subgroup_balanced_accuracy",
    )
    plot_tradeoff_with_std(
        summary=summary_worst,
        experiment_keys=MAIN_EXPERIMENT_KEYS,
        x_col="fairness_mean",
        x_label="Worst Fitzpatrick Subgroup Balanced Accuracy",
        y_col="auroc_mean",
        y_std_col=auroc_std_col,
        y_label="AUROC",
        title="AUROC vs Worst Fitzpatrick Subgroup Balanced Accuracy",
        output_path=output_dir / "pareto_auroc_vs_worst_subgroup_balanced_accuracy_with_std",
    )
    print_tradeoff_values(
        summary=summary_worst,
        x_col="fairness_mean",
        y_col="auroc_mean",
        y_std_col=auroc_std_col,
        title="AUROC vs Worst Fitzpatrick Subgroup Balanced Accuracy",
        split_name=args.split,
    )
    plot_tradeoff(
        summary=summary_eod,
        experiment_keys=MAIN_EXPERIMENT_KEYS,
        x_col="fairness_mean",
        x_label="Fairness Score (1 - Fitzpatrick EOD)",
        y_col="auroc_mean",
        y_label="AUROC",
        title="AUROC vs Fitzpatrick Fairness Score",
        output_path=output_dir / "pareto_auroc_vs_eod",
    )
    plot_tradeoff_with_std(
        summary=summary_eod,
        experiment_keys=MAIN_EXPERIMENT_KEYS,
        x_col="fairness_mean",
        x_label="Fairness Score (1 - Fitzpatrick EOD)",
        y_col="auroc_mean",
        y_std_col=auroc_std_col,
        y_label="AUROC",
        title="AUROC vs Fitzpatrick Fairness Score",
        output_path=output_dir / "pareto_auroc_vs_eod_with_std",
    )
    print_tradeoff_values(
        summary=summary_eod,
        x_col="fairness_mean",
        y_col="auroc_mean",
        y_std_col=auroc_std_col,
        title="AUROC vs Fitzpatrick Fairness Score",
        split_name=args.split,
    )

    hardt_summary = comparison_summary[
        comparison_summary["ExperimentKey"].isin(HARDT_EXPERIMENT_KEYS)
    ].copy()
    if hardt_summary.empty:
        print("No Hardt post-processing points found for separate plots.")
        return
    hardt_summary = prepare_hardt_threshold_summary(hardt_summary)

    hardt_summary_worst = build_plot_summary(
        comparison_summary=hardt_summary,
        fairness_mean_col="fitzpatrick_worst_balancedAcc_mean",
        fairness_std_cols=[
            "fitzpatrick_worst_balancedAcc_fold_std_mean",
            "fitzpatrick_worst_balancedAcc_std",
        ],
    )
    hardt_summary_eod = build_plot_summary(
        comparison_summary=hardt_summary,
        fairness_mean_col="fitzpatrick_eod_mean",
        fairness_std_cols=[
            "fitzpatrick_eod_fold_std_mean",
            "overall_eod_mean_to_overall_mean_fold_std_mean",
            "eod_mean_to_overall_mean_fold_std_mean",
            "fitzpatrick_eod_std",
        ],
        invert_fairness=True,
    )
    hardt_balancedacc_std_col = get_first_existing_column(
        hardt_summary_eod,
        ["balancedAcc_fold_std_mean", "balancedAcc_std"],
    )
    hardt_auroc_std_col = get_first_existing_column(
        hardt_summary,
        ["auroc_fold_std_mean", "auroc_std"],
    )

    hardt_output_dir = output_dir / "hardt_postprocessing"
    plot_tradeoff(
        summary=hardt_summary_eod,
        experiment_keys=HARDT_EXPERIMENT_KEYS,
        x_col="fairness_mean",
        x_label="Fairness Score (1 - Fitzpatrick EOD)",
        y_col="balancedAcc_mean",
        y_label="Balanced Accuracy",
        title="Hardt Post-Processing: Balanced Accuracy vs Fitzpatrick Fairness Score",
        output_path=hardt_output_dir / "pareto_balanced_accuracy_vs_eod",
    )
    plot_tradeoff_with_std(
        summary=hardt_summary_eod,
        experiment_keys=HARDT_EXPERIMENT_KEYS,
        x_col="fairness_mean",
        x_label="Fairness Score (1 - Fitzpatrick EOD)",
        y_col="balancedAcc_mean",
        y_std_col=hardt_balancedacc_std_col,
        y_label="Balanced Accuracy",
        title="Hardt Post-Processing: Balanced Accuracy vs Fitzpatrick Fairness Score",
        output_path=hardt_output_dir / "pareto_balanced_accuracy_vs_eod_with_std",
    )
    plot_tradeoff(
        summary=hardt_summary_worst,
        experiment_keys=HARDT_EXPERIMENT_KEYS,
        x_col="fairness_mean",
        x_label="Worst Fitzpatrick Subgroup Balanced Accuracy",
        y_col="auroc_mean",
        y_label="AUROC",
        title="Hardt Post-Processing: AUROC vs Worst Fitzpatrick Subgroup Balanced Accuracy",
        output_path=hardt_output_dir / "pareto_auroc_vs_worst_subgroup_balanced_accuracy",
    )
    plot_tradeoff_with_std(
        summary=hardt_summary_worst,
        experiment_keys=HARDT_EXPERIMENT_KEYS,
        x_col="fairness_mean",
        x_label="Worst Fitzpatrick Subgroup Balanced Accuracy",
        y_col="auroc_mean",
        y_std_col=hardt_auroc_std_col,
        y_label="AUROC",
        title="Hardt Post-Processing: AUROC vs Worst Fitzpatrick Subgroup Balanced Accuracy",
        output_path=hardt_output_dir / "pareto_auroc_vs_worst_subgroup_balanced_accuracy_with_std",
    )
    plot_tradeoff(
        summary=hardt_summary_eod,
        experiment_keys=HARDT_EXPERIMENT_KEYS,
        x_col="fairness_mean",
        x_label="Fairness Score (1 - Fitzpatrick EOD)",
        y_col="auroc_mean",
        y_label="AUROC",
        title="Hardt Post-Processing: AUROC vs Fitzpatrick Fairness Score",
        output_path=hardt_output_dir / "pareto_auroc_vs_eod",
    )
    plot_tradeoff_with_std(
        summary=hardt_summary_eod,
        experiment_keys=HARDT_EXPERIMENT_KEYS,
        x_col="fairness_mean",
        x_label="Fairness Score (1 - Fitzpatrick EOD)",
        y_col="auroc_mean",
        y_std_col=hardt_auroc_std_col,
        y_label="AUROC",
        title="Hardt Post-Processing: AUROC vs Fitzpatrick Fairness Score",
        output_path=hardt_output_dir / "pareto_auroc_vs_eod_with_std",
    )


if __name__ == "__main__":
    main()
