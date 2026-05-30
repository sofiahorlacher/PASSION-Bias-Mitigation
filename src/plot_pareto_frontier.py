import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse


FITZPATRICK_GROUP = "fitzpatrick"
OPTIONAL_VARIANT_COLUMNS = ["MitigationStrength", "StrengthLabel", "SubgroupLabel"]
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
}


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


def get_join_columns(*dfs: pd.DataFrame) -> list[str]:
    join_cols = ["Experiment", "Split", "Seed"]
    if all("RunLabel" in df.columns for df in dfs):
        join_cols.append("RunLabel")
    for col in OPTIONAL_VARIANT_COLUMNS:
        if all(col in df.columns for df in dfs):
            join_cols.append(col)
    return join_cols


def load_experiment_points(
    exp_key: str,
    input_dir: Path,
    split_name: str,
) -> pd.DataFrame:
    perf = load_csv(input_dir / f"{exp_key}_fold_performance_raw.csv")
    fairness = load_csv(input_dir / f"{exp_key}_fold_fairness_raw.csv")
    subgroups = load_csv(input_dir / f"{exp_key}_fold_subgroups_raw.csv")
    if perf.empty or fairness.empty or subgroups.empty:
        return pd.DataFrame()

    perf = perf[perf["Split"] == split_name].copy()
    fairness = fairness[fairness["Split"] == split_name].copy()
    subgroups = subgroups[subgroups["Split"] == split_name].copy()
    if perf.empty or fairness.empty or subgroups.empty:
        return pd.DataFrame()

    fairness = fairness[fairness["GroupBy"] == FITZPATRICK_GROUP].copy()
    subgroups = subgroups[subgroups["GroupBy"] == FITZPATRICK_GROUP].copy()
    if perf.empty or fairness.empty or subgroups.empty:
        return pd.DataFrame()

    join_cols = get_join_columns(perf, fairness, subgroups)
    required_perf_cols = [*join_cols, "auroc"]
    required_fairness_cols = [*join_cols, "overall_eod_mean_to_overall_mean"]
    required_subgroup_cols = [*join_cols, "balancedAcc"]
    missing_perf = [col for col in required_perf_cols if col not in perf.columns]
    missing_fairness = [col for col in required_fairness_cols if col not in fairness.columns]
    missing_subgroups = [col for col in required_subgroup_cols if col not in subgroups.columns]
    if missing_perf or missing_fairness or missing_subgroups:
        print(
            f"Skipping {exp_key} because required columns are missing. "
            f"perf={missing_perf}, fairness={missing_fairness}, subgroups={missing_subgroups}"
        )
        return pd.DataFrame()

    worst_subgroup = (
        subgroups.groupby(join_cols, dropna=False)["balancedAcc"]
        .min()
        .reset_index(name="worst_subgroup_balancedAcc")
    )
    points = perf[required_perf_cols].merge(
        fairness[required_fairness_cols].rename(
            columns={"overall_eod_mean_to_overall_mean": "fitzpatrick_eod"}
        ),
        on=join_cols,
        how="inner",
    )
    points = points.merge(worst_subgroup, on=join_cols, how="inner")
    if points.empty:
        return points

    if "MitigationStrength" not in points.columns:
        points["MitigationStrength"] = 0.0
    points["MitigationStrength"] = points["MitigationStrength"].astype(float)
    if "StrengthLabel" not in points.columns:
        points["StrengthLabel"] = points["MitigationStrength"].apply(format_strength_label)
    else:
        points["StrengthLabel"] = points["MitigationStrength"].apply(format_strength_label)

    style = METHOD_STYLES.get(
        exp_key,
        {"label": exp_key, "color": "#333333", "marker": "o"},
    )
    points["ExperimentKey"] = exp_key
    points["MethodLabel"] = style["label"]
    points["Color"] = style["color"]
    points["Marker"] = style["marker"]
    return points


def summarize_points(points: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame()

    group_cols = [
        "ExperimentKey",
        "MethodLabel",
        "Color",
        "Marker",
        "MitigationStrength",
        "StrengthLabel",
    ]
    return (
        points.groupby(group_cols, dropna=False)
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            metric_mean=(metric_col, "mean"),
            metric_std=(metric_col, "std"),
        )
        .reset_index()
        .sort_values(["ExperimentKey", "MitigationStrength"])
    )


def compute_pareto_frontier(
    summary: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
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
    title: str,
    split_name: str,
) -> None:
    print(f"\n=== {title} ({split_name}) ===")

    if summary.empty:
        print("No CV summary points available.")
        return

    cv_cols = ["MethodLabel", "StrengthLabel", x_col, "fairness_std", y_col, "auroc_std"]
    cv_table = summary[cv_cols].copy().rename(
        columns={
            "MethodLabel": "method",
            "StrengthLabel": "strength",
            x_col: "x_value",
            "fairness_std": "x_std",
            y_col: "auroc",
            "auroc_std": "auroc_std",
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
                y_col: "auroc",
                "auroc_std": "auroc_std",
            }
        )
        print("\nPareto frontier points:")
        print(frontier_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def plot_tradeoff(
    summary: pd.DataFrame,
    x_col: str,
    x_label: str,
    y_col: str,
    y_label: str,
    title: str,
    split_name: str,
    output_path: Path,
) -> None:
    if summary.empty:
        print(f"No data available for {title}.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for exp_key in ["exp6", "exp8", "exp9", "exp10"]:
        method_summary = summary[summary["ExperimentKey"] == exp_key].copy()
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
    ax.set_title(f"{title} ({split_name})")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path.with_suffix('.png')}")


def plot_tradeoff_with_std(
    summary: pd.DataFrame,
    x_col: str,
    x_label: str,
    y_col: str,
    y_label: str,
    title: str,
    split_name: str,
    output_path: Path,
) -> None:
    if summary.empty:
        print(f"No data available for {title} (with std).")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for exp_key in ["exp6", "exp8", "exp9", "exp10"]:
        method_summary = summary[summary["ExperimentKey"] == exp_key].copy()
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
            yerr=method_summary["auroc_std"].fillna(0.0),
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
    ax.set_title(f"{title} with CV Std ({split_name})")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
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

    all_points = []
    for exp_key in ["exp6", "exp8", "exp9", "exp10"]:
        exp_points = load_experiment_points(
            exp_key,
            input_dir,
            args.split,
        )
        if not exp_points.empty:
            all_points.append(exp_points)

    if not all_points:
        print("No experiment points were found. Did you run compare_fairness_across_seeds first?")
        return

    points = pd.concat(all_points, ignore_index=True)
    summary_worst = summarize_points(
        points=points,
        metric_col="worst_subgroup_balancedAcc",
    ).rename(
        columns={
            "metric_mean": "fairness_mean",
            "metric_std": "fairness_std",
            "auroc_mean": "auroc_mean",
            "auroc_std": "auroc_std",
        }
    )
    summary_eod = summarize_points(
        points=points.assign(eod_fairness_score=1.0 - points["fitzpatrick_eod"]),
        metric_col="eod_fairness_score",
    ).rename(
        columns={
            "metric_mean": "fairness_mean",
            "metric_std": "fairness_std",
            "auroc_mean": "auroc_mean",
            "auroc_std": "auroc_std",
        }
    )
    plot_tradeoff(
        summary=summary_worst,
        x_col="fairness_mean",
        x_label="Worst Fitzpatrick Subgroup Balanced Accuracy",
        y_col="auroc_mean",
        y_label="AUROC",
        title="AUROC vs Worst Fitzpatrick Subgroup Balanced Accuracy",
        split_name=args.split,
        output_path=output_dir / "pareto_auroc_vs_worst_subgroup_balanced_accuracy",
    )
    plot_tradeoff_with_std(
        summary=summary_worst,
        x_col="fairness_mean",
        x_label="Worst Fitzpatrick Subgroup Balanced Accuracy",
        y_col="auroc_mean",
        y_label="AUROC",
        title="AUROC vs Worst Fitzpatrick Subgroup Balanced Accuracy",
        split_name=args.split,
        output_path=output_dir / "pareto_auroc_vs_worst_subgroup_balanced_accuracy_with_std",
    )
    print_tradeoff_values(
        summary=summary_worst,
        x_col="fairness_mean",
        y_col="auroc_mean",
        title="AUROC vs Worst Fitzpatrick Subgroup Balanced Accuracy",
        split_name=args.split,
    )
    plot_tradeoff(
        summary=summary_eod,
        x_col="fairness_mean",
        x_label="Fairness Score (1 - Fitzpatrick EOD)",
        y_col="auroc_mean",
        y_label="AUROC",
        title="AUROC vs Fitzpatrick Fairness Score",
        split_name=args.split,
        output_path=output_dir / "pareto_auroc_vs_eod",
    )
    plot_tradeoff_with_std(
        summary=summary_eod,
        x_col="fairness_mean",
        x_label="Fairness Score (1 - Fitzpatrick EOD)",
        y_col="auroc_mean",
        y_label="AUROC",
        title="AUROC vs Fitzpatrick Fairness Score",
        split_name=args.split,
        output_path=output_dir / "pareto_auroc_vs_eod_with_std",
    )
    print_tradeoff_values(
        summary=summary_eod,
        x_col="fairness_mean",
        y_col="auroc_mean",
        title="AUROC vs Fitzpatrick Fairness Score",
        split_name=args.split,
    )


if __name__ == "__main__":
    main()
