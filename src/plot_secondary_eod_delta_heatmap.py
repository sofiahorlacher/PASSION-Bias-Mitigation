import argparse
import os
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from compare_fairness_across_seeds import (
    OPTIONAL_VARIANT_COLUMNS,
    collapse_folds_within_seed,
    collect_fairness_for_experiment,
    get_experiment_configs,
    mitigation_fold_tag,
    summarize_fairness,
)


SECONDARY_GROUPS = {
    "sex": "Sex",
    "ageGroup": "Age",
    "country": "Country",
}
DEFAULT_EXPERIMENTS = ["exp6", "exp8", "exp9", "exp10", "exp12"]
METHOD_LABELS = {
    "exp6": "Baseline",
    "exp8": "Augmentation",
    "exp9": "Reweighing",
    "exp10": "Group DRO",
    "exp11": "Threshold adjustment",
    "exp12": "Fairness-Regularized Loss",
}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def find_csv(input_dir: Path, filename: str) -> Path | None:
    direct_path = input_dir / filename
    if direct_path.exists():
        return direct_path

    matches = sorted(input_dir.rglob(filename))
    if matches:
        return matches[0]
    return None


def format_strength_label(strength: float) -> str:
    return f"{float(strength):.2f}"


def configure_experiments(mitigation_n_folds: int) -> dict:
    exp_configs = get_experiment_configs()
    raw_fold_tag = f"{mitigation_n_folds}folds"
    dro_fold_tag = mitigation_fold_tag(mitigation_n_folds)

    for exp_key in ("exp8", "exp9"):
        if exp_key in exp_configs:
            exp_configs[exp_key]["fold_tag"] = raw_fold_tag
    for exp_key in ("exp10", "exp11", "exp12"):
        if exp_key in exp_configs:
            exp_configs[exp_key]["fold_tag"] = dro_fold_tag

    return exp_configs


def load_fold_fairness_summary(
    exp_key: str,
    exp_config: dict,
    seeds: list[int],
    eval_path: Path,
    model: str,
) -> pd.DataFrame:
    fairness_folds = collect_fairness_for_experiment(
        exp_key=exp_key,
        exp_config=exp_config,
        seeds=seeds,
        eval_path=eval_path,
        model=model,
        run_mode="folds",
    )
    if fairness_folds.empty:
        return pd.DataFrame()

    fairness_fold_by_seed = collapse_folds_within_seed(
        fairness_folds,
        group_cols=[
            "Experiment",
            "Split",
            "GroupBy",
            *[col for col in OPTIONAL_VARIANT_COLUMNS if col in fairness_folds.columns],
        ],
    )
    summary = summarize_fairness(fairness_fold_by_seed)
    if summary.empty:
        return summary

    if "MitigationStrength" not in summary.columns:
        summary["MitigationStrength"] = 0.0
    summary["MitigationStrength"] = summary["MitigationStrength"].astype(float)
    summary["StrengthLabel"] = summary["MitigationStrength"].apply(format_strength_label)
    summary["MethodLabel"] = METHOD_LABELS.get(exp_key, exp_key)
    return summary


def load_aggregated_fold_fairness_summary(
    exp_key: str,
    input_dir: Path,
) -> pd.DataFrame:
    summary_path = find_csv(input_dir, f"{exp_key}_fold_fairness_summary.csv")
    if summary_path is None:
        return pd.DataFrame()

    summary = load_csv(summary_path)
    if summary.empty:
        return pd.DataFrame()

    if "MitigationStrength" not in summary.columns:
        summary["MitigationStrength"] = 0.0
    summary["MitigationStrength"] = summary["MitigationStrength"].fillna(0.0).astype(float)
    summary["StrengthLabel"] = summary["MitigationStrength"].apply(format_strength_label)
    summary["MethodLabel"] = METHOD_LABELS.get(exp_key, exp_key)
    print(f"Loaded aggregated fairness summary: {summary_path}")
    return summary


def build_delta_table(
    fairness_summary: pd.DataFrame,
    split_name: str,
    baseline_experiment: str,
) -> pd.DataFrame:
    if fairness_summary.empty:
        return pd.DataFrame()

    filtered = fairness_summary[
        fairness_summary["Split"].eq(split_name)
        & fairness_summary["GroupBy"].isin(SECONDARY_GROUPS.keys())
    ].copy()
    if filtered.empty:
        return pd.DataFrame()

    metric_candidates = [
        "eod_mean_to_overall_mean_mean",
        "overall_eod_mean_to_overall_mean_mean",
        "eod_mean_to_overall_mean",
        "overall_eod_mean_to_overall_mean",
    ]
    metric_col = next((col for col in metric_candidates if col in filtered.columns), None)
    if metric_col is None:
        raise ValueError(
            "Required EOD column not found in summary. Tried: "
            + ", ".join(metric_candidates)
        )

    baseline = filtered[filtered["Experiment"].eq(baseline_experiment)].copy()
    if baseline.empty:
        raise ValueError(
            f"No baseline rows found for experiment '{baseline_experiment}' and split '{split_name}'."
        )

    baseline = baseline[["Split", "GroupBy", metric_col]].rename(
        columns={metric_col: "baseline_eod"}
    )

    mitigations = filtered[~filtered["Experiment"].eq(baseline_experiment)].copy()
    if mitigations.empty:
        return pd.DataFrame()

    merged = mitigations.merge(baseline, on=["Split", "GroupBy"], how="left")
    merged["delta_eod"] = merged[metric_col] - merged["baseline_eod"]
    merged["GroupLabel"] = merged["GroupBy"].map(SECONDARY_GROUPS)
    merged["RowLabel"] = merged["MethodLabel"] + " " + merged["StrengthLabel"]

    row_order = (
        merged[["Experiment", "MethodLabel", "MitigationStrength", "RowLabel"]]
        .drop_duplicates()
        .sort_values(["Experiment", "MitigationStrength"])
    )

    pivot = merged.pivot_table(
        index="RowLabel",
        columns="GroupLabel",
        values="delta_eod",
        aggfunc="first",
    )
    pivot = pivot.reindex(index=row_order["RowLabel"], columns=["Sex", "Age", "Country"])
    pivot = pivot.reset_index()

    return pivot


def plot_heatmap(delta_table: pd.DataFrame, output_base: Path) -> None:
    if delta_table.empty:
        raise ValueError("No rows available to plot.")

    value_cols = ["Sex", "Age", "Country"]
    labels = delta_table["RowLabel"].tolist()
    values = delta_table[value_cols].to_numpy(dtype=float)

    vmax = float(np.nanmax(np.abs(values)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    im = ax.imshow(values, cmap="RdYlGn_r", norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(value_cols)))
    ax.set_xticklabels(value_cols)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            text_color = "white" if abs(value) > vmax * 0.55 else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{value:+.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    family_breaks = []
    previous_family = None
    for idx, label in enumerate(labels):
        family = label.rsplit(" ", 1)[0]
        if previous_family is not None and family != previous_family:
            family_breaks.append(idx - 0.5)
        previous_family = family

    for y in family_breaks:
        ax.axhline(y, color="white", linewidth=2.2)

    for x in np.arange(-0.5, len(value_cols), 1):
        ax.axvline(x, color="white", linewidth=1.0, alpha=0.8)
    for y in np.arange(-0.5, len(labels), 1):
        ax.axhline(y, color="white", linewidth=1.0, alpha=0.5)

    ax.set_title("Secondary Subgroup Fairness Change vs Baseline", pad=12)
    ax.set_xlabel("Reporting dimension")
    ax.set_ylabel("Mitigation setting")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\Delta$ EOD vs baseline")

    fig.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a heatmap of secondary-group EOD change relative to baseline."
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default=None,
        help="Path to raw evaluation result folders.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing aggregated compare_fairness_across_seeds CSV outputs.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[32],
        help="Seeds to aggregate.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="imagenet_tiny",
        help="Model name used in experiment file names.",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="Split 1",
        help="Split label to visualize, e.g. 'Split 1'.",
    )
    parser.add_argument(
        "--mitigation_n_folds",
        type=int,
        default=5,
        help="Fold count used in mitigation experiment names.",
    )
    parser.add_argument(
        "--baseline_experiment",
        type=str,
        default="exp6",
        help="Baseline experiment key used for delta computation.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        help="Experiment keys to include.",
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="figs/secondary_eod_delta_heatmap",
        help="Output path without extension.",
    )
    parser.add_argument(
        "--export_csv",
        type=str,
        default="tmp/secondary_eod_delta_heatmap.csv",
        help="Optional CSV export of the plotted delta values.",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_path) if args.eval_path else None
    input_dir = Path(args.input_dir) if args.input_dir else None
    output_base = Path(args.output_base)
    export_csv = Path(args.export_csv)

    summaries = []
    if input_dir is not None:
        for exp_key in args.experiments:
            summary = load_aggregated_fold_fairness_summary(exp_key=exp_key, input_dir=input_dir)
            if not summary.empty:
                summaries.append(summary)
    else:
        if eval_path is None:
            raise ValueError("Provide either --input_dir for aggregated CSVs or --eval_path for raw results.")

        exp_configs = configure_experiments(args.mitigation_n_folds)
        for exp_key in args.experiments:
            exp_config = exp_configs.get(exp_key)
            if exp_config is None:
                print(f"Skipping unknown experiment key: {exp_key}")
                continue
            summary = load_fold_fairness_summary(
                exp_key=exp_key,
                exp_config=exp_config,
                seeds=args.seeds,
                eval_path=eval_path,
                model=args.model,
            )
            if not summary.empty:
                summaries.append(summary)

    if not summaries:
        extra_hint = ""
        if input_dir is None and eval_path is not None and eval_path.name == "pareto":
            extra_hint = (
                " You appear to be pointing at the Pareto plot directory. "
                "For aggregated CSVs, use --input_dir results/fairness_comparison instead."
            )
        raise ValueError(
            "No fairness summaries could be loaded. Check your input path and experiment names."
            + extra_hint
        )

    fairness_summary = pd.concat(summaries, ignore_index=True)
    delta_table = build_delta_table(
        fairness_summary=fairness_summary,
        split_name=args.split_name,
        baseline_experiment=args.baseline_experiment,
    )
    if delta_table.empty:
        raise ValueError(
            f"No delta rows available for split '{args.split_name}'. Check the selected experiments."
        )

    export_csv.parent.mkdir(parents=True, exist_ok=True)
    delta_table.to_csv(export_csv, index=False)
    print(f"Saved CSV: {export_csv}")

    plot_heatmap(delta_table, output_base)
    print(f"Saved figure: {output_base.with_suffix('.png')}")
    print(f"Saved figure: {output_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
