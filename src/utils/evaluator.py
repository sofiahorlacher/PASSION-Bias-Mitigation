import ast
import itertools
import math
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    equalized_odds_difference,
    equalized_odds_ratio,
    false_positive_rate,
    true_positive_rate,
)
from fairlearn.reductions import EqualizedOdds
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.utils.passion_metadata import (
    exclude_fitzpatrick_rows,
    filter_split_to_subjects,
)


DEFAULT_THESIS_REPORTING_GROUPS = {
    "fitzpatrick": {
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
    },
    "age_bins": [0, 5, 18, 40, np.inf],
    "age_labels": ["0-4", "5-17", "18-39", "40+"],
    "country": {
        "Malawi": "EastAfrica",
        "Tanzania": "EastAfrica",
    },
    "sex": "keep_as_is",
}

DEFAULT_THESIS_EVALUATION_PROTOCOL = {
    "primary_group": "fitzpatrick",
    "secondary_groups": ["sex", "ageGroup", "country"],
    "primary_performance_metric": "score_mean",
    "primary_fairness_metric": "fitzpatrick_primary_eod_mean",
    "primary_subgroup_metric": "fitzpatrick_worst_supported_balancedAcc_mean",
    "secondary_fairness_metric": "fitzpatrick_worst_eod_mean",
    "main_result_file": "comparison_summary_test.csv",
    "robustness_file": "comparison_summary_foldsummary.csv",
    "raw_sidecar_suffix": "__raw",
}

DEFAULT_GROUPING_COLUMNS = ["fitzpatrick", "sex", "ageGroup", "country"]


# Detailed evaluation keeps two views:
# - reporting: support-aware grouped demographics used for the main thesis results
# - raw: fine-grained audit outputs kept for transparency and appendix checks
class BiasEvaluator:
    def __init__(
        self,
        passion_exp: str = "experiment_standard_split_conditions",
        eval_data_path="../../assets/evaluation",
        dataset_dir: Union[str, Path] = "../../data/PASSION/",
        meta_data_file: Union[str, Path] = "label.csv",
        split_file: Union[str, Path, None] = "PASSION_split.csv",
        threshold_eq_rates: float = 0.015,
        target_names: [str] = None,
        labels: [str] = None,
        evaluator_config: Optional[dict] = None,
        exclude_fitzpatrick_values: Optional[Sequence[Union[str, int]]] = None,
    ):
        self.passion_exp = passion_exp
        self.eval_data_path = Path(eval_data_path)
        self.dataset_dir = Path(dataset_dir)
        self.out_dir = self.eval_data_path / passion_exp
        os.makedirs(self.out_dir, exist_ok=True)

        self.threshold_eq_rates = threshold_eq_rates

        self.target_names = target_names
        self.labels = labels
        self.evaluator_config = evaluator_config or {}

        self.reporting_groups = deepcopy(DEFAULT_THESIS_REPORTING_GROUPS)
        configured_reporting_groups = self.evaluator_config.get("reporting_groups", {})
        for key, value in configured_reporting_groups.items():
            if isinstance(value, dict) and isinstance(
                self.reporting_groups.get(key), dict
            ):
                self.reporting_groups[key].update(value)
            else:
                self.reporting_groups[key] = value

        self.evaluation_protocol = deepcopy(DEFAULT_THESIS_EVALUATION_PROTOCOL)
        self.evaluation_protocol.update(self.evaluator_config.get("protocol", {}))
        self.grouping_columns = self.evaluator_config.get(
            "grouping_columns",
            DEFAULT_GROUPING_COLUMNS,
        )
        self.include_intersectional_groups = bool(
            self.evaluator_config.get("include_intersectional_groups", False)
        )

        # TODO: only read once in the whole pipeline
        self.df_labels = exclude_fitzpatrick_rows(
            pd.read_csv(self.dataset_dir / meta_data_file),
            exclude_fitzpatrick_values=exclude_fitzpatrick_values,
        )
        self.df_split = filter_split_to_subjects(
            pd.read_csv(self.dataset_dir / split_file),
            self.df_labels["subject_id"],
        )

    def get_subgroup_metric_results_file_name(
        self,
        add_run_info,
        evaluation_view: str = "reporting",
    ):
        suffix = "" if evaluation_view == "reporting" else f"__{evaluation_view}"
        return (
            self.out_dir
            / f"subgroup_metric_results_{self.passion_exp}__{add_run_info}{suffix}.csv"
        )

    def get_fairness_metric_results_file_name(
        self,
        add_run_info,
        evaluation_view: str = "reporting",
    ):
        suffix = "" if evaluation_view == "reporting" else f"__{evaluation_view}"
        return (
            self.out_dir
            / f"fairness_metric_results_{self.passion_exp}__{add_run_info}{suffix}.csv"
        )

    def get_predictions_with_meta_data_file_name(self, add_run_info):
        return (
            self.out_dir
            / f"predictions_with_metadata_{self.passion_exp}__{add_run_info}_img_lvl.csv"
        )

    @staticmethod
    def _parse_image_paths(s: str):
        if isinstance(s, np.ndarray):
            return s.tolist()
        if isinstance(s, (list, tuple)):
            return list(s)
        if pd.isna(s):
            return []
        text = str(s).strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, np.ndarray)):
                return [str(path) for path in parsed]
        except (SyntaxError, ValueError):
            pass
        return (
            text.replace("\n", " ")
            .replace("'", '"')
            .replace("[", "")
            .replace("]", "")
            .split()
        )

    @staticmethod
    def _parse_numpy_array(s: str):
        if isinstance(s, np.ndarray):
            return s.astype(int, copy=False)
        if isinstance(s, (list, tuple)):
            return np.asarray(s, dtype=int)
        if pd.isna(s):
            return np.asarray([], dtype=int)

        text = str(s).strip()
        if not text:
            return np.asarray([], dtype=int)

        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, np.ndarray)):
                return np.asarray(parsed, dtype=int)
        except (SyntaxError, ValueError):
            pass

        return np.fromstring(text.strip("[]").replace("\n", " "), sep=" ", dtype=int)

    @staticmethod
    def _parse_probability_matrix(s: str):
        if isinstance(s, np.ndarray):
            return s
        if isinstance(s, list):
            return np.asarray(s, dtype=float)
        if pd.isna(s):
            return None

        text = str(s).strip()
        if not text:
            return None

        try:
            return np.asarray(ast.literal_eval(text), dtype=float)
        except (SyntaxError, ValueError):
            rows = []
            for row_text in re.findall(r"\[([^\[\]]+)\]", text):
                row = np.fromstring(row_text.replace(",", " "), sep=" ", dtype=float)
                if row.size:
                    rows.append(row)
            if rows:
                return np.vstack(rows)
        return None

    @staticmethod
    def _extract_subject_id(path: str):
        match = re.search(r"([A-Za-z]+[0-9]+)", path)
        return str(match.group(1)).strip() if match else np.nan

    @staticmethod
    def _sanitize_metric_name(name: str) -> str:
        return re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_").lower()

    def _get_probability_column_names(self) -> list[str]:
        return [
            f"prob_{self._sanitize_metric_name(class_name)}"
            for class_name in self.target_names
        ]

    def _prepare_serialized_results(self, df_results: pd.DataFrame) -> pd.DataFrame:
        """Parse serialized prediction columns back into iterable objects."""
        df_results = df_results.copy()
        parse_map = {
            "FileNames": self._parse_image_paths,
            "Indices": self._parse_numpy_array,
            "EvalTargets": self._parse_numpy_array,
            "EvalPredictions": self._parse_numpy_array,
            "EvalProbabilities": self._parse_probability_matrix,
        }
        for col, parser in parse_map.items():
            if col in df_results.columns:
                df_results[col] = df_results[col].apply(parser)
        return df_results

    def _get_data_with_metadata_from_csv(
        self, create_data: bool = True, add_run_info: str = ""
    ):
        predictions_n_meta_data_file = self.get_predictions_with_meta_data_file_name(
            add_run_info
        )
        if create_data:
            input_file = self.eval_data_path / f"{self.passion_exp}__{add_run_info}.csv"
            df_results = pd.read_csv(input_file)
            df_results = df_results.iloc[[-1]]
            df_results = self._prepare_serialized_results(df_results)

            data = self._aggregate_data_with_metadata(df_results)
            predictions_n_meta_data_file.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(predictions_n_meta_data_file, index=False)
            return data

        return pd.read_csv(predictions_n_meta_data_file)

    def _get_data_with_metadata(
        self, df_results: pd.DataFrame = None, add_run_info: str = ""
    ):
        predictions_n_meta_data_file = self.get_predictions_with_meta_data_file_name(
            add_run_info
        )
        predictions_n_meta_data_file.parent.mkdir(parents=True, exist_ok=True)
        df_results = self._prepare_serialized_results(df_results)
        data = self._aggregate_data_with_metadata(df_results)
        data.to_csv(predictions_n_meta_data_file, index=False)
        return data

    def _aggregate_data_with_metadata(self, df_results):
        output_rows = []
        seen_subject_ids = set()
        prob_cols = self._get_probability_column_names()
        for _, row in df_results.iterrows():
            probabilities = row.get("EvalProbabilities")
            if probabilities is None or (
                isinstance(probabilities, float) and pd.isna(probabilities)
            ):
                probabilities = [None] * len(row["EvalPredictions"])

            for img_name, idx, lbl, pred, pred_scores in zip(
                row["FileNames"],
                row["Indices"],
                row["EvalTargets"],
                row["EvalPredictions"],
                probabilities,
            ):
                subject_id = self._extract_subject_id(img_name)
                if pd.isna(subject_id):
                    raise ValueError(
                        f"Could not extract subject_id from image path: {img_name}"
                    )

                label_matches = self.df_labels[
                    self.df_labels["subject_id"].astype(str).str.strip() == subject_id
                ]
                if label_matches.empty:
                    raise ValueError(
                        "No metadata row found in label file for "
                        f"subject_id={subject_id} extracted from image path {img_name}"
                    )
                labels = label_matches.iloc[0]

                split_matches = self.df_split[
                    self.df_split["subject_id"].astype(str).str.strip() == subject_id
                ]
                if split_matches.empty:
                    raise ValueError(
                        "No split row found in split file for "
                        f"subject_id={subject_id} extracted from image path {img_name}"
                    )
                split = split_matches.iloc[0]
                probability_dict = {}
                if pred_scores is not None:
                    pred_scores = np.asarray(pred_scores, dtype=float).reshape(-1)
                    if pred_scores.size:
                        probability_dict = {
                            col: float(score)
                            for col, score in zip(prob_cols, pred_scores)
                        }
                output_rows.append(
                    {
                        "correct": lbl == pred,
                        "image_path": img_name,
                        "index": idx,
                        "targets": lbl,
                        "predictions": pred,
                        **probability_dict,
                        **labels.to_dict(),
                        **split.drop("subject_id").to_dict(),
                    }
                )
                seen_subject_ids.add(subject_id)
        print(f"{len(seen_subject_ids)} unique subjects processed.")
        data = pd.DataFrame(output_rows)
        return data

    def _extract_probability_matrix(
        self, data: pd.DataFrame
    ) -> Optional[np.ndarray]:
        prob_cols = self._get_probability_column_names()
        if not all(col in data.columns for col in prob_cols):
            return None

        prob_df = data[prob_cols]
        if prob_df.isna().any().any():
            return None
        return prob_df.to_numpy(dtype=float)

    def _print_classification_stats(
        self, y_true, y_pred, balanced_accuracy, macro_auroc=np.nan
    ):
        print(
            classification_report(
                y_true,
                y_pred,
                labels=self.labels,
                target_names=self.target_names,
                zero_division=0,
            )
        )
        print(f"Balanced Accuracy: {balanced_accuracy}")
        if not np.isnan(macro_auroc):
            print(f"Macro AUROC: {macro_auroc}")

    def _get_auroc_metrics(self, y_true, y_score):
        if y_score is None:
            return {
                "macroAUROC": np.nan,
                "per_class": {},
            }

        y_score = np.asarray(y_score, dtype=float)
        if y_score.ndim != 2 or y_score.shape[0] != len(y_true):
            raise ValueError(
                "Expected y_score to have shape (n_samples, n_classes) for AUROC."
            )

        per_class = {}
        auroc_values = []
        labels = self.labels or list(range(y_score.shape[1]))
        for class_idx, (class_label, class_name) in enumerate(
            zip(labels, self.target_names)
        ):
            y_true_bin = (y_true == class_label).astype(int)
            if np.unique(y_true_bin).size < 2:
                per_class[class_name] = np.nan
                continue
            try:
                auroc = float(roc_auc_score(y_true_bin, y_score[:, class_idx]))
            except ValueError:
                auroc = np.nan
            per_class[class_name] = auroc
            if not np.isnan(auroc):
                auroc_values.append(auroc)

        return {
            "macroAUROC": (
                float(np.mean(auroc_values)) if auroc_values else np.nan
            ),
            "per_class": per_class,
        }

    def _get_confusion_metrics(self, y_true, y_pred, y_score=None):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        cm = confusion_matrix(y_true, y_pred, labels=self.labels)
        print("Confusion Matrix:")
        print(cm)
        print(f"y_true: {set(y_true)}")
        print(f"y_pred: {set(y_pred)}")

        cumulated = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "tpr": [],
            "fpr": [],
            "eq_odds_diffs": [],
            "eq_odds_diff_toOveralls": [],
            "eq_odds_diff_means": [],
            "eq_odds_diff_mean_toOveralls": [],
            "eq_odds_ratios": [],
            "eq_odds_ratio_toOveralls": [],
            "eq_odds_ratio_means": [],
            "eq_odds_ratio_mean_toOveralls": [],
        }
        per_class = {}

        # per class
        for i, name in enumerate(self.target_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)

            tpr = round((tp / (tp + fn)), 2) if (tp + fn) else 0
            fpr = round((fp / (fp + tn)), 2) if (fp + tn) else 0

            cumulated["tp"] += tp
            cumulated["fp"] += fp
            cumulated["fn"] += fn
            cumulated["tn"] += tn
            cumulated["tpr"].append(tpr)
            cumulated["fpr"].append(fpr)

            print(
                f"{name} — TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}, TPR: {tpr}, FPR: {fpr}"
            )

            per_class[name] = {
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "TPR": tpr,
                "FPR": fpr,
            }

        tp_fn = cumulated["tp"] + cumulated["fn"]
        micro_tpr = round(cumulated["tp"] / tp_fn, 2) if tp_fn else 0
        fp_tn = cumulated["fp"] + cumulated["tn"]
        micro_fpr = round(cumulated["fp"] / fp_tn, 2) if fp_tn else 0
        macro_tpr = round(np.mean(cumulated["tpr"]), 2)
        macro_fpr = round(np.mean(cumulated["fpr"]), 2)

        print(
            f"cumulated - TP: {cumulated['tp']}, FP: {cumulated['fp']}, FN: {cumulated['fn']}, TN: {cumulated['tn']}"
            f", macro-TPR (avg): {macro_tpr}, macro-FPR (avg): {macro_fpr}"
            f", micro-TPR: {micro_tpr}, micro-FPR: {micro_fpr}"
        )

        auroc_metrics = self._get_auroc_metrics(y_true, y_score)
        for class_name, auroc in auroc_metrics["per_class"].items():
            per_class[class_name]["AUROC"] = auroc

        return {
            "macro-tpr": macro_tpr,
            "macro-fpr": macro_fpr,
            "micro-tpr": micro_tpr,
            "micro-fpr": micro_fpr,
            "macroAUROC": auroc_metrics["macroAUROC"],
            "per_class": per_class,
            "cumulated": {
                "TP": cumulated["tp"],
                "FP": cumulated["fp"],
                "FN": cumulated["fn"],
                "TN": cumulated["tn"],
            },
            "balancedAcc": balanced_accuracy_score(y_true, y_pred),
        }

    def run_full_evaluation(
        self,
        analysis_name: str,
        data: pd.DataFrame = None,
        add_run_info: str = None,
        run_detailed_evaluation: bool = False,
        detailed_evaluation_mode: str = "reporting",
    ):
        if data is None:
            df = self._get_data_with_metadata_from_csv(
                create_data=True, add_run_info=add_run_info
            )
        else:
            df = self._get_data_with_metadata(
                df_results=data, add_run_info=add_run_info
            )

        print(
            f"********************* Overall Evaluation - {analysis_name} *********************"
        )
        y_true = df["targets"]
        y_pred = df["predictions"]
        y_score = self._extract_probability_matrix(df)
        self._calculate_metrics(y_pred, y_true, y_score=y_score)

        if run_detailed_evaluation:
            print(
                f"********************* Detailed Evaluation - {analysis_name} *********************"
            )
            grouping_columns = self.grouping_columns

            if detailed_evaluation_mode not in {"reporting", "raw", "both"}:
                raise ValueError(
                    "detailed_evaluation_mode must be one of reporting, raw, both"
                )

            if detailed_evaluation_mode in {"reporting", "both"}:
                reporting_df = self._apply_reporting_groups(df)
                self._detailed_evaluation(
                    reporting_df,
                    grouping_columns,
                    add_run_info,
                    evaluation_view="reporting",
                )

            if detailed_evaluation_mode in {"raw", "both"}:
                raw_df = self._apply_raw_groups(df)
                self._detailed_evaluation(
                    raw_df,
                    grouping_columns,
                    add_run_info,
                    evaluation_view="raw",
                )

    def fairlearn_output(self, sensitive_features, y_pred, y_true):
        cumulated = {
            "eq_odds_diffs": [],
            "eq_odds_diff_toOveralls": [],
            "eq_odds_diff_means": [],
            "eq_odds_diff_mean_toOveralls": [],
            "eq_odds_ratios": [],
            "eq_odds_ratio_toOveralls": [],
            "eq_odds_ratio_means": [],
            "eq_odds_ratio_mean_toOveralls": [],
        }
        per_class = {}

        for i, name in enumerate(self.target_names):
            (
                eq_odds_diff,
                eq_odds_diff_mean,
                eq_odds_diff_mean_toOverall,
                eq_odds_diff_toOverall,
                eq_odds_ratio,
                eq_odds_ratio_mean,
                eq_odds_ratio_mean_toOverall,
                eq_odds_ratio_toOverall,
                group_metrics,
                metric_frame,
                metrics,
                overall_metrics,
            ) = self._calc_fairness(i, name, sensitive_features, y_pred, y_true)

            cumulated["eq_odds_diffs"].append(eq_odds_diff)
            cumulated["eq_odds_diff_toOveralls"].append(eq_odds_diff_toOverall)
            cumulated["eq_odds_diff_means"].append(eq_odds_diff_mean)
            cumulated["eq_odds_diff_mean_toOveralls"].append(
                eq_odds_diff_mean_toOverall
            )
            cumulated["eq_odds_ratios"].append(eq_odds_ratio)
            cumulated["eq_odds_ratio_toOveralls"].append(eq_odds_ratio_toOverall)
            cumulated["eq_odds_ratio_means"].append(eq_odds_ratio_mean)
            cumulated["eq_odds_ratio_mean_toOveralls"].append(
                eq_odds_ratio_mean_toOverall
            )

            print("Fairness Metrics:")
            print(f"Equalized Odds Difference: {eq_odds_diff}")
            print(
                f"Equalized Odds Difference Mean To Overall: {eq_odds_diff_mean_toOverall}"
            )
            print(f"Equalized Odds Ratio: {eq_odds_ratio}")
            print(
                f"Equalized Odds Ratio Mean To Overall: {eq_odds_ratio_mean_toOverall}"
            )
            print("Overall metrics:")
            print(overall_metrics)

            print("Summary per metric:")
            summary = {}
            for metric_name in metrics.keys():
                max_val = group_metrics[metric_name].max()
                max_group = group_metrics[metric_name].idxmax()
                min_val = group_metrics[metric_name].min()
                min_group = group_metrics[metric_name].idxmin()
                diff = metric_frame.difference()[metric_name]
                ratio = metric_frame.ratio()[metric_name]

                print(
                    f"{metric_name}: \n"
                    f"max        = {max_val:.6f} (Group: {max_group}), \n"
                    f"min        = {min_val:.6f} (Group: {min_group}), \n"
                    f"difference = {diff:.6f}, \n"
                    f"ratio      = {ratio:.6f}"
                )

                summary[metric_name] = {
                    "max": max_val,
                    "max_group": max_group,
                    "min": min_val,
                    "min_group": min_group,
                    "difference": diff,
                    "ratio": ratio,
                }

            # Convert the MultiIndex to tuples of native Python types to fix fitzpatrick
            def convert_index_value(val):
                if isinstance(val, (np.integer, np.int64, np.int32)):
                    return int(val)
                return val

            if isinstance(group_metrics.index, pd.MultiIndex):
                group_metrics.index = group_metrics.index.map(
                    lambda tup: tuple(convert_index_value(i) for i in tup)
                )
            else:
                group_metrics.index = group_metrics.index.map(convert_index_value)

            print("Group-wise Metrics:")
            print(group_metrics)

            per_class[name] = {
                "eq_odds_diff": eq_odds_diff,
                "eq_odds_diff_mean": eq_odds_diff_mean,
                "eq_odds_diff_mean_toOverall": eq_odds_diff_mean_toOverall,
                "eq_odds_diff_toOverall": eq_odds_diff_toOverall,
                "eq_odds_ratio": eq_odds_ratio,
                "eq_odds_ratio_mean": eq_odds_ratio_mean,
                "eq_odds_ratio_mean_toOverall": eq_odds_ratio_mean_toOverall,
                "eq_odds_ratio_toOverall": eq_odds_ratio_toOverall,
                "metrics": list(metrics.keys()),
                "overall_metrics": overall_metrics.to_dict(),
                "group_wise_metric_results": group_metrics.to_dict(orient="index"),
                "fairness_summary": summary,
            }

        overall_eod_worst = max(cumulated["eq_odds_diffs"])
        overall_eod_mean = sum(cumulated["eq_odds_diffs"]) / len(
            cumulated["eq_odds_diffs"]
        )
        overall_eod_best = min(cumulated["eq_odds_diffs"])
        overall_eod_to_overall_worst = max(cumulated["eq_odds_diff_toOveralls"])
        overall_eod_to_overall_mean = sum(cumulated["eq_odds_diff_toOveralls"]) / len(
            cumulated["eq_odds_diff_toOveralls"]
        )
        overall_eod_to_overall_best = min(cumulated["eq_odds_diff_toOveralls"])
        overall_eod_mean_worst = max(cumulated["eq_odds_diff_means"])
        overall_eod_mean_mean = sum(cumulated["eq_odds_diff_means"]) / len(
            cumulated["eq_odds_diff_means"]
        )
        overall_eod_mean_best = min(cumulated["eq_odds_diff_means"])
        overall_eod_mean_to_overall_worst = max(
            cumulated["eq_odds_diff_mean_toOveralls"]
        )
        overall_eod_mean_to_overall_mean = sum(
            cumulated["eq_odds_diff_mean_toOveralls"]
        ) / len(cumulated["eq_odds_diff_mean_toOveralls"])
        overall_eod_mean_to_overall_best = min(
            cumulated["eq_odds_diff_mean_toOveralls"]
        )
        overall_eor_worst = min(cumulated["eq_odds_ratios"])
        overall_eor_mean = sum(cumulated["eq_odds_ratios"]) / len(
            cumulated["eq_odds_ratios"]
        )
        overall_eor_best = max(cumulated["eq_odds_ratios"])
        overall_eor_to_overall_worst = min(cumulated["eq_odds_ratio_toOveralls"])
        overall_eor_to_overall_mean = sum(cumulated["eq_odds_ratio_toOveralls"]) / len(
            cumulated["eq_odds_ratio_toOveralls"]
        )
        overall_eor_to_overall_best = max(cumulated["eq_odds_ratio_toOveralls"])
        overall_eor_mean_worst = min(cumulated["eq_odds_ratio_means"])
        overall_eor_mean_mean = sum(cumulated["eq_odds_ratio_means"]) / len(
            cumulated["eq_odds_ratio_means"]
        )
        overall_eor_mean_best = max(cumulated["eq_odds_ratio_means"])
        overall_eor_mean_to_overall_worst = min(
            cumulated["eq_odds_ratio_mean_toOveralls"]
        )
        overall_eor_mean_to_overall_mean = sum(
            cumulated["eq_odds_ratio_mean_toOveralls"]
        ) / len(cumulated["eq_odds_ratio_mean_toOveralls"])
        overall_eor_mean_to_overall_best = max(
            cumulated["eq_odds_ratio_mean_toOveralls"]
        )

        print("most important eod & eor values, more can be found in the csv")
        print(f"overall_eod_mean_to_overall_worst: {overall_eod_mean_to_overall_worst}")
        print(f"overall_eod_mean_to_overall_mean: {overall_eod_mean_to_overall_mean}")
        print(f"overall_eod_mean_to_overall_best: {overall_eod_mean_to_overall_best}")
        print()
        print(f"overall_eor_mean_to_overall_worst: {overall_eor_mean_to_overall_worst}")
        print(f"overall_eor_mean_to_overall_mean: {overall_eor_mean_to_overall_mean}")
        print(f"overall_eor_mean_to_overall_best: {overall_eor_mean_to_overall_best}")
        print()

        results = {
            #  "cumulated": {
            "overall_eod_worst": overall_eod_worst,
            "overall_eod_mean": overall_eod_mean,
            "overall_eod_best": overall_eod_best,
            "overall_eod_to_overall_worst": overall_eod_to_overall_worst,
            "overall_eod_to_overall_mean": overall_eod_to_overall_mean,
            "overall_eod_to_overall_best": overall_eod_to_overall_best,
            "overall_eod_mean_worst": overall_eod_mean_worst,
            "overall_eod_mean_mean": overall_eod_mean_mean,
            "overall_eod_mean_best": overall_eod_mean_best,
            "overall_eod_mean_to_overall_worst": overall_eod_mean_to_overall_worst,
            "overall_eod_mean_to_overall_mean": overall_eod_mean_to_overall_mean,
            "overall_eod_mean_to_overall_best": overall_eod_mean_to_overall_best,
            "overall_eor_worst": overall_eor_worst,
            "overall_eor_mean": overall_eor_mean,
            "overall_eor_best": overall_eor_best,
            "overall_eor_to_overall_worst": overall_eor_to_overall_worst,
            "overall_eor_to_overall_mean": overall_eor_to_overall_mean,
            "overall_eor_to_overall_best": overall_eor_to_overall_best,
            "overall_eor_mean_worst": overall_eor_mean_worst,
            "overall_eor_mean_mean": overall_eor_mean_mean,
            "overall_eor_mean_best": overall_eor_mean_best,
            "overall_eor_mean_to_overall_worst": overall_eor_mean_to_overall_worst,
            "overall_eor_mean_to_overall_mean": overall_eor_mean_to_overall_mean,
            "overall_eor_mean_to_overall_best": overall_eor_mean_to_overall_best,
            #            },
            "per_class": per_class,
        }

        return results

    def _calc_fairness(self, i, name, sensitive_features, y_pred, y_true):
        print(
            f"i: {i}, name: {name}; sensitive_features: {list(sensitive_features.columns)}"
        )
        y_true_bin = (y_true == i).astype(int)
        y_pred_bin = (y_pred == i).astype(int)
        eq_odds_diff = equalized_odds_difference(
            y_true=y_true_bin, y_pred=y_pred_bin, sensitive_features=sensitive_features
        )
        eq_odds_diff_mean = equalized_odds_difference(
            y_true=y_true_bin,
            y_pred=y_pred_bin,
            sensitive_features=sensitive_features,
            agg="mean",
        )
        eq_odds_diff_toOverall = equalized_odds_difference(
            y_true=y_true_bin,
            y_pred=y_pred_bin,
            sensitive_features=sensitive_features,
            method="to_overall",
        )
        eq_odds_diff_mean_toOverall = equalized_odds_difference(
            y_true=y_true_bin,
            y_pred=y_pred_bin,
            sensitive_features=sensitive_features,
            agg="mean",
            method="to_overall",
        )
        eq_odds_ratio = equalized_odds_ratio(
            y_true=y_true_bin, y_pred=y_pred_bin, sensitive_features=sensitive_features
        )
        eq_odds_ratio_mean = equalized_odds_ratio(
            y_true=y_true_bin,
            y_pred=y_pred_bin,
            sensitive_features=sensitive_features,
            agg="mean",
        )
        eq_odds_ratio_toOverall = equalized_odds_ratio(
            y_true=y_true_bin,
            y_pred=y_pred_bin,
            sensitive_features=sensitive_features,
            method="to_overall",
        )
        eq_odds_ratio_mean_toOverall = equalized_odds_ratio(
            y_true=y_true_bin,
            y_pred=y_pred_bin,
            sensitive_features=sensitive_features,
            method="to_overall",
            agg="mean",
        )
        metrics = {
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
            "support": count,
        }
        metric_frame = MetricFrame(
            metrics=metrics,
            y_true=y_true_bin,
            y_pred=y_pred_bin,
            sensitive_features=sensitive_features,
        )
        overall_metrics = metric_frame.overall
        group_metrics = metric_frame.by_group

        return (
            eq_odds_diff,
            eq_odds_diff_mean,
            eq_odds_diff_mean_toOverall,
            eq_odds_diff_toOverall,
            eq_odds_ratio,
            eq_odds_ratio_mean,
            eq_odds_ratio_mean_toOverall,
            eq_odds_ratio_toOverall,
            group_metrics,
            metric_frame,
            metrics,
            overall_metrics,
        )

    def _generate_age_group_raw(self, df):
        bins = range(0, 101, 5)
        return pd.cut(
            df["age"],
            bins=bins,
            labels=[f"{i:02}-{i + 4:02}" for i in bins[:-1]],
            right=False,
        )

    def _generate_age_group(self, df):
        # Use broader reporting bins so subgroup fairness metrics stay interpretable
        # and do not collapse into many tiny age buckets.
        bins = self.reporting_groups["age_bins"]
        return pd.cut(
            df["age"],
            bins=bins,
            labels=self.reporting_groups["age_labels"],
            right=False,
        )

    def _generate_country_group(self, df):
        return df["country"].replace(self.reporting_groups["country"])

    def _generate_fitzpatrick_group(self, df):
        fitzpatrick = df["fitzpatrick"].astype(str).str.strip()
        return fitzpatrick.replace(self.reporting_groups["fitzpatrick"])

    def _apply_reporting_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        grouped_df = df.copy()
        grouped_df["fitzpatrick"] = self._generate_fitzpatrick_group(grouped_df)
        grouped_df["ageGroup"] = self._generate_age_group(grouped_df)
        grouped_df["country"] = self._generate_country_group(grouped_df)
        return grouped_df

    def _apply_raw_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        raw_df = df.copy()
        raw_df["fitzpatrick"] = raw_df["fitzpatrick"].astype(str).str.strip()
        raw_df["ageGroup"] = self._generate_age_group_raw(raw_df)
        return raw_df

    def collect_subgroup_results(self, data, group_by: list[str]):
        def to_pascal_case(s: str) -> str:
            return "".join(word.capitalize() for word in s.split("_"))

        grouped = sorted(data.groupby(group_by, observed=False))
        results = []
        result_keys = None

        # per subgroup
        for group_values, _df in grouped:
            if isinstance(group_values, str):
                group_values = (group_values,)  # Ensure tuple

            support = _df.shape[0]
            if support == 0:
                print(f"no support: group_by: {group_by}, group_values: {group_values}")
                results.append(
                    {
                        **{attr: val for attr, val in zip(group_by, group_values)},
                        "Support": 0,
                    }
                )
                continue

            group_info = ", ".join(
                f"{to_pascal_case(attr)}: {val}"
                for attr, val in zip(group_by, group_values)
            )
            print("~" * 20 + f" {group_info}, Support: {support} " + "~" * 20)

            y_true = data.loc[_df.index, ["targets"]]
            y_pred = data.loc[_df.index, ["predictions"]]
            y_score = self._extract_probability_matrix(_df)
            metrics = self._calculate_metrics(y_pred, y_true, y_score=y_score)

            result = {
                **{attr: val for attr, val in zip(group_by, group_values)},
                "Support": support,
                "Macro-TPR": metrics["macro-tpr"],
                "Macro-FPR": metrics["macro-fpr"],
                "Micro-TPR": metrics["micro-tpr"],
                "Micro-FPR": metrics["micro-fpr"],
                "macroAUROC": metrics["macroAUROC"],
                "TP": metrics["cumulated"]["TP"],
                "FP": metrics["cumulated"]["FP"],
                "FN": metrics["cumulated"]["FN"],
                "TN": metrics["cumulated"]["TN"],
                "balancedAcc": metrics["balancedAcc"],
            }
            if not result_keys:
                result_keys = list(result.keys())

            for k, v in metrics["cumulated"].items():
                if k not in ("TP", "FP", "FN", "TN"):  # already added
                    result[k] = v

            # Add per-class metrics
            for class_name, values in metrics["per_class"].items():
                for metric_name, metric_val in values.items():
                    # Skip MetricFrame or large objects like group_metrics, metric_frame if not serializing to CSV
                    if metric_name in (
                        "group_metrics",
                        "metric_frame",
                        "metrics",
                        "overall_metrics",
                        "fairness_summary",
                    ):
                        continue
                    if "fairness_summary" in values:
                        for metric_name, summary in values["fairness_summary"].items():
                            result[
                                f"{class_name}_fairnessSummary_{metric_name}_max"
                            ] = summary["max"]
                            result[
                                f"{class_name}_fairnessSummary_{metric_name}_min"
                            ] = summary["min"]
                            result[
                                f"{class_name}_fairnessSummary_{metric_name}_difference"
                            ] = summary["difference"]
                            result[
                                f"{class_name}_fairnessSummary_{metric_name}_ratio"
                            ] = summary["ratio"]
                        continue
                    result[f"{class_name}_{metric_name}"] = metric_val

            results.append(result)

        return pd.DataFrame(results), result_keys

    def _calculate_metrics(self, y_pred, y_true, y_score=None):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        metrics = self._get_confusion_metrics(y_true, y_pred, y_score=y_score)
        self._print_classification_stats(
            y_true,
            y_pred,
            metrics["balancedAcc"],
            metrics["macroAUROC"],
        )
        return metrics

    def _detailed_evaluation(
        self,
        data,
        grouping_columns,
        add_run_info,
        evaluation_view: str = "reporting",
    ):
        dfs = []
        fairness_dfs = []
        group_by_key = "GroupBy"
        report = {}
        result_keys = None
        for group in self._get_groupings(grouping_columns):
            subgroup_df, result_keys = self.collect_subgroup_results(
                data, group_by=group
            )
            subgroup_df[group_by_key] = ", ".join(group)
            dfs.append(subgroup_df)

            macro_tpr_avg = subgroup_df["Macro-TPR"].mean()
            macro_fpr_avg = subgroup_df["Macro-FPR"].mean()

            fairness_group_df = self.fairlearn_output(
                sensitive_features=data[group],
                y_pred=data["predictions"],
                y_true=data["targets"],
            )
            fairness_group_df[group_by_key] = ", ".join(group)
            fairness_dfs.append(fairness_group_df)

            privileged = []
            underprivileged = []
            avgprivileged = []
            unclear = []
            no_support = []

            # TODO: should rather be based on fairlearn output than on the self calculated metrics
            for _, row in subgroup_df.iterrows():
                support = row["Support"]
                label = (
                    ", ".join(str(row[col]) for col in group)
                    + f"; Support: {support:>4}"
                )
                if support > 0:
                    macro_tpr = row["Macro-TPR"]
                    macro_fpr = row["Macro-FPR"]

                    # Compare TPR
                    tpr_privilege = 0
                    tpr_diff = macro_tpr - macro_tpr_avg
                    if abs(tpr_diff) <= self.threshold_eq_rates:
                        tpr_comp = f"TPR ~ ({macro_tpr:.2f})"
                    elif tpr_diff > 0:
                        tpr_comp = f"TPR ↑ ({macro_tpr:.2f})"
                        tpr_privilege = 1
                    else:
                        tpr_comp = f"TPR ↓ ({macro_tpr:.2f})"
                        tpr_privilege = -1

                    # Compare FPR
                    fpr_privilege = 0
                    fpr_diff = macro_fpr - macro_fpr_avg
                    if abs(fpr_diff) <= self.threshold_eq_rates:
                        fpr_comp = f"FPR ~ ({macro_fpr:.2f})"
                    elif fpr_diff > 0:
                        fpr_comp = f"FPR ↑ ({macro_fpr:.2f})"
                        fpr_privilege = -1
                    else:
                        fpr_comp = f"FPR ↓ ({macro_fpr:.2f})"
                        fpr_privilege = 1

                    reason = f"{tpr_comp}, {fpr_comp}"
                    info = (label, reason)

                    privilege = tpr_privilege + fpr_privilege
                    if privilege > 0:
                        privileged.append(info)
                    elif privilege < 0:
                        underprivileged.append(info)
                    # privilege = 0
                    elif tpr_privilege == fpr_privilege:  # both are 0
                        avgprivileged.append(info)
                    else:
                        unclear.append(info)  # one is +1, one -1
                else:
                    no_support.append((label, None))

            report_key = ", ".join(group)
            report[report_key] = {
                "macro_tpr_avg": macro_tpr_avg,
                "macro_fpr_avg": macro_fpr_avg,
                "categories": {
                    "privileged": privileged,
                    "underprivileged": underprivileged,
                    "average": avgprivileged,
                    "unclear": unclear,
                    "no support": no_support,
                },
            }

        final_df = pd.concat(dfs, ignore_index=True)
        all_other_cols = list(set(final_df.columns) - set(result_keys) - {group_by_key})
        all_other_cols.sort()
        ordered_cols = [group_by_key, *result_keys, *all_other_cols]
        float_cols = final_df.select_dtypes(include="float").columns
        final_df[float_cols] = final_df[float_cols].round(3)

        subgroup_file = self.get_subgroup_metric_results_file_name(
            add_run_info,
            evaluation_view=evaluation_view,
        )
        subgroup_file.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(
            subgroup_file,
            columns=ordered_cols,
            index=False,
        )

        all_general_cols = [
            k for k in fairness_dfs[0].keys() if k != "per_class" and k != group_by_key
        ]

        fairness_dfs = [
            self.flatten_fairlearn_output_to_df(fairness_df)
            for fairness_df in fairness_dfs
        ]
        fairness_df = pd.concat(fairness_dfs, ignore_index=True)

        all_other_cols = list(
            set(fairness_df.columns)
            - set(result_keys)
            - set(all_general_cols)
            - {group_by_key}
        )
        all_other_cols.sort()
        ordered_fairness_cols = [group_by_key, *all_general_cols, *all_other_cols]
        fairness_file = self.get_fairness_metric_results_file_name(
            add_run_info,
            evaluation_view=evaluation_view,
        )
        fairness_file.parent.mkdir(parents=True, exist_ok=True)
        fairness_df.to_csv(
            fairness_file,
            columns=ordered_fairness_cols,
            index=False,
        )

        # Print privilege report
        for group_key, group_report in report.items():
            print(f"\n=== Grouping: {group_key} ===")
            print(
                f"macro-TPR avg: {group_report['macro_tpr_avg']}; macro-FPR avg: {group_report['macro_fpr_avg']}"
            )
            for category, entries in group_report["categories"].items():
                print(f"{category}:")
                for label, reasons in entries:
                    if reasons:
                        print(f"  {label} - Reasons: {reasons}")
                    else:
                        print(f"  {label}")
                if len(entries) == 0:
                    print("  -")

    def flatten_fairlearn_output_to_df(self, results: dict) -> dict:
        flat = {}
        # Copy top-level overall metrics
        for k, v in results.items():
            if k != "per_class":
                flat[k] = v
        # Flatten per_class section
        for class_label, metrics in results.get("per_class", {}).items():
            prefix = f"class_{class_label}"

            for mk, mv in metrics.items():
                if mk == "overall_metrics":
                    for omk, omv in mv.items():
                        flat[f"{prefix}_overall_{omk}"] = omv
                elif mk == "group_wise_metric_results":
                    flat[f"{prefix}_group_wise_metric_results"] = mv
                elif mk == "metrics":
                    flat[f"{prefix}_metrics"] = ",".join(mv)
                elif mk == "fairness_summary":
                    flat[f"{prefix}_summary"] = mv
                else:
                    flat[f"{prefix}_{mk}"] = mv
        return pd.DataFrame([flat])

    def _get_groupings(self, grouping_columns):
        if not self.include_intersectional_groups:
            return [[col] for col in grouping_columns]

        group_combinations = []
        for r in range(1, len(grouping_columns) + 1):
            group_combinations.extend(
                [list(comb) for comb in itertools.combinations(grouping_columns, r)]
            )
        return group_combinations


if __name__ == "__main__":
    import argparse
    import sys
    import yaml

    from src.utils.loader import Loader

    class Tee:
        def __init__(self, filename, mode="a"):
            self.terminal = sys.stdout
            self.log = open(filename, mode, encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Tee("baseline_bias_evaluation.log", mode="w")
    sys.stderr = sys.stdout

    parser = argparse.ArgumentParser(
        description="Run fairness evaluation on an existing experiment CSV without retraining."
    )
    parser.add_argument(
        "--eval_csv",
        type=str,
        required=True,
        help="Path to an existing experiment result CSV, e.g. assets/evaluation/seed_42/<experiment>.csv",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/default.yaml",
        help="Path to the shared config yaml.",
    )
    parser.add_argument(
        "--run_label",
        type=str,
        default="Test",
        help="AdditionalRunInfo to evaluate, e.g. Test or Fold-0.",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        default="finetuning",
        help="EvalType row to select from the experiment CSV.",
    )
    parser.add_argument(
        "--detailed_evaluation_mode",
        type=str,
        choices=["reporting", "raw", "both"],
        default=None,
        help="Override fairness_evaluation.detailed_evaluation_mode from the config.",
    )
    args = parser.parse_args()

    eval_csv = Path(args.eval_csv)
    if not eval_csv.exists():
        raise ValueError(f"Could not find eval_csv: {eval_csv}")

    config_path = Path(args.config_path)
    if not config_path.exists():
        raise ValueError(f"Could not find config_path: {config_path}")

    config = yaml.load(config_path.open("r"), Loader=Loader)
    fairness_config = deepcopy(config.get("fairness_evaluation", {}))
    if args.detailed_evaluation_mode is not None:
        fairness_config["detailed_evaluation_mode"] = args.detailed_evaluation_mode

    df_results = pd.read_csv(eval_csv)
    required_cols = {"EvalType", "AdditionalRunInfo"}
    if not required_cols.issubset(df_results.columns):
        raise ValueError(
            f"Expected columns {required_cols} in {eval_csv}, found {set(df_results.columns)}"
        )

    selected_rows = df_results[
        (df_results["EvalType"] == args.eval_type)
        & (df_results["AdditionalRunInfo"] == args.run_label)
    ].copy()
    if selected_rows.empty:
        raise ValueError(
            f"No row found in {eval_csv} for EvalType={args.eval_type} and AdditionalRunInfo={args.run_label}"
        )

    df_row = selected_rows.iloc[[-1]].copy()

    dataset_config = deepcopy(config["dataset"]["passion"])
    label_col = dataset_config.get("label_col", "conditions")
    if str(label_col).upper() == "IMPETIGO":
        label_config = dataset_config["impetigo_labels"]
    else:
        label_config = dataset_config["condition_labels"]

    evaluator = BiasEvaluator(
        passion_exp=eval_csv.stem,
        eval_data_path=eval_csv.parent,
        dataset_dir=Path(dataset_config["path"]),
        meta_data_file=dataset_config["meta_data_file"],
        split_file=dataset_config["split_file"],
        target_names=label_config["target_names"],
        labels=label_config["labels"],
        evaluator_config=fairness_config,
        exclude_fitzpatrick_values=dataset_config.get("exclude_fitzpatrick_values"),
    )
    evaluator.run_full_evaluation(
        analysis_name="evaluator standalone",
        data=df_row,
        add_run_info=args.run_label,
        run_detailed_evaluation=True,
        detailed_evaluation_mode=fairness_config.get(
            "detailed_evaluation_mode",
            "reporting",
        ),
    )
