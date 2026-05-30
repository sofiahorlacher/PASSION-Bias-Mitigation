import argparse
import copy
from pathlib import Path

import yaml

from src.datasets.helper import DatasetName
from src.trainers.experiment_age_group_generalization import (
    ExperimentAgeGroupGeneralization,
)
from src.trainers.experiment_center_generalization import ExperimentCenterGeneralization
from src.trainers.experiment_standard_split import ExperimentStandardSplit
from src.trainers.experiment_stratified_validation_split import (
    ExperimentStratifiedValidationSplit,
)
from src.utils.loader import Loader
from src.utils.stratified_split_generator import StratifiedSplitGenerator

my_parser = argparse.ArgumentParser(description="Experiments for the PASSION paper.")
my_parser.add_argument(
    "--config_path",
    type=str,
    required=True,
    help="Path to the config yaml.",
)
my_parser.add_argument(
    "--exp1",
    action="store_true",
    help="If the experiment 1, i.e. differential diagnosis, should be run.",
)
my_parser.add_argument(
    "--exp2",
    action="store_true",
    help="If the experiment 2, i.e. detecting impetigo, should be run.",
)
my_parser.add_argument(
    "--exp3",
    action="store_true",
    help="If the experiment 3, i.e. generalization collection centers, should be run.",
)
my_parser.add_argument(
    "--exp4",
    action="store_true",
    help="If the experiment 4, i.e. generalization age groups, should be run.",
)
my_parser.add_argument(
    "--exp5",
    action="store_true",
    help="If the experiment 5 should be run - differential diagnosis with different split files (no folds, using validation set instead of test set).",
)
my_parser.add_argument(
    "--exp6",
    action="store_true",
    help="If the experiment 6 should be run - differential diagnosis with different split files (5-fold cross-validation, using validation set instead of test set).",
)
my_parser.add_argument(
    "--exp7",
    action="store_true",
    help="If the experiment 7 should be run - evaluate differential diagnosis models trained with exp5/6 on the original test set",
)
my_parser.add_argument(
    "--exp8",
    action="store_true",
    help="If the color jitter augmentation experiment should be run - differential diagnosis with color jitter augmentation as a bias mitigation strategy.",
)
my_parser.add_argument(
    "--exp9",
    action="store_true",
    help="If the instance reweighting experiment should be run - differential diagnosis with Kamiran-Calders instance reweighting as a bias mitigation strategy.",
)

my_parser.add_argument(
    "--append_results",
    action="store_true",
    help="If the results should be appended to the existing df (needs to be used with care!)",
)
my_parser.add_argument(
    "--split_ids",
    type=int,
    nargs="+",
    help=(
        "Optional 1-based split ids to run for exp5/exp6/exp7/exp8/exp9. "
        "Example: --split_ids 1 4 6"
    ),
)
args = my_parser.parse_args()


def select_requested_splits(splits, split_ids):
    """Return only the requested 1-based split ids while preserving order."""
    if not split_ids:
        return splits

    invalid_ids = [split_id for split_id in split_ids if split_id < 1 or split_id > len(splits)]
    if invalid_ids:
        raise ValueError(
            f"Invalid split ids {invalid_ids}. Valid range is 1 to {len(splits)}."
        )

    selected_indices = []
    for split_id in split_ids:
        split_index = split_id - 1
        if split_index not in selected_indices:
            selected_indices.append(split_index)

    return [splits[idx] for idx in selected_indices]


def select_default_validation_split(splits, split_ids):
    """Return the plain validation split by default, or explicit user selections."""
    if split_ids:
        return select_requested_splits(splits, split_ids)

    if not splits:
        raise ValueError("No generated splits available.")

    return [splits[0]]

if __name__ == "__main__":
    # load config yaml
    args.config_path = Path(args.config_path)
    if not args.config_path.exists():
        raise ValueError(f"Unable to find config yaml file: {args.config_path}")
    config = yaml.load(open(args.config_path, "r"), Loader=Loader)

    # overall parameters used for all datasets
    log_wandb = config.pop("log_wandb")
    model = config.pop("model")

    if "seeds" in config:
        seeds = config.pop("seeds")
    elif "seed" in config:
        seeds = [config.pop("seed")]
    else:
        seeds = [42]  # default seed if not specified

    print(f"Running experiments with seeds: {seeds}")

    for seed in seeds:
        # Create seed-specific config
        seed_config = copy.deepcopy(config)
        seed_config["seed"] = seed
        dataset_config = seed_config["dataset"]["passion"]

        if args.exp1:
            trainer = ExperimentStandardSplit(
                dataset_name=DatasetName.PASSION,
                config=seed_config,
                SSL_model=model,
                append_to_df=args.append_results,
                log_wandb=log_wandb,
                add_info="conditions",
            )
            trainer.evaluate()

        if args.exp2:
            _config = copy.deepcopy(seed_config)
            _config["dataset"]["passion"]["label_col"] = "IMPETIGO"
            trainer = ExperimentStandardSplit(
                dataset_name=DatasetName.PASSION,
                config=_config,
                SSL_model=model,
                append_to_df=args.append_results,
                log_wandb=log_wandb,
                add_info="impetigo",
            )
            trainer.evaluate()

        if args.exp3:
            trainer = ExperimentCenterGeneralization(
                dataset_name=DatasetName.PASSION,
                config=seed_config,
                SSL_model=model,
                append_to_df=args.append_results,
                log_wandb=log_wandb,
            )
            trainer.evaluate()

        if args.exp4:
            trainer = ExperimentAgeGroupGeneralization(
                dataset_name=DatasetName.PASSION,
                config=seed_config,
                SSL_model=model,
                append_to_df=args.append_results,
                log_wandb=log_wandb,
            )
            trainer.evaluate()

        if args.exp5 or args.exp6 or args.exp7 or args.exp8 or args.exp9:
            evaluator = StratifiedSplitGenerator(
                passion_exp=f"experiment_stratified_validation_split_conditions",
                eval_data_path="assets/evaluation",
                dataset_dir="data/PASSION",
                meta_data_file="label.csv",
                split_file="PASSION_split.csv",
                seed=seed,
                exclude_fitzpatrick_values=dataset_config.get(
                    "exclude_fitzpatrick_values"
                ),
            )
            splits = evaluator.create_stratified_splits()
            print(f"Created {len(splits)} splits: {splits}")
            splits = select_requested_splits(splits, args.split_ids)
            if args.split_ids:
                print(f"Running selected split ids {args.split_ids}: {splits}")

        if args.exp5:
            for split_name in splits:
                _config = copy.deepcopy(seed_config)
                _config["fine_tuning"]["n_folds"] = None
                _config["fine_tuning"]["train"] = True

                # Extract stratification part from filename: split_dataset__STRATIFICATION.csv -> STRATIFICATION
                stratify_str = split_name.replace("split_dataset__", "").replace(".csv", "")
                print(f"  Processing split: {stratify_str} (seed={seed})")
                _config["dataset"]["passion"]["split_file"] = split_name
                trainer = ExperimentStratifiedValidationSplit(
                    dataset_name=DatasetName.PASSION,
                    config=_config,
                    SSL_model=model,
                    append_to_df=args.append_results,
                    log_wandb=log_wandb,
                    add_info=f"conditions__{stratify_str}",
                )
                trainer.evaluate()

        if args.exp6:
            for split_name in splits:
                _config = copy.deepcopy(seed_config)
                _config["fine_tuning"]["n_folds"] = 5
                _config["fine_tuning"]["train"] = True
                stratify_str = split_name.replace("split_dataset__", "").replace(".csv", "")
                print(f"  Processing split: {stratify_str} (seed={seed})")
                _config["dataset"]["passion"]["split_file"] = split_name
                trainer = ExperimentStratifiedValidationSplit(
                    dataset_name=DatasetName.PASSION,
                    config=_config,
                    SSL_model=model,
                    append_to_df=args.append_results,
                    log_wandb=log_wandb,
                    add_info=f"conditions_5folds__{stratify_str}",
                )
                trainer.evaluate()

        if args.exp7:
            for split_name in splits:
                _config = copy.deepcopy(seed_config)
                _config["fine_tuning"]["n_folds"] = None
                _config["fine_tuning"]["train"] = False
                stratify_str = split_name.replace("split_dataset__", "").replace(".csv", "")
                print(f"  Evaluating with split: {stratify_str} (seed={seed})")
                trainer = ExperimentStandardSplit(
                    dataset_name=DatasetName.PASSION,
                    config=_config,
                    SSL_model=model,
                    model_path=(
                        "experiment_stratified_validation_split_conditions_5folds"
                        f"__{stratify_str}__passion__{model}"
                    ),
                    append_to_df=args.append_results,
                    log_wandb=log_wandb,
                    add_info=f"conditions__model_trained_with_{stratify_str}",
                )
                trainer.evaluate()

        if args.exp8:
            exp8_splits = select_default_validation_split(splits, args.split_ids)
            for split_name in exp8_splits:
                _config = copy.deepcopy(seed_config)
                #_config["fine_tuning"]["n_folds"] = 5
                _config["fine_tuning"]["n_folds"] = None
                _config["fine_tuning"]["train"] = True
                underrepresented_group_columns = ["fitzpatrick"]
                if isinstance(underrepresented_group_columns, str):
                    subgroup_label = underrepresented_group_columns
                else:
                    subgroup_label = "_".join(underrepresented_group_columns)

                stratify_str = split_name.replace("split_dataset__", "").replace(".csv", "")
                print(f"  Processing split: {stratify_str} (seed={seed})")
                _config["dataset"]["passion"]["split_file"] = split_name
                _config["fine_tuning"].update(
                    {
                        "color_jitter_implementation": "paper_plain",
                        "enable_data_balancing": True,
                        "underrepresented_group_columns": underrepresented_group_columns,
                        "balance_target": "reference_group",
                        "balance_target_group_value": 4,
                    }
                )
                trainer = ExperimentStratifiedValidationSplit(
                    dataset_name=DatasetName.PASSION,
                    config=_config,
                    SSL_model=model,
                    append_to_df=args.append_results,
                    log_wandb=log_wandb,
                    add_info=(
                        f"conditions_color_jitter_oversampled__{subgroup_label}"
                        f"__{stratify_str}"
                    ),
                )
                trainer.evaluate()

        if args.exp9:
            exp9_splits = select_default_validation_split(splits, args.split_ids)
            for split_name in exp9_splits:
                _config = copy.deepcopy(seed_config)
                #_config["fine_tuning"]["n_folds"] = 5
                _config["fine_tuning"]["n_folds"] = None
                _config["fine_tuning"]["train"] = True
                instance_reweighting_columns = ["fitzpatrick"]
                subgroup_label = "_".join(instance_reweighting_columns)

                stratify_str = split_name.replace("split_dataset__", "").replace(".csv", "")
                print(f"  Processing split: {stratify_str} (seed={seed})")
                _config["dataset"]["passion"]["split_file"] = split_name

                _config["fine_tuning"].update(
                    {
                        "enable_instance_reweighting": True,
                        "instance_reweighting_columns": instance_reweighting_columns,
                        "disable_class_weights": False,
                        "learning_rate": 1.35E-04,
                        "find_optimal_lr": False,
                    }
                )
                trainer = ExperimentStratifiedValidationSplit(
                    dataset_name=DatasetName.PASSION,
                    config=_config,
                    SSL_model=model,
                    append_to_df=args.append_results,
                    log_wandb=log_wandb,
                    add_info=(
                        f"conditions_instance_reweighting__{subgroup_label}"
                        f"__{stratify_str}"
                    ),
                )
                trainer.evaluate()
