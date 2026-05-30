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
    help=(
        "Deprecated for the fold-only validation workflow. "
        "Originally evaluated models trained with exp5/6 on the original test set."
    ),
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
    "--exp10",
    action="store_true",
    help="If the Group DRO experiment should be run - differential diagnosis with Group DRO over Fitzpatrick groups.",
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
        "Optional 1-based split ids to run for exp5/exp6/exp7/exp8/exp9/exp10. "
        "Example: --split_ids 1 4 6"
    ),
)
my_parser.add_argument(
    "--mitigation_strengths",
    type=float,
    nargs="+",
    default=[1 / 3, 2 / 3, 1.0],
    help=(
        "Strength sweep for mitigation experiments. "
        "Use values in [0, 1], where 0 is baseline and 1 is full mitigation."
    ),
)
my_parser.add_argument(
    "--mitigation_n_folds",
    type=int,
    default=5,
    help=(
        "Number of CV folds for mitigation sweeps. "
        "Set to 5 by default; reduce to 3 if needed. "
        "Use 0 for a quick no-fold tuning run."
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


def format_strength_label(strength: float) -> str:
    return f"{float(strength):.2f}".replace(".", "p")


def mitigation_fold_count_to_config(n_folds: int):
    return None if n_folds <= 0 else n_folds


def mitigation_fold_tag(n_folds: int) -> str:
    return "nofolds" if n_folds <= 0 else f"{n_folds}folds"


def set_stratified_split_protocol(
    config: dict,
    train_splits: list[str],
    evaluation_split: str,
    pool_for_cross_validation: bool = False,
):
    """Configure which predefined split labels feed training and evaluation."""
    config["stratified_validation_split"] = {
        "train_splits": train_splits,
        "evaluation_split": evaluation_split,
        "pool_for_cross_validation": pool_for_cross_validation,
    }

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

        if args.exp5 or args.exp6 or args.exp7 or args.exp8 or args.exp9 or args.exp10:
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
                set_stratified_split_protocol(
                    _config,
                    train_splits=["TRAIN"],
                    evaluation_split="VALIDATION",
                )

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
                _config["fine_tuning"]["eval_test_performance"] = False
                set_stratified_split_protocol(
                    _config,
                    train_splits=["TRAIN", "VALIDATION"],
                    evaluation_split="VALIDATION",
                    pool_for_cross_validation=True,
                )
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
            continue
            # With new folding splits this doesn't work anymore.
            # It assumes exp6 produces a final train-all checkpoint that can be
            # evaluated on the original TEST split.
            #
            # for split_name in splits:
            #     _config = copy.deepcopy(seed_config)
            #     _config["fine_tuning"]["n_folds"] = None
            #     _config["fine_tuning"]["train"] = False
            #     stratify_str = split_name.replace("split_dataset__", "").replace(".csv", "")
            #     print(f"  Evaluating with split: {stratify_str} (seed={seed})")
            #     trainer = ExperimentStandardSplit(
            #         dataset_name=DatasetName.PASSION,
            #         config=_config,
            #         SSL_model=model,
            #         model_path=(
            #             "experiment_stratified_validation_split_conditions_5folds"
            #             f"__{stratify_str}__passion__{model}"
            #         ),
            #         append_to_df=args.append_results,
            #         log_wandb=log_wandb,
            #         add_info=f"conditions__model_trained_with_{stratify_str}",
            #     )
            #     trainer.evaluate()

        if args.exp8:
            exp8_splits = select_default_validation_split(splits, args.split_ids)
            for split_name in exp8_splits:
                stratify_str = split_name.replace("split_dataset__", "").replace(".csv", "")
                underrepresented_group_columns = ["fitzpatrick"]
                if isinstance(underrepresented_group_columns, str):
                    subgroup_label = underrepresented_group_columns
                else:
                    subgroup_label = "_".join(underrepresented_group_columns)

                for strength in args.mitigation_strengths:
                    strength_label = format_strength_label(strength)
                    print(
                        f"  Processing split: {stratify_str} "
                        f"(seed={seed}, color_jitter_strength={strength:.2f}, folds={args.mitigation_n_folds})"
                    )
                    _config = copy.deepcopy(seed_config)
                    _config["fine_tuning"]["n_folds"] = args.mitigation_n_folds
                    _config["fine_tuning"]["train"] = True
                    _config["dataset"]["passion"]["split_file"] = split_name
                    if args.mitigation_n_folds > 0:
                        _config["fine_tuning"]["eval_test_performance"] = False
                        set_stratified_split_protocol(
                            _config,
                            train_splits=["TRAIN", "VALIDATION"],
                            evaluation_split="VALIDATION",
                            pool_for_cross_validation=True,
                        )
                    else:
                        set_stratified_split_protocol(
                            _config,
                            train_splits=["TRAIN"],
                            evaluation_split="VALIDATION",
                        )
                    _config["fine_tuning"].update(
                        {
                            "color_jitter_implementation": "paper_plain",
                            "enable_data_balancing": True,
                            "underrepresented_group_columns": underrepresented_group_columns,
                            "balance_target": "reference_group",
                            "balance_target_group_value": 4,
                            "data_balancing_strength": strength,
                        }
                    )
                    trainer = ExperimentStratifiedValidationSplit(
                        dataset_name=DatasetName.PASSION,
                        config=_config,
                        SSL_model=model,
                        append_to_df=args.append_results,
                        log_wandb=log_wandb,
                        add_info=(
                            f"conditions_color_jitter_oversampled_{args.mitigation_n_folds}folds__{subgroup_label}"
                            f"__strength_{strength_label}__{stratify_str}"
                        ),
                    )
                    trainer.evaluate()

        if args.exp9:
            exp9_splits = select_default_validation_split(splits, args.split_ids)
            for split_name in exp9_splits:
                instance_reweighting_columns = ["fitzpatrick"]
                subgroup_label = "_".join(instance_reweighting_columns)

                stratify_str = split_name.replace("split_dataset__", "").replace(".csv", "")
                for strength in args.mitigation_strengths:
                    strength_label = format_strength_label(strength)
                    print(
                        f"  Processing split: {stratify_str} "
                        f"(seed={seed}, reweighting_strength={strength:.2f}, folds={args.mitigation_n_folds})"
                    )
                    _config = copy.deepcopy(seed_config)
                    _config["fine_tuning"]["n_folds"] = args.mitigation_n_folds
                    _config["fine_tuning"]["train"] = True
                    _config["dataset"]["passion"]["split_file"] = split_name
                    if args.mitigation_n_folds > 0:
                        _config["fine_tuning"]["eval_test_performance"] = False
                        set_stratified_split_protocol(
                            _config,
                            train_splits=["TRAIN", "VALIDATION"],
                            evaluation_split="VALIDATION",
                            pool_for_cross_validation=True,
                        )
                    else:
                        set_stratified_split_protocol(
                            _config,
                            train_splits=["TRAIN"],
                            evaluation_split="VALIDATION",
                        )

                    _config["fine_tuning"].update(
                        {
                            "enable_instance_reweighting": True,
                            "instance_reweighting_columns": instance_reweighting_columns,
                            "instance_reweighting_strength": strength,
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
                            f"conditions_instance_reweighting_{args.mitigation_n_folds}folds__{subgroup_label}"
                            f"__strength_{strength_label}__{stratify_str}"
                        ),
                    )
                    trainer.evaluate()

        if args.exp10:
            exp10_splits = select_default_validation_split(splits, args.split_ids)
            exp10_n_folds = mitigation_fold_count_to_config(args.mitigation_n_folds)
            exp10_fold_tag = mitigation_fold_tag(args.mitigation_n_folds)
            for split_name in exp10_splits:
                stratify_str = split_name.replace("split_dataset__", "").replace(".csv", "")
                for strength in args.mitigation_strengths:
                    strength_label = format_strength_label(strength)
                    print(
                        f"  Processing split: {stratify_str} "
                        f"(seed={seed}, group_dro_strength={strength:.2f}, folds={exp10_fold_tag})"
                    )
                    _config = copy.deepcopy(seed_config)
                    _config["fine_tuning"]["n_folds"] = exp10_n_folds
                    _config["fine_tuning"]["train"] = True
                    _config["dataset"]["passion"]["split_file"] = split_name
                    if exp10_n_folds is not None:
                        _config["fine_tuning"]["eval_test_performance"] = False
                        set_stratified_split_protocol(
                            _config,
                            train_splits=["TRAIN", "VALIDATION"],
                            evaluation_split="VALIDATION",
                            pool_for_cross_validation=True,
                        )
                    else:
                        set_stratified_split_protocol(
                            _config,
                            train_splits=["TRAIN"],
                            evaluation_split="VALIDATION",
                        )
                    group_dro_columns = _config["fine_tuning"].get(
                        "group_dro_columns",
                        ["fitzpatrick"],
                    )
                    base_group_dro_adjustment = float(
                        _config["fine_tuning"].get("group_dro_adjustment", 0.0)
                    )
                    if isinstance(group_dro_columns, str):
                        subgroup_label = group_dro_columns
                    else:
                        subgroup_label = "_".join(group_dro_columns)
                    _config["fine_tuning"].update(
                        {
                            "enable_group_dro": True,
                            "group_dro_columns": group_dro_columns,
                            "group_dro_adjustment": base_group_dro_adjustment,
                            "group_dro_strength": strength,
                            "learning_rate": 1.0e-05,
                            "weight_decay": 1.0e-04,
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
                            f"conditions_group_dro_{exp10_fold_tag}__{subgroup_label}"
                            f"__strength_{strength_label}__{stratify_str}"
                        ),
                    )
                    trainer.evaluate()
