from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import numpy as np

from src.datasets.helper import DatasetName
from src.trainers.evaluation_trainer import EvaluationTrainer


class ExperimentStandardSplit(EvaluationTrainer):
    def __init__(
        self,
        dataset_name: DatasetName,
        config: dict,
        SSL_model: str = "imagenet",
        output_path: Union[Path, str] = "assets/evaluation",
        cache_path: Union[Path, str] = "assets/evaluation/cache",
        model_path: Union[Path, str] = None,
        n_layers: int = 1,
        append_to_df: bool = False,
        wandb_project_name: str = "PASSION-Evaluation",
        log_wandb: bool = False,
        add_info: Optional[str] = None,
    ):
        self.add_info = add_info
        super().__init__(
            dataset_name=dataset_name,
            config=config,
            SSL_model=SSL_model,
            output_path=output_path,
            cache_path=cache_path,
            model_path=model_path,
            n_layers=n_layers,
            append_to_df=append_to_df,
            wandb_project_name=wandb_project_name,
            log_wandb=log_wandb,
        )

    @property
    def experiment_name(self) -> str:
        if self.add_info is not None:
            return f"experiment_standard_split_{self.add_info}"
        else:
            return "experiment_standard_split"

    def split_dataframe_iterator(self) -> Iterator[Tuple[np.ndarray, np.ndarray, str]]:
        split_config = self.config.get("standard_split_protocol", {})
        train_splits = split_config.get("train_splits", ["TRAIN"])
        evaluation_split = split_config.get("evaluation_split", "TEST")

        train_valid_range = self.dataset.meta_data[
            self.dataset.meta_data["Split"].isin(train_splits)
        ].index.values
        test_range = self.dataset.meta_data[
            self.dataset.meta_data["Split"] == evaluation_split
        ].index.values

        if len(train_valid_range) == 0:
            raise ValueError(
                f"No samples found for training splits {train_splits}."
            )
        if len(test_range) == 0:
            raise ValueError(
                f"No samples found for evaluation split '{evaluation_split}'."
            )

        train_split_label = "_".join(train_splits)
        split_name = f"Standard_{train_split_label}_TO_{evaluation_split}"
        yield train_valid_range, test_range, split_name
