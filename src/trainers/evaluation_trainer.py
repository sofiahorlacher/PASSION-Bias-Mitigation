import copy
import pickle
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import wandb
from loguru import logger
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torchvision import transforms
from tqdm import tqdm

from src.datasets.helper import DatasetName, get_dataset
from src.models.embedder import Embedder
from src.models.helper import embed_dataset
from src.trainers.eval_types.base import BaseEvalType
from src.trainers.eval_types.dummy_classifier import (
    EvalDummyConstant,
    EvalDummyMostFrequent,
    EvalDummyUniform,
)
from src.trainers.eval_types.fine_tuning import EvalFineTuning
from src.trainers.eval_types.knn import EvalKNN
from src.trainers.eval_types.lin import EvalLin
from src.utils.evaluator import BiasEvaluator
from src.utils.utils import fix_random_seeds

eval_type_dict = {
    # Baselines
    "dummy_most_frequent": EvalDummyMostFrequent,
    "dummy_uniform": EvalDummyUniform,
    "dummy_constant": EvalDummyConstant,
    # Models
    "fine_tuning": EvalFineTuning,
    # "kNN": EvalKNN,
    # "lin": EvalLin,
}


class EvaluationTrainer(ABC, object):
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
    ):
        self.dataset_name = dataset_name
        self.config = config
        self.SSL_model = SSL_model 
        logger.debug(f"config: {self.config}")
        self.cache_path = Path(cache_path)
        self.append_to_df = append_to_df
        self.wandb_project_name = wandb_project_name
        self.log_wandb = log_wandb
        self.seed = config["seed"]
        fix_random_seeds(self.seed)
        self.output_path = Path(output_path) / f"seed_{self.seed}"

        self.df_description = (
            f"{self.experiment_name}__{self.dataset_name.value}__{SSL_model}"
        )
        self.df_name = f"{self.df_description}.csv"
        self.df_path = self.output_path / self.df_name

        self.model_path = self.output_path / self.df_description
        if model_path:
            self.model_path = self.output_path / model_path

        # make sure the output and cache path exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.model_path.mkdir(parents=True, exist_ok=True)

        # parse the config to get the eval types
        self.eval_types = []
        for k, v in self.config.items():
            if k in eval_type_dict.keys():
                self.eval_types.append((eval_type_dict.get(k), v))

        # save the results to the dataframe
        self.df = pd.DataFrame(
            [],
            columns=[
                "Score",
                "FileNames",
                "Indices",
                "EvalTargets",
                "EvalPredictions",
                "EvalType",
                "AdditionalRunInfo",
                "SplitName",
            ],
        )
        if append_to_df:
            if not self.df_path.exists():
                print(f"Results for dataset: {self.dataset_name.value} not available.")
            else:
                print(f"Appending results to: {self.df_path}")
                self.df = pd.read_csv(self.df_path)

        # load the dataset to evaluate on
        self.input_size = config["input_size"]
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        data_config = copy.deepcopy(config["dataset"])[dataset_name.value]
        data_path = data_config.pop("path")
        data_label_config = (
            data_config.get("impetigo_labels")
            if data_config.get("label_col") == "IMPETIGO"
            else data_config.get("condition_labels")
        )
        print(f"data_label_config: {data_label_config}")
        self.evaluator = BiasEvaluator(
            passion_exp=self.df_description,
            eval_data_path=self.output_path,
            dataset_dir=Path(data_path),
            meta_data_file=data_config["meta_data_file"],
            split_file=data_config["split_file"],
            target_names=data_label_config["target_names"],
            labels=data_label_config["labels"],
        )

        self.dataset, self.torch_dataset = get_dataset(
            dataset_name=dataset_name,
            dataset_path=Path(data_path),
            batch_size=config.get("batch_size", 128),
            transform=self.transform,
            # num_workers=config.get("num_workers", 4),
            **data_config,
        )

        # load the correct model to use as initialization
        if SSL_model == "GoogleDermFound":
            self.dataset.return_embedding = True
            self.torch_dataset.dataset.return_embedding = True
        else:
            self.model, self.model_out_dim = self.load_model(SSL_model=SSL_model)
        #
        # check if the cache contains the embeddings already
        logger.debug("embed data")
        cache_names = [self.experiment_name]
        if "_5folds__" in self.experiment_name:
            cache_names.append(self.experiment_name.replace("_5folds__", "__", 1))

        cache_file = None
        for cache_name in cache_names:
            candidate = self.cache_path / f"{dataset_name.value}_{cache_name}_{SSL_model}.pickle"
            if candidate.exists():
                cache_file = candidate
                break

        if cache_file is not None:
            print(f"Found cached file loading: {cache_file}")
            with open(cache_file, "rb") as file:
                cached_dict = pickle.load(file)
            self.emb_space = cached_dict["embedding_space"]
            self.labels = cached_dict["labels"]
            # TODO: figure out where this is reloaded / should be used
            self.paths = cached_dict["paths"]
            self.indices = cached_dict["indices"]
            del cached_dict
        else:
            cache_file = (
                self.cache_path / f"{dataset_name.value}_{self.experiment_name}_{SSL_model}.pickle"
            )
            (
                self.emb_space,
                self.labels,
                self.images,
                self.paths,
                self.indices,
            ) = embed_dataset(
                torch_dataset=self.torch_dataset,
                model=self.model,
                n_layers=n_layers,
                normalize=False,
            )
            # save the embeddings and issues to cache
            save_dict = {
                "embedding_space": self.emb_space,
                "labels": self.labels,
                "paths": self.paths,
                "indices": self.indices,
            }
            with open(cache_file, "wb") as handle:
                pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @property
    @abstractmethod
    def experiment_name(self) -> str:
        pass

    @abstractmethod
    def split_dataframe_iterator(self) -> Iterator[Tuple[np.ndarray, np.ndarray, str]]:
        pass

    def load_model(self, SSL_model: str):
        logger.debug("load model")
        model, info, _ = Embedder.load_pretrained(
            SSL_model,
            return_info=True,
            n_head_layers=0,
        )
        # set the model in evaluation mode
        model = model.eval()
        # move to correct device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"device: {device}")
        model = model.to(device)
        return model, info.out_dim

    def evaluate(self):
        if self.df_path.exists() and not self.append_to_df:
            raise ValueError(
                "Dataframe already exists. "
                f"Use --append_results to resume on existing outputs: {self.df_path}. "
                "If you want a fresh run, remove this file (or the whole seed output folder) first."
            )

        for e_type, config in self.eval_types:
            for (
                train_valid_range,
                test_range,
                split_name,
            ) in self.split_dataframe_iterator():
                if (
                    config.get("train", True)
                    and config.get("n_folds", None) is not None
                ):
                    k_fold = StratifiedGroupKFold(
                        n_splits=config["n_folds"],
                        random_state=self.seed,
                        shuffle=True,
                    )
                    labels = self.dataset.meta_data.loc[
                        train_valid_range, self.dataset.LBL_COL
                    ].values
                    groups = self.dataset.meta_data.loc[
                        train_valid_range, "subject_id"
                    ].values
                    fold_generator = k_fold.split(train_valid_range, labels, groups)
                    for i_fold, (train_range, valid_range) in tqdm(
                        enumerate(fold_generator),
                        total=config["n_folds"],
                        desc="K-Folds",
                    ):
                        add_run_info = f"Fold-{i_fold}"
                        # Skip already completed folds
                        if self._is_fold_completed(
                            e_type.name(), split_name, add_run_info
                        ):
                            logger.info(
                                f"{add_run_info} already completed for "
                                f"{split_name}/{e_type.name()}, skipping."
                            )
                            continue

                        fold_checkpoint_dir = self.model_path / f"fold_{i_fold}_training"
                        self._run_evaluation_on_range(
                            e_type=e_type,
                            train_range=train_range,
                            eval_range=valid_range,
                            config=config,
                            add_run_info=add_run_info,
                            split_name=split_name,
                            saved_model_path=None,
                            detailed_evaluation=True,
                            checkpoint_dir=fold_checkpoint_dir,
                        )
                if config["eval_test_performance"]:
                    add_run_info = "Test"
                    if not self._is_fold_completed(
                        e_type.name(), split_name, add_run_info
                    ):
                        test_checkpoint_dir = self.model_path / "test_training"
                        self._run_evaluation_on_range(
                            e_type=e_type,
                            train_range=train_valid_range,
                            eval_range=test_range,
                            config=config,
                            add_run_info=add_run_info,
                            split_name=split_name,
                            saved_model_path=self.model_path,
                            detailed_evaluation=True,
                            checkpoint_dir=test_checkpoint_dir,
                        )
                    else:
                        logger.info(
                            f"{add_run_info} already completed for "
                            f"{split_name}/{e_type.name()}, skipping."
                        )

    def _run_evaluation_on_range(
        self,
        e_type: BaseEvalType,
        train_range: np.ndarray,
        eval_range: np.ndarray,
        config: dict,
        add_run_info: Optional[str] = None,
        split_name: Optional[str] = None,
        saved_model_path: Union[Path, str, None] = None,
        detailed_evaluation: bool = False,
        checkpoint_dir: Union[Path, str, None] = None,
    ):
        if checkpoint_dir is not None and e_type is EvalFineTuning:
            ckpt_file = Path(checkpoint_dir) / "training_checkpoint.pth"
            logger.info(
                f"Checkpoint file for {add_run_info}: {ckpt_file} "
                f"(exists={ckpt_file.exists()})"
            )

        # Check if there's a wandb run ID to resume from a training checkpoint
        resume_wandb_run_id = None
        if checkpoint_dir is not None and e_type is EvalFineTuning:
            resume_wandb_run_id = EvalFineTuning.get_wandb_run_id_from_checkpoint(
                checkpoint_dir
            )

        self.configure_wandb(
            add_run_info, e_type, split_name,
            resume_run_id=resume_wandb_run_id,
        )

        current_wandb_run_id = None
        if e_type is EvalFineTuning and self.log_wandb and wandb.run:
            current_wandb_run_id = wandb.run.id

        # get train / test set
        score_dict = e_type.evaluate(
            emb_space=self.emb_space,
            labels=self.labels,
            train_range=train_range,
            evaluation_range=eval_range,
            # only needed for fine-tuning
            dataset=self.dataset,
            model=self.model,
            model_out_dim=self.model_out_dim,
            log_wandb=self.log_wandb,
            saved_model_path=saved_model_path,
            seed=self.seed,
            checkpoint_dir=checkpoint_dir,
            wandb_run_id=current_wandb_run_id,
            # rest of the method specific parameters set with kwargs
            **config,
        )
        # save the results to the overall dataframe + save df
        self.df.loc[len(self.df)] = list(score_dict.values()) + [
            e_type.name(),
            add_run_info,
            split_name,
        ]
        self.df.to_csv(self.df_path, index=False)
        if detailed_evaluation:
            # Detailed evaluation
            # TODO: remove this backup analysis once binary eval is supported in the evaluator too
            print("*" * 20 + f" {e_type.name()} " + "*" * 20)
            self.print_eval_scores(
                y_true=score_dict["targets"],
                y_pred=score_dict["predictions"],
            )

            run_detailed_evaluation = config.get("detailed_evaluation", False)
            if run_detailed_evaluation:
                self.evaluator.run_full_evaluation(
                    e_type.name(),
                    self.df.iloc[[-1]],
                    add_run_info=add_run_info,
                    run_detailed_evaluation=run_detailed_evaluation,
                )

        self.finish_wandb(e_type)

    def finish_wandb(self, e_type):
        # finish the W&B run if needed
        if e_type is EvalFineTuning and self.log_wandb:
            wandb.finish()

    def _is_fold_completed(self, e_type_name, split_name, add_run_info):
        """Check if a fold/test run is already completed and present in the DataFrame."""
        if self.df.empty:
            return False
        mask = self.df["AdditionalRunInfo"] == add_run_info
        if e_type_name is not None:
            mask = mask & (self.df["EvalType"] == e_type_name)
        if split_name is not None:
            mask = mask & (self.df["SplitName"] == split_name)
        return mask.any()

    def configure_wandb(self, add_run_info, e_type, split_name, resume_run_id=None):
        # W&B configurations
        if e_type is EvalFineTuning and self.log_wandb:
            _config = copy.deepcopy(self.config)
            if split_name is not None:
                _config["split_name"] = split_name
            _config["SSL_model"] = self.SSL_model

            init_kwargs = {
                "config": _config,
                "project": self.wandb_project_name,
            }
            if resume_run_id is not None:
                init_kwargs["id"] = resume_run_id
                init_kwargs["resume"] = "must"
                logger.info(
                    f"Resuming wandb run {resume_run_id}"
                )

            wandb.init(**init_kwargs)
            wandb_run_name = f"{self.experiment_name}-{self.SSL_model}-{wandb.run.name}"
            if add_run_info is not None:
                wandb_run_name += f"-{add_run_info}"
            wandb.run.name = wandb_run_name

    def print_eval_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        if len(self.dataset.classes) == 2:
            f1 = f1_score(
                y_true=y_true,
                y_pred=y_pred,
                pos_label=1,
                average="binary",
            )
            precision = precision_score(
                y_true=y_true,
                y_pred=y_pred,
                pos_label=1,
                average="binary",
            )
            recall = recall_score(
                y_true=y_true,
                y_pred=y_pred,
                pos_label=1,
                average="binary",
            )
            print(f"Bin. F1: {f1:.2f}")
            print(f"Bin. Precision: {precision:.2f}")
            print(f"Bin. Recall: {recall:.2f}")
        else:
            try:
                print(
                    classification_report(
                        y_true=y_true,
                        y_pred=y_pred,
                        target_names=self.dataset.classes,
                    )
                )
            except Exception as e:
                print(f"Error generating classification report: {e}")
            b_acc = balanced_accuracy_score(
                y_true=y_true,
                y_pred=y_pred,
            )
            print(f"Balanced Acc: {b_acc}")
