import copy
import functools
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchmetrics
import wandb
from loguru import logger
from sklearn.metrics import f1_score as sk_f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch_lr_finder import LRFinder, TrainDataLoaderIter
from torchinfo import summary
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from src.datasets.offline_augmentation import generate_offline_augmented_rows
from src.models.classifiers import LinearClassifier
from src.optimizers.utils import get_optimizer_type
from src.trainers.group_dro import GroupDROLossComputer
from src.trainers.eval_types.base import BaseEvalType
from src.utils.utils import (
    EarlyStopping,
    fix_random_seeds,
    restart_from_checkpoint,
    save_checkpoint,
    set_requires_grad
)


class EvalFineTuning(BaseEvalType):
    SAMPLE_WEIGHT_COL = "sample_weight"
    GROUP_DRO_INDEX_COL = "group_dro_group_index"

    @classmethod
    def train_transform(cls):
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(cls.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                # Experiments showed color jitter hinders performance,
                # but check again if problems with models and datasets arise
                # transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    @classmethod
    def val_transform(cls):
        return transforms.Compose(
            [
                transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(cls.input_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    @staticmethod
    def name() -> str:
        return "finetuning"

    @staticmethod
    def _seed_worker(worker_id: int, seed: int) -> None:
        torch.manual_seed(seed + worker_id)
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    @classmethod
    def evaluate(
        cls,
        train_range: np.ndarray,
        evaluation_range: np.ndarray,
        dataset: torch.utils.data.Dataset,
        model: Optional[torch.nn.Module],
        model_out_dim: int,
        learning_rate: float,
        batch_size: int,
        input_size: int,
        train_epochs: int,
        warmup_epochs: int,
        early_stopping_patience: int,
        use_bn_in_head: bool,
        dropout_in_head: float,
        num_workers: int,
        seed: int = 42,
        saved_model_path: Union[Path, str, None] = None,
        find_optimal_lr: bool = False,
        use_lr_scheduler: bool = False,
        log_wandb: bool = False,
        debug: bool = False,
        train: bool = True,
        checkpoint_dir: Union[Path, str, None] = None,
        wandb_run_id: str = None,
        color_jitter_brightness: float = None,
        color_jitter_contrast: float = None,
        color_jitter_saturation: float = None,
        color_jitter_hue: float = None,
        color_jitter_implementation: str = "torchvision",
        enable_data_balancing: bool = False,
        underrepresented_group_columns: Optional[Union[str, list]] = None,
        balance_target: str = "median",
        balance_target_group_value: Optional[Union[str, int, float]] = None,
        data_balancing_strength: float = 1.0,
        enable_instance_reweighting: bool = False,
        instance_reweighting_columns: Optional[Union[str, list]] = None,
        instance_reweighting_strength: float = 1.0,
        enable_group_dro: bool = False,
        group_dro_columns: Optional[Union[str, list]] = None,
        group_dro_include_label: bool = False,
        group_dro_step_size: float = 0.01,
        group_dro_adjustment: float = 0.0,
        group_dro_normalize_loss: bool = False,
        group_dro_strength: float = 1.0,
        weight_decay: float = 0.0,
        disable_class_weights: bool = False,
        **kwargs,
    ) -> dict:
        cls.input_size = input_size
        fix_random_seeds(seed)
        device = cls.get_device(model)

        offline_augmentation_dir = None
        if checkpoint_dir is not None:
            offline_augmentation_dir = Path(checkpoint_dir) / "offline_augmented_train"
        elif saved_model_path is not None:
            offline_augmentation_dir = Path(saved_model_path) / "offline_augmented_train"

        cls.color_jitter_implementation = color_jitter_implementation

        # Configure color jitter if parameters are provided
        if (
            any(
                x is not None
                for x in [
                    color_jitter_brightness,
                    color_jitter_contrast,
                    color_jitter_saturation,
                    color_jitter_hue,
                ]
            )
            or color_jitter_implementation == "paper_plain"
        ):
            cls.enable_color_jitter = True
            cls.color_jitter_brightness = color_jitter_brightness or 0.12
            cls.color_jitter_contrast = color_jitter_contrast or 0.12
            cls.color_jitter_saturation = color_jitter_saturation or 0.08
            cls.color_jitter_hue = color_jitter_hue or 0.01
            logger.debug(
                f"Color jitter augmentation enabled: "
                f"implementation={cls.color_jitter_implementation}, "
                f"brightness={cls.color_jitter_brightness}, "
                f"contrast={cls.color_jitter_contrast}, "
                f"saturation={cls.color_jitter_saturation}, "
                f"hue={cls.color_jitter_hue}"
            )
        else:
            cls.enable_color_jitter = False

        data_balancing_strength = cls.normalize_mitigation_strength(
            data_balancing_strength
        )
        instance_reweighting_strength = cls.normalize_mitigation_strength(
            instance_reweighting_strength
        )
        group_dro_strength = cls.normalize_mitigation_strength(group_dro_strength)

        balanced_train_indices = np.array(train_range, copy=True)
        offline_augmented_rows = dataset.meta_data.iloc[0:0].copy()
        if (
            train
            and enable_data_balancing
            and cls.enable_color_jitter
            and underrepresented_group_columns is not None
            and offline_augmentation_dir is not None
        ):
            balanced_train_indices, offline_augmented_rows = generate_offline_augmented_rows(
                dataset=dataset,
                train_range=train_range,
                underrepresented_group_columns=underrepresented_group_columns,
                output_dir=offline_augmentation_dir,
                balance_target=balance_target,
                balance_target_group_value=balance_target_group_value,
                balance_strength=data_balancing_strength,
                brightness=cls.color_jitter_brightness,
                contrast=cls.color_jitter_contrast,
                saturation=cls.color_jitter_saturation,
                hue=cls.color_jitter_hue,
                jitter_implementation=cls.color_jitter_implementation,
                seed=seed,
                debug=debug,
            )

        legacy_enable_instance_reweighting = kwargs.pop(
            "enable_preprocessing_reweighing_sampler",
            False,
        )
        legacy_instance_reweighting_columns = kwargs.pop(
            "preprocessing_reweighing_columns",
            None,
        )
        if legacy_enable_instance_reweighting:
            enable_instance_reweighting = True
        if (
            instance_reweighting_columns is None
            and legacy_instance_reweighting_columns is not None
        ):
            instance_reweighting_columns = legacy_instance_reweighting_columns

        sample_reweighting = None
        if train and enable_instance_reweighting:
            sample_reweighting = cls.compute_sample_reweighting(
                dataset=dataset,
                train_range=train_range,
                sensitive_columns=instance_reweighting_columns,
                strength=instance_reweighting_strength,
            )

        if enable_group_dro and sample_reweighting is not None:
            raise ValueError(
                "Group DRO is currently not supported together with instance reweighting."
            )
        if enable_group_dro and len(offline_augmented_rows) > 0:
            raise ValueError(
                "Group DRO is currently not supported together with offline augmentation."
            )

        group_dro_train = None
        group_dro_eval = None
        if train and enable_group_dro:
            group_dro_train = cls.compute_group_dro_metadata(
                dataset=dataset,
                selected_range=train_range,
                group_columns=group_dro_columns,
                include_label=group_dro_include_label,
                generalization_adjustment=group_dro_adjustment,
            )
            group_dro_eval = cls.compute_group_dro_metadata(
                dataset=dataset,
                selected_range=evaluation_range,
                group_columns=group_dro_columns,
                include_label=group_dro_include_label,
                generalization_adjustment=0.0,
            )

        # get dataloader for batched compute
        train_loader, eval_loader = cls.get_train_eval_loaders(
            dataset=dataset,
            train_range=train_range,
            evaluation_range=evaluation_range,
            balanced_train_indices=balanced_train_indices,
            offline_augmented_rows=offline_augmented_rows,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            sample_reweighting=sample_reweighting,
            group_dro=group_dro_train,
        )

        if train is True:
            classifier, model = cls.create_classifier(
                dataset, dropout_in_head, model, model_out_dim, use_bn_in_head
            )
            classifier.to(device)
            cls.configure_classifier_base(
                classifier, debug, log_wandb, model, train_loader
            )
            # loss function, optimizer, scores
            if disable_class_weights:
                criterion = torch.nn.CrossEntropyLoss()
            else:
                criterion = torch.nn.CrossEntropyLoss(
                    weight=train_loader.dataset.get_class_weights(),
                )
            criterion = criterion.to(device)
            class_weights = criterion.weight
            optimizer = cls.configure_optimizer(
                classifier,
                criterion,
                device,
                find_optimal_lr,
                learning_rate,
                weight_decay,
                log_wandb,
                train_loader,
                seed,
                sample_reweighting=sample_reweighting,
                group_dro=group_dro_train,
                group_dro_step_size=group_dro_step_size,
                group_dro_normalize_loss=group_dro_normalize_loss,
                group_dro_strength=group_dro_strength,
                class_weights=class_weights,
            )

            # we use early stopping to speed up the training
            early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                log_messages=debug,
            )

            scheduler = None
            if use_lr_scheduler:
                # define the learning rate scheduler
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=train_epochs,
                    eta_min=0,
                )

        # load the model from checkpoint if provided
        # TODO: fix whole reloading, including the epoch restoration
        if train is False:
            if saved_model_path is not None:
                # Path to the checkpoint file
                checkpoint_path = saved_model_path / "checkpoints" / "model_best.pth"
                checkpoint = torch.load(checkpoint_path, weights_only=False)
                # Restore model
                classifier = checkpoint["classifier"]
                classifier.to(device)
                cls.print_model(classifier, debug, model)
                if log_wandb:
                    wandb.watch(classifier, log="all", log_freq=len(train_loader))
                # TODO: This approach above only works if the checkpoint was saved with full model object, instead, only save the state_dict, like this:
                # classifier = YourModelClass(...)
                # classifier.load_state_dict(checkpoint["classifier_state_dict"])

                # restart_from_checkpoint(
                #     Path(saved_model_path) / "checkpoints" / "model_best.pth",
                #     classifier=classifier,
                #     # run_variables=to_restore,
                #     # optimizer=optimizer,
                #     # loss=criterion,
                # )

        # define metrics
        metric_param = {
            "task": "multiclass",
            "num_classes": classifier.fc.num_labels,
            "average": "macro",
        }
        loss_metric_val = torchmetrics.MeanMetric().to(device)
        f1_score_val = torchmetrics.F1Score(**metric_param).to(device)
        precision_val = torchmetrics.Precision(**metric_param).to(device)
        recall_val = torchmetrics.Recall(**metric_param).to(device)
        auroc_val = torchmetrics.AUROC(
            task=metric_param["task"],
            num_classes=metric_param["num_classes"],
        ).to(device)

        if train is True:
            # start training
            start_epoch, step = 0, 0
            group_dro_loss_computer = None
            if group_dro_train is not None:
                group_dro_loss_computer = GroupDROLossComputer(
                    group_counts=group_dro_train["group_counts"],
                    step_size=group_dro_step_size,
                    adjustment=group_dro_train["adjustment"],
                    normalize_loss=group_dro_normalize_loss,
                    strength=group_dro_strength,
                    device=device,
                )
            eval_scores_dict = {
                "f1": {
                    "metric": f1_score_val,
                    "scores": [],
                },
                "precision": {
                    "metric": precision_val,
                    "scores": [],
                },
                "recall": {
                    "metric": recall_val,
                    "scores": [],
                },
                "auroc": {
                    "metric": auroc_val,
                    "scores": [],
                },
            }
            l_loss_val = []
            selection_score_history = []
            best_val_score = 0.0
            best_model_wts = copy.deepcopy(classifier.state_dict())

            # Resume from training checkpoint if available
            if checkpoint_dir is not None:
                resume_result = cls._maybe_resume_from_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    classifier=classifier,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    early_stopping=early_stopping,
                    eval_scores_dict=eval_scores_dict,
                    group_dro_loss_computer=group_dro_loss_computer,
                    device=device,
                )
                if resume_result is not None:
                    (
                        start_epoch,
                        step,
                        best_val_score,
                        best_model_wts,
                        l_loss_val,
                        selection_score_history,
                    ) = resume_result

            # start with frozen backbone and only let the classifier be trained
            set_requires_grad(classifier, True)
            if hasattr(classifier, "backbone"):
                set_requires_grad(classifier.backbone, False)
            fully_unfreezed = False
            last_epoch = start_epoch - 1

            for epoch in tqdm(
                range(start_epoch, train_epochs),
                total=train_epochs,
                initial=start_epoch,
                desc="Model Training",
            ):
                last_epoch = epoch
                if not fully_unfreezed and epoch >= warmup_epochs:
                    # make sure the classifier and backbone get trained
                    set_requires_grad(classifier, True)
                    fully_unfreezed = True

                # training
                classifier.train()
                for batch in train_loader:
                    batch_group_indices = None
                    if sample_reweighting is not None:
                        img, target, sample_indices = batch
                        batch_sample_weights = cls.get_batch_sample_weights(
                            sample_reweighting=sample_reweighting,
                            sample_indices=sample_indices,
                            device=device,
                        )
                    elif group_dro_train is not None:
                        img, target, batch_group_indices = batch
                        batch_sample_weights = None
                    else:
                        img, target = batch
                        batch_sample_weights = None
                    img = img.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    if batch_group_indices is not None:
                        batch_group_indices = batch_group_indices.to(
                            device, non_blocking=True
                        )

                    optimizer.zero_grad()

                    pred = classifier(img)
                    if batch_sample_weights is not None:
                        loss = cls.compute_training_loss(
                            pred=pred,
                            target=target,
                            class_weights=class_weights,
                            sample_weights=batch_sample_weights,
                        )
                    elif group_dro_loss_computer is not None:
                        per_sample_loss = cls.compute_per_sample_training_loss(
                            pred=pred,
                            target=target,
                            class_weights=class_weights,
                        )
                        loss, _, _, adv_probs = group_dro_loss_computer.compute_robust_loss(
                            per_sample_losses=per_sample_loss,
                            group_idx=batch_group_indices,
                        )
                    else:
                        loss = criterion(pred, target)

                    loss.backward()
                    optimizer.step()

                    # W&B logging if needed
                    if log_wandb:
                        batch_train_f1 = torchmetrics.functional.f1_score(
                            pred,
                            target,
                            task=metric_param["task"],
                            num_classes=metric_param["num_classes"],
                            average=metric_param["average"],
                        )
                        log_dict = {
                            "train_loss": loss.item(),
                            "train_f1": batch_train_f1,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "weight_decay": optimizer.param_groups[0]["weight_decay"],
                            "epoch": epoch,
                            "step": step,
                        }
                        if group_dro_loss_computer is not None:
                            _, _, batch_worst_group_acc = cls.compute_group_dro_accuracy(
                                pred=pred,
                                target=target,
                                group_idx=batch_group_indices,
                                n_groups=group_dro_loss_computer.n_groups,
                            )
                            log_dict["train_worst_group_acc"] = batch_worst_group_acc
                            log_dict["train_adv_prob_max"] = adv_probs.max().item()
                        wandb.log(log_dict)
                    step += 1

                if use_lr_scheduler:
                    scheduler.step()

                # Evaluation
                loss_metric_val.reset()
                for _score_dict in eval_scores_dict.values():
                    _score_dict["metric"].reset()
                eval_group_loss_sum = None
                eval_group_correct_sum = None
                eval_group_count_sum = None
                if group_dro_eval is not None:
                    n_eval_groups = len(group_dro_eval["group_names"])
                    eval_group_loss_sum = torch.zeros(n_eval_groups, device=device)
                    eval_group_correct_sum = torch.zeros(n_eval_groups, device=device)
                    eval_group_count_sum = torch.zeros(n_eval_groups, device=device)
                classifier.eval()
                with torch.no_grad():
                    for img, _, target, index in eval_loader:
                        img = img.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)

                        pred = classifier(img)
                        loss = criterion(pred, target)

                        loss_metric_val.update(loss)
                        for _score_dict in eval_scores_dict.values():
                            _score_dict["metric"].update(pred, target)
                        if group_dro_eval is not None:
                            batch_group_indices = cls.get_batch_group_indices(
                                group_mapping=group_dro_eval["group_index_by_index"],
                                sample_indices=index,
                                device=device,
                            )
                            batch_loss_sum, batch_group_count = (
                                GroupDROLossComputer.compute_group_sum(
                                    cls.compute_per_sample_training_loss(
                                        pred=pred,
                                        target=target,
                                        class_weights=class_weights,
                                    ),
                                    batch_group_indices,
                                    n_eval_groups,
                                )
                            )
                            batch_correct_sum, _ = GroupDROLossComputer.compute_group_sum(
                                (pred.argmax(dim=1) == target).float(),
                                batch_group_indices,
                                n_eval_groups,
                            )
                            eval_group_loss_sum += batch_loss_sum
                            eval_group_correct_sum += batch_correct_sum
                            eval_group_count_sum += batch_group_count

                current_eval_loss = float(loss_metric_val.compute().item())
                current_selection_score = float(
                    eval_scores_dict["f1"]["metric"].compute().item()
                )
                early_stopping_loss = current_eval_loss
                current_worst_group_acc = None
                current_robust_eval_loss = None
                if group_dro_eval is not None:
                    observed_groups = eval_group_count_sum > 0
                    if observed_groups.any():
                        eval_group_loss = eval_group_loss_sum / eval_group_count_sum.clamp_min(
                            1.0
                        )
                        eval_group_acc = eval_group_correct_sum / eval_group_count_sum.clamp_min(
                            1.0
                        )
                        current_robust_eval_loss = float(
                            eval_group_loss[observed_groups].max().item()
                        )
                        current_worst_group_acc = float(
                            eval_group_acc[observed_groups].min().item()
                        )
                        early_stopping_loss = current_robust_eval_loss
                        current_selection_score = current_worst_group_acc

                l_loss_val.append(early_stopping_loss)
                selection_score_history.append(current_selection_score)
                for _score_dict in eval_scores_dict.values():
                    _score_dict["scores"].append(float(_score_dict["metric"].compute().item()))
                # check if we have new best model
                if current_selection_score > best_val_score:
                    best_val_score = current_selection_score
                    best_model_wts = copy.deepcopy(classifier.state_dict())
                # check early stopping
                early_stopping(early_stopping_loss)

                # save training checkpoint for resume capability
                if checkpoint_dir is not None:
                    cls._save_training_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        classifier=classifier,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        early_stopping=early_stopping,
                        epoch=epoch,
                        step=step,
                        best_val_score=best_val_score,
                        best_model_wts=best_model_wts,
                        l_loss_val=l_loss_val,
                        selection_score_history=selection_score_history,
                        eval_scores_dict=eval_scores_dict,
                        group_dro_loss_computer=group_dro_loss_computer,
                        wandb_run_id=wandb_run_id,
                    )

                # W&B logging if needed
                if log_wandb:
                    log_dict = {
                        "eval_loss": l_loss_val[-1],
                        "epoch": epoch,
                        "step": step,
                    }
                    for score_name, _score_dict in eval_scores_dict.items():
                        log_dict[f"eval_{score_name}"] = _score_dict["scores"][-1]
                    if current_worst_group_acc is not None:
                        log_dict["eval_worst_group_acc"] = current_worst_group_acc
                    if current_robust_eval_loss is not None:
                        log_dict["eval_robust_loss"] = current_robust_eval_loss
                    wandb.log(log_dict)

                if early_stopping.early_stop:
                    if debug:
                        print("EarlyStopping, evaluation did not decrease.")
                    break

            # get the best epoch in terms of F1 score
            wandb.unwatch()
            best_epoch = cls.get_best_epoch(
                last_epoch=last_epoch,
                selection_score_history=selection_score_history,
                selection_metric_name=(
                    "worst_group_acc" if group_dro_train is not None else "f1"
                ),
                l_loss_val=l_loss_val,
                eval_scores_dict=eval_scores_dict,
                log_wandb=log_wandb,
                step=step,
            )
            classifier.load_state_dict(best_model_wts)
            if saved_model_path is not None:
                cls.save_model_checkpoint(
                    classifier, criterion, last_epoch, optimizer, saved_model_path
                )

        # create eval predictions for saving
        img_names, targets, prediction_logits, indices = [], [], [], []
        classifier.eval()
        for img, img_name, target, index in eval_loader:
            img = img.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.no_grad():
                pred = classifier(img)
            targets.append(target.cpu())
            prediction_logits.append(pred.cpu())

            img_names.append(img_name)
            indices.append(index)

        img_names = np.hstack(img_names)
        targets = torch.concat(targets).cpu().numpy()
        prediction_scores = torch.softmax(
            torch.concat(prediction_logits), dim=-1
        ).cpu().numpy()
        predictions = prediction_scores.argmax(axis=-1)
        indices = torch.concat(indices).numpy()

        auroc = np.nan
        auroc_values = []
        for class_idx in range(prediction_scores.shape[1]):
            y_true_bin = (targets == class_idx).astype(int)
            if np.unique(y_true_bin).size < 2:
                continue
            try:
                auroc_values.append(
                    roc_auc_score(y_true_bin, prediction_scores[:, class_idx])
                )
            except ValueError:
                continue
        if auroc_values:
            auroc = float(np.mean(auroc_values) * 100)

        macro_f1 = float(sk_f1_score(targets, predictions, average="macro") * 100)
        results = {
            "score": macro_f1,
            "auroc": auroc,
            "filenames": img_names,
            "indices": indices,
            "targets": targets,
            "predictions": predictions,
            "probabilities": prediction_scores.tolist(),
        }
        return results

    @classmethod
    def save_model_checkpoint(
        cls, classifier, criterion, epoch, optimizer, saved_model_path
    ):
        # TODO: consider saving classifier.state_dict() only
        save_dict = {
            "arch": type(classifier).__name__,
            "epoch": epoch,
            "classifier": classifier,
            "optimizer": optimizer.state_dict(),
            "loss": criterion.state_dict(),
        }
        save_checkpoint(
            run_dir=saved_model_path,
            save_dict=save_dict,
            epoch=epoch,
            save_best=True,
        )

    @classmethod
    def _save_training_checkpoint(
        cls,
        checkpoint_dir,
        classifier,
        optimizer,
        scheduler,
        early_stopping,
        epoch,
        step,
        best_val_score,
        best_model_wts,
        l_loss_val,
        selection_score_history,
        eval_scores_dict,
        group_dro_loss_computer=None,
        wandb_run_id=None,
    ):
        """Save a training checkpoint for resume capability."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / "training_checkpoint.pth"

        eval_scores = {}
        for key, val in eval_scores_dict.items():
            eval_scores[key] = [
                s.cpu() if torch.is_tensor(s) else s for s in val["scores"]
            ]

        save_dict = {
            "classifier_state_dict": {
                k: v.cpu() for k, v in classifier.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_val_score": (
                best_val_score.cpu()
                if torch.is_tensor(best_val_score)
                else best_val_score
            ),
            "best_model_wts": {k: v.cpu() for k, v in best_model_wts.items()},
            "l_loss_val": [
                l.cpu() if torch.is_tensor(l) else l for l in l_loss_val
            ],
            "selection_score_history": selection_score_history,
            "eval_scores": eval_scores,
            "early_stopping_state": early_stopping.state_dict(),
        }
        if scheduler is not None:
            save_dict["scheduler_state_dict"] = scheduler.state_dict()
        if group_dro_loss_computer is not None:
            save_dict["group_dro_state"] = group_dro_loss_computer.state_dict()
        if wandb_run_id is not None:
            save_dict["wandb_run_id"] = wandb_run_id

        torch.save(save_dict, ckpt_path)
        logger.debug(f"Saved training checkpoint at epoch {epoch}: {ckpt_path}")

    @classmethod
    def _maybe_resume_from_checkpoint(
        cls,
        checkpoint_dir,
        classifier,
        optimizer,
        scheduler,
        early_stopping,
        eval_scores_dict,
        group_dro_loss_computer,
        device,
    ):
        """Try to resume training from a checkpoint. Returns None if no checkpoint found."""
        checkpoint_dir = Path(checkpoint_dir)
        ckpt_path = checkpoint_dir / "training_checkpoint.pth"

        if not ckpt_path.exists():
            logger.info(f"No training checkpoint found at: {ckpt_path}")
            return None

        logger.info(f"Resuming training from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        classifier.load_state_dict(ckpt["classifier_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        early_stopping.load_state_dict(ckpt["early_stopping_state"])
        if (
            group_dro_loss_computer is not None
            and "group_dro_state" in ckpt
        ):
            group_dro_loss_computer.load_state_dict(ckpt["group_dro_state"])

        # Restore eval scores
        for key in eval_scores_dict:
            if key in ckpt["eval_scores"]:
                eval_scores_dict[key]["scores"] = ckpt["eval_scores"][key]

        start_epoch = ckpt["epoch"] + 1  # Resume from the next epoch
        step = ckpt["step"]
        best_val_score = ckpt["best_val_score"]
        best_model_wts = ckpt["best_model_wts"]
        l_loss_val = ckpt["l_loss_val"]
        selection_score_history = ckpt.get("selection_score_history", [])

        logger.info(
            f"Resumed from epoch {ckpt['epoch']}, continuing at epoch {start_epoch}"
        )
        return (
            start_epoch,
            step,
            best_val_score,
            best_model_wts,
            l_loss_val,
            selection_score_history,
        )

    @classmethod
    def _cleanup_training_checkpoint(cls, checkpoint_dir):
        """Remove training checkpoint after successful completion."""
        checkpoint_dir = Path(checkpoint_dir)
        ckpt_path = checkpoint_dir / "training_checkpoint.pth"
        if ckpt_path.exists():
            ckpt_path.unlink()
            logger.info(
                f"Removed training checkpoint after successful completion: {ckpt_path}"
            )

    @classmethod
    def get_wandb_run_id_from_checkpoint(cls, checkpoint_dir):
        """Read the wandb run ID from a training checkpoint for resume."""
        if checkpoint_dir is None:
            return None
        checkpoint_dir = Path(checkpoint_dir)
        ckpt_path = checkpoint_dir / "training_checkpoint.pth"
        if not ckpt_path.exists():
            return None
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            return ckpt.get("wandb_run_id")
        except Exception:
            return None

    @classmethod
    def get_best_epoch(
        cls,
        last_epoch,
        selection_score_history,
        selection_metric_name,
        l_loss_val,
        eval_scores_dict,
        log_wandb,
        step,
    ):
        if not selection_score_history:
            return 0

        best_epoch = int(np.argmax(np.asarray(selection_score_history, dtype=float)))
        if log_wandb:
            log_dict = {
                "best_eval_epoch": best_epoch,
                "best_eval_loss": l_loss_val[best_epoch],
                "best_eval_selection_metric": selection_score_history[best_epoch],
                "epoch": last_epoch,
                "step": step,
            }
            for score_name, _score_dict in eval_scores_dict.items():
                log_dict[f"best_eval_{score_name}"] = _score_dict["scores"][best_epoch]
            log_dict[f"best_eval_{selection_metric_name}"] = selection_score_history[
                best_epoch
            ]
            wandb.log(log_dict)
        return best_epoch

    @classmethod
    def configure_optimizer(
        cls,
        classifier,
        criterion,
        device,
        find_optimal_lr,
        learning_rate,
        weight_decay,
        log_wandb,
        train_loader,
        seed,
        sample_reweighting: Optional[dict] = None,
        group_dro: Optional[dict] = None,
        group_dro_step_size: float = 0.01,
        group_dro_normalize_loss: bool = False,
        group_dro_strength: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        optimizer_cls = get_optimizer_type(optimizer_name="adam")
        optimizer = optimizer_cls(
            params=classifier.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        if find_optimal_lr:
            fix_random_seeds(seed)
            # automatic learning rate finder
            lr_train_loader = train_loader
            lr_criterion = criterion
            if sample_reweighting is not None:
                lr_train_loader = WeightedTrainDataLoaderIter(train_loader)
                lr_criterion = WeightedLRCriterion(
                    sample_reweighting=sample_reweighting,
                    class_weights=class_weights,
                )
            elif group_dro is not None:
                lr_train_loader = GroupDROTrainDataLoaderIter(train_loader)
                lr_criterion = GroupDROLRCriterion(
                    group_dro=group_dro,
                    class_weights=class_weights,
                    step_size=group_dro_step_size,
                    normalize_loss=group_dro_normalize_loss,
                    strength=group_dro_strength,
                )

            lr_finder = LRFinder(classifier, optimizer, lr_criterion, device=device)
            lr_finder.range_test(lr_train_loader, end_lr=100, num_iter=100)
            lrs = lr_finder.history["lr"]
            losses = lr_finder.history["loss"]
            # log the LRFinder plot
            fig, ax = plt.subplots()
            lr_finder.plot(ax=ax)
            if log_wandb:
                wandb.log({"LRFinder_Plot": wandb.Image(fig)})
            # to reset the model and optimizer to their initial state
            lr_finder.reset()
            try:
                losses_np = np.array(losses, dtype=float)
                if sample_reweighting is not None or group_dro is not None:
                    smoothing_window = 5
                    if len(losses_np) >= smoothing_window:
                        smoothing_kernel = np.ones(smoothing_window) / smoothing_window
                        losses_for_selection = np.convolve(
                            losses_np,
                            smoothing_kernel,
                            mode="same",
                        )
                    else:
                        losses_for_selection = losses_np
                else:
                    losses_for_selection = losses_np
                min_grad_idx = np.gradient(losses_for_selection).argmin()
                best_lr = lrs[min_grad_idx]
                if sample_reweighting is not None or group_dro is not None:
                    best_lr = min(best_lr, 1.0e-3)
                optimizer = optimizer_cls(
                    params=classifier.parameters(),
                    lr=best_lr,
                    weight_decay=weight_decay,
                )
            except ValueError:
                print("Failed to compute the gradients. Relying on default lr.")
        return optimizer

    @classmethod
    def configure_classifier_base(
        cls, classifier, debug, log_wandb, model, train_loader
    ):
        # make sure the classifier can get trained
        set_requires_grad(classifier, True)
        cls.print_model(classifier, debug, model)
        if log_wandb:
            wandb.watch(classifier, log="all", log_freq=len(train_loader))

    @classmethod
    def print_model(cls, classifier, debug, model):
        if debug and model is not None:
            try:
                summary(classifier, input_size=(1, 3, cls.input_size, cls.input_size))
            except RuntimeError:
                print("Summary can not be displayed for a Huggingface model.")
                print(
                    f"Number of parameters backbone: {classifier.backbone.model.num_parameters():,}"
                )

    @classmethod
    def get_device(cls, model):
        if model is not None:
            device = model.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"device: {device}")
        return device

    @classmethod
    def create_classifier(
        cls, dataset, dropout_in_head, model, model_out_dim, use_bn_in_head
    ):
        # create the classifier
        classifier_list = []
        if model is not None:
            model = copy.deepcopy(model)
            classifier_list = [
                ("backbone", model),
                ("flatten", torch.nn.Flatten()),
            ]
        classifier_list.append(
            (
                "fc",
                LinearClassifier(
                    model_out_dim,
                    dataset.n_classes,
                    large_head=False,
                    use_bn=use_bn_in_head,
                    dropout_rate=dropout_in_head,
                ),
            ),
        )
        classifier = torch.nn.Sequential(OrderedDict(classifier_list))
        return classifier, model

    @classmethod
    def normalize_group_columns(cls, group_columns: Optional[Union[str, list]]) -> list:
        if group_columns is None:
            raise ValueError("At least one group column must be provided.")
        if isinstance(group_columns, str):
            return [group_columns]
        if len(group_columns) == 0:
            raise ValueError("At least one group column must be provided.")
        return list(group_columns)

    @classmethod
    def build_group_name_series(
        cls,
        meta_data: pd.DataFrame,
        group_columns: list[str],
        label_col: Optional[str] = None,
    ) -> pd.Series:
        if len(group_columns) == 1:
            group_name = meta_data[group_columns[0]].astype(str).str.strip()
        else:
            group_name = meta_data[group_columns].astype(str).agg("__".join, axis=1)

        if label_col is not None:
            label_name = meta_data[label_col].astype(str).str.strip()
            group_name = label_name + "__" + group_name
        return group_name

    @classmethod
    def compute_group_dro_metadata(
        cls,
        dataset: torch.utils.data.Dataset,
        selected_range: np.ndarray,
        group_columns: Optional[Union[str, list]],
        include_label: bool = False,
        generalization_adjustment: float = 0.0,
    ) -> dict:
        group_columns = cls.normalize_group_columns(group_columns)
        missing_columns = [
            column for column in group_columns if column not in dataset.meta_data.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Unable to compute Group DRO groups. Missing columns: {missing_columns}"
            )

        if generalization_adjustment < 0:
            raise ValueError(
                "group_dro_adjustment must be non-negative."
            )

        selected_columns = list(group_columns)
        if include_label:
            selected_columns.append(dataset.LBL_COL)

        selected_meta = dataset.meta_data.loc[selected_range, selected_columns].copy()
        if selected_meta.empty:
            raise ValueError(
                "Unable to compute Group DRO groups because the selected split is empty."
            )

        group_name = cls.build_group_name_series(
            meta_data=selected_meta,
            group_columns=group_columns,
            label_col=dataset.LBL_COL if include_label else None,
        )
        group_codes, unique_group_names = pd.factorize(group_name, sort=True)
        group_counts = np.bincount(group_codes, minlength=len(unique_group_names))
        group_counts_tensor = torch.as_tensor(group_counts, dtype=torch.float32)
        adjustment = torch.zeros_like(group_counts_tensor)
        if generalization_adjustment > 0:
            adjustment = float(generalization_adjustment) / torch.sqrt(
                group_counts_tensor
            )
        return {
            "group_columns": group_columns,
            "include_label": include_label,
            "group_names": [str(name) for name in unique_group_names],
            "group_index_by_index": dict(
                zip(selected_meta.index.astype(int), group_codes.astype(int))
            ),
            "group_counts": group_counts_tensor,
            "adjustment": adjustment,
        }

    @classmethod
    def compute_sample_reweighting(
        cls,
        dataset: torch.utils.data.Dataset,
        train_range: np.ndarray,
        sensitive_columns: Optional[Union[str, list]],
        strength: float = 1.0,
    ) -> dict:
        """Compute Kamiran-Calders sample weights on the training split."""
        strength = cls.normalize_mitigation_strength(strength)
        sensitive_columns = cls.normalize_group_columns(sensitive_columns)
        missing_columns = [
            column
            for column in sensitive_columns
            if column not in dataset.meta_data.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Unable to compute instance reweighting. Missing columns: {missing_columns}"
            )

        label_col = dataset.LBL_COL
        train_meta = dataset.meta_data.loc[
            train_range, [label_col, *sensitive_columns]
        ].copy()
        if train_meta.empty:
            raise ValueError(
                "Unable to compute instance reweighting because the training split is empty."
            )
        train_meta["sensitive_group"] = train_meta[sensitive_columns].astype(str).agg(
            "__".join, axis=1
        )

        n_samples = len(train_meta)
        label_counts = train_meta[label_col].value_counts()
        sensitive_counts = train_meta["sensitive_group"].value_counts()
        joint_counts = train_meta.groupby([label_col, "sensitive_group"]).size()

        joint_weights = {}
        for (label_value, sensitive_group), joint_count in joint_counts.items():
            full_weight = (
                label_counts[label_value] * sensitive_counts[sensitive_group]
            ) / (n_samples * joint_count)
            joint_weights[(label_value, sensitive_group)] = (
                1.0 + strength * (full_weight - 1.0)
            )

        train_meta[cls.SAMPLE_WEIGHT_COL] = train_meta.apply(
            lambda row: joint_weights[(row[label_col], row["sensitive_group"])], axis=1
        )

        logger.info(
            "Kamiran-Calders instance reweighting enabled for columns {} with {} joint groups at strength {}.",
            sensitive_columns,
            len(joint_weights),
            strength,
        )
        return {
            "label_col": label_col,
            "sensitive_columns": sensitive_columns,
            "weights_by_index": train_meta[cls.SAMPLE_WEIGHT_COL].to_dict(),
            "weights_by_joint_group": joint_weights,
            "strength": strength,
        }

    @staticmethod
    def normalize_mitigation_strength(strength: Optional[float]) -> float:
        if strength is None:
            return 1.0

        normalized = float(strength)
        if normalized < 0.0 or normalized > 1.0:
            raise ValueError(
                f"Mitigation strength must be in [0, 1], got {strength}."
            )
        return normalized

    @classmethod
    def get_batch_sample_weights(
        cls,
        sample_reweighting: dict,
        sample_indices: torch.Tensor,
        device: Union[str, torch.device],
    ) -> torch.Tensor:
        weights_by_index = sample_reweighting["weights_by_index"]
        return torch.as_tensor(
            [float(weights_by_index[int(idx)]) for idx in sample_indices],
            dtype=torch.float32,
            device=device,
        )

    @classmethod
    def get_batch_group_indices(
        cls,
        group_mapping: dict,
        sample_indices: torch.Tensor,
        device: Union[str, torch.device],
    ) -> torch.Tensor:
        return torch.as_tensor(
            [int(group_mapping[int(idx)]) for idx in sample_indices],
            dtype=torch.long,
            device=device,
        )

    @classmethod
    def compute_per_sample_training_loss(
        cls,
        pred: torch.Tensor,
        target: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.cross_entropy(
            pred,
            target,
            weight=class_weights,
            reduction="none",
        )

    @classmethod
    def compute_training_loss(
        cls,
        pred: torch.Tensor,
        target: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        per_sample_loss = cls.compute_per_sample_training_loss(
            pred=pred,
            target=target,
            class_weights=class_weights,
        )
        if sample_weights is None:
            return per_sample_loss.mean()

        weighted_loss = per_sample_loss * sample_weights
        return weighted_loss.mean()

    @classmethod
    def compute_group_dro_accuracy(
        cls,
        pred: torch.Tensor,
        target: torch.Tensor,
        group_idx: torch.Tensor,
        n_groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        group_correct_sum, group_count = GroupDROLossComputer.compute_group_sum(
            (pred.argmax(dim=1) == target).float(),
            group_idx,
            n_groups,
        )
        group_acc = group_correct_sum / group_count.clamp_min(1.0)
        observed_groups = group_count > 0
        worst_group_acc = float(group_acc[observed_groups].min().item())
        return group_acc, group_count, worst_group_acc

    @classmethod
    def get_train_eval_loaders(
        cls,
        dataset: torch.utils.data.Dataset,
        train_range: np.ndarray,
        evaluation_range: np.ndarray,
        balanced_train_indices: np.ndarray,
        offline_augmented_rows,
        batch_size: int,
        num_workers: int,
        seed: int,
        sample_reweighting: Optional[dict] = None,
        group_dro: Optional[dict] = None,
    ):
        g = torch.Generator()
        g.manual_seed(seed)

        train_dataset = copy.deepcopy(dataset)
        train_sampler_indices = np.array(balanced_train_indices, copy=True)
        if (
            sample_reweighting is None
            and offline_augmented_rows is not None
            and len(offline_augmented_rows) > 0
        ):
            original_train_len = len(train_dataset.meta_data)
            train_dataset.meta_data = (
                pd.concat(
                    [train_dataset.meta_data, offline_augmented_rows.copy()],
                    ignore_index=True,
                ).reset_index(drop=True)
            )
            augmented_indices = np.arange(
                original_train_len,
                original_train_len + len(offline_augmented_rows),
            )
            train_sampler_indices = np.concatenate([train_sampler_indices, augmented_indices])
        elif sample_reweighting is not None and len(offline_augmented_rows) > 0:
            logger.info(
                "Skipping offline augmented rows because instance reweighting is enabled."
            )
        train_dataset.transform = cls.train_transform()
        train_dataset.val_transform = None
        train_dataset.training = True
        train_dataset.return_index_in_training = sample_reweighting is not None
        train_dataset.return_group_in_training = group_dro is not None
        if group_dro is not None:
            train_dataset.group_index_col = cls.GROUP_DRO_INDEX_COL
            train_dataset.meta_data[cls.GROUP_DRO_INDEX_COL] = -1
            for sample_index, group_index in group_dro["group_index_by_index"].items():
                train_dataset.meta_data.loc[sample_index, cls.GROUP_DRO_INDEX_COL] = (
                    int(group_index)
                )
        train_sampler = SubsetRandomSampler(train_sampler_indices, generator=g)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            drop_last=True,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=functools.partial(cls._seed_worker, seed=seed),
            generator=g,
        )
        del train_dataset

        eval_dataset = copy.deepcopy(dataset)
        eval_dataset.training = False
        eval_dataset.transform = None
        eval_dataset.val_transform = cls.val_transform()
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(evaluation_range, generator=g),
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=functools.partial(cls._seed_worker, seed=seed),
            generator=g,
        )
        del eval_dataset
        return train_loader, eval_loader


class WeightedTrainDataLoaderIter(TrainDataLoaderIter):
    """Adapter so torch-lr-finder can consume weighted training batches."""

    def inputs_labels_from_batch(self, batch_data):
        img, target, sample_indices = batch_data
        return img, (target, sample_indices)


class WeightedLRCriterion(torch.nn.Module):
    """Criterion wrapper for LR finding with per-sample reweighting."""

    def __init__(
        self,
        sample_reweighting: dict,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.sample_reweighting = sample_reweighting
        self.class_weights = class_weights.detach().clone() if class_weights is not None else None

    def forward(self, pred, target_with_indices):
        target, sample_indices = target_with_indices
        class_weights = self.class_weights
        if class_weights is not None:
            class_weights = class_weights.to(pred.device)
        batch_sample_weights = EvalFineTuning.get_batch_sample_weights(
            sample_reweighting=self.sample_reweighting,
            sample_indices=sample_indices,
            device=pred.device,
        )
        return EvalFineTuning.compute_training_loss(
            pred=pred,
            target=target,
            class_weights=class_weights,
            sample_weights=batch_sample_weights,
        )


class GroupDROTrainDataLoaderIter(TrainDataLoaderIter):
    """Adapter so torch-lr-finder can consume Group DRO training batches."""

    def inputs_labels_from_batch(self, batch_data):
        img, target, group_indices = batch_data
        return img, (target, group_indices)


class GroupDROLRCriterion(torch.nn.Module):
    """Criterion wrapper for LR finding with Group DRO batches."""

    def __init__(
        self,
        group_dro: dict,
        class_weights: Optional[torch.Tensor] = None,
        step_size: float = 0.01,
        normalize_loss: bool = False,
        strength: float = 1.0,
    ):
        super().__init__()
        self.class_weights = (
            class_weights.detach().clone() if class_weights is not None else None
        )
        self.loss_computer = GroupDROLossComputer(
            group_counts=group_dro["group_counts"],
            step_size=step_size,
            adjustment=group_dro["adjustment"],
            normalize_loss=normalize_loss,
            strength=strength,
        )

    def forward(self, pred, target_with_group_indices):
        target, group_indices = target_with_group_indices
        class_weights = self.class_weights
        if class_weights is not None:
            class_weights = class_weights.to(pred.device)
        self.loss_computer.to(pred.device)
        per_sample_loss = EvalFineTuning.compute_per_sample_training_loss(
            pred=pred,
            target=target,
            class_weights=class_weights,
        )
        robust_loss, _, _, _ = self.loss_computer.compute_robust_loss(
            per_sample_losses=per_sample_loss,
            group_idx=group_indices.to(pred.device),
        )
        return robust_loss
