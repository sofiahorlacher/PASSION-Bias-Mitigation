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
        enable_instance_reweighting: bool = False,
        instance_reweighting_columns: Optional[Union[str, list]] = None,
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
                log_wandb,
                train_loader,
                seed,
                sample_reweighting=sample_reweighting,
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
            best_val_score = 0
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
                    device=device,
                )
                if resume_result is not None:
                    start_epoch, step, best_val_score, best_model_wts, l_loss_val = resume_result

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
                    if sample_reweighting is not None:
                        img, target, sample_indices = batch
                        batch_sample_weights = cls.get_batch_sample_weights(
                            sample_reweighting=sample_reweighting,
                            sample_indices=sample_indices,
                            device=device,
                        )
                    else:
                        img, target = batch
                        batch_sample_weights = None
                    img = img.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                    optimizer.zero_grad()

                    pred = classifier(img)
                    if batch_sample_weights is None:
                        loss = criterion(pred, target)
                    else:
                        loss = cls.compute_training_loss(
                            pred=pred,
                            target=target,
                            class_weights=class_weights,
                            sample_weights=batch_sample_weights,
                        )

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
                        wandb.log(log_dict)
                    step += 1

                if use_lr_scheduler:
                    scheduler.step()

                # Evaluation
                loss_metric_val.reset()
                for _score_dict in eval_scores_dict.values():
                    _score_dict["metric"].reset()
                classifier.eval()
                with torch.no_grad():
                    for img, _, target, _ in eval_loader:
                        img = img.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)

                        pred = classifier(img)
                        loss = criterion(pred, target)

                        loss_metric_val.update(loss)
                        for _score_dict in eval_scores_dict.values():
                            _score_dict["metric"].update(pred, target)
                l_loss_val.append(loss_metric_val.compute())
                for _score_dict in eval_scores_dict.values():
                    _score_dict["scores"].append(_score_dict["metric"].compute())
                # check if we have new best model
                if eval_scores_dict["f1"]["scores"][-1] > best_val_score:
                    best_val_score = eval_scores_dict["f1"]["scores"][-1]
                    best_model_wts = copy.deepcopy(classifier.state_dict())
                # check early stopping
                early_stopping(l_loss_val[-1])

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
                        eval_scores_dict=eval_scores_dict,
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
                    wandb.log(log_dict)

                if early_stopping.early_stop:
                    if debug:
                        print("EarlyStopping, evaluation did not decrease.")
                    break

            # get the best epoch in terms of F1 score
            wandb.unwatch()
            best_epoch = cls.get_best_epoch(
                last_epoch, eval_scores_dict, l_loss_val, log_wandb, step
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
        eval_scores_dict,
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
            "eval_scores": eval_scores,
            "early_stopping_state": early_stopping.state_dict(),
        }
        if scheduler is not None:
            save_dict["scheduler_state_dict"] = scheduler.state_dict()
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

        # Restore eval scores
        for key in eval_scores_dict:
            if key in ckpt["eval_scores"]:
                eval_scores_dict[key]["scores"] = ckpt["eval_scores"][key]

        start_epoch = ckpt["epoch"] + 1  # Resume from the next epoch
        step = ckpt["step"]
        best_val_score = ckpt["best_val_score"]
        best_model_wts = ckpt["best_model_wts"]
        l_loss_val = ckpt["l_loss_val"]

        logger.info(
            f"Resumed from epoch {ckpt['epoch']}, continuing at epoch {start_epoch}"
        )
        return start_epoch, step, best_val_score, best_model_wts, l_loss_val

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
    def get_best_epoch(cls, epoch, eval_scores_dict, l_loss_val, log_wandb, step):
        best_epoch = torch.Tensor(eval_scores_dict["f1"]["scores"]).argmax()
        if log_wandb:
            log_dict = {
                "best_eval_epoch": best_epoch,
                "best_eval_loss": l_loss_val[best_epoch],
                "epoch": epoch,
                "step": step,
            }
            for score_name, _score_dict in eval_scores_dict.items():
                log_dict[f"best_eval_{score_name}"] = _score_dict["scores"][best_epoch]
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
        log_wandb,
        train_loader,
        seed,
        sample_reweighting: Optional[dict] = None,
        class_weights: Optional[torch.Tensor] = None,
    ):
        optimizer_cls = get_optimizer_type(optimizer_name="adam")
        optimizer = optimizer_cls(
            params=classifier.parameters(),
            lr=learning_rate,
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
                if sample_reweighting is not None:
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
                if sample_reweighting is not None:
                    best_lr = min(best_lr, 1.0e-3)
                optimizer = optimizer_cls(
                    params=classifier.parameters(),
                    lr=best_lr,
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
            raise ValueError(
                "Instance reweighting requires `instance_reweighting_columns`."
            )
        if isinstance(group_columns, str):
            return [group_columns]
        if len(group_columns) == 0:
            raise ValueError(
                "Instance reweighting requires at least one sensitive group column."
            )
        return list(group_columns)

    @classmethod
    def compute_sample_reweighting(
        cls,
        dataset: torch.utils.data.Dataset,
        train_range: np.ndarray,
        sensitive_columns: Optional[Union[str, list]],
    ) -> dict:
        """Compute Kamiran-Calders sample weights on the training split."""
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
            joint_weights[(label_value, sensitive_group)] = (
                label_counts[label_value] * sensitive_counts[sensitive_group]
            ) / (n_samples * joint_count)

        train_meta[cls.SAMPLE_WEIGHT_COL] = train_meta.apply(
            lambda row: joint_weights[(row[label_col], row["sensitive_group"])], axis=1
        )

        logger.info(
            "Kamiran-Calders instance reweighting enabled for columns {} with {} joint groups.",
            sensitive_columns,
            len(joint_weights),
        )
        return {
            "label_col": label_col,
            "sensitive_columns": sensitive_columns,
            "weights_by_index": train_meta[cls.SAMPLE_WEIGHT_COL].to_dict(),
            "weights_by_joint_group": joint_weights,
        }

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
    def compute_training_loss(
        cls,
        pred: torch.Tensor,
        target: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        per_sample_loss = F.cross_entropy(
            pred,
            target,
            weight=class_weights,
            reduction="none",
        )
        if sample_weights is None:
            return per_sample_loss.mean()

        weighted_loss = per_sample_loss * sample_weights
        return weighted_loss.mean()

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
