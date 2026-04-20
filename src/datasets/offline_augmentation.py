import random
from pathlib import Path
import re
from typing import Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from torchvision import transforms


def _group_series(meta_data: pd.DataFrame, group_columns: list[str]) -> pd.Series:
    if len(group_columns) == 1:
        return meta_data[group_columns[0]]
    return meta_data[group_columns].astype(str).agg("_".join, axis=1)


def _sanitize_for_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _apply_random_color_jitter(
    image: Image.Image,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
) -> Image.Image:
    out = image.copy().convert("RGB")
    jitter = transforms.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )
    return jitter(out)


def _apply_paper_plain_color_jitter(
    image: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    """ Taken from paper implementation https://github.com/elieppr/SkinClassificationBias/blob/main/utils.py"""
    jittered = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    jittered = np.array(jittered, dtype=np.float64)

    random_brightness_coefficient = rng.uniform(0.5, 1.5)
    random_saturation_coefficient = rng.uniform(0.5, 1.5)
    random_hue_coefficient = rng.uniform(0.5, 1.5)

    jittered[:, :, 0] = jittered[:, :, 0] * random_hue_coefficient
    jittered[:, :, 1] = jittered[:, :, 1] * random_saturation_coefficient
    jittered[:, :, 2] = jittered[:, :, 2] * random_brightness_coefficient

    jittered[:, :, 0][jittered[:, :, 0] > 255] = 255
    jittered[:, :, 1][jittered[:, :, 1] > 255] = 255
    jittered[:, :, 2][jittered[:, :, 2] > 255] = 255

    jittered = np.array(jittered, dtype=np.uint8)
    return cv2.cvtColor(jittered, cv2.COLOR_HSV2BGR)


def _save_augmented_image(image: Union[Image.Image, np.ndarray], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, np.ndarray):
        cv2.imwrite(str(output_path), image)
        return

    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(output_path, quality=95)
        return
    image.save(output_path)


def generate_offline_augmented_rows(
    dataset: torch.utils.data.Dataset,
    train_range: np.ndarray,
    underrepresented_group_columns: Optional[Union[str, list]],
    output_dir: Path,
    balance_target: str = "median",
    balance_target_group_value: Optional[Union[str, int, float]] = None,
    balance_reference_range: Optional[np.ndarray] = None,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
    jitter_implementation: str = "torchvision",
    seed: int = 42,
    debug: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    if underrepresented_group_columns is None:
        return np.array(train_range, copy=True), dataset.meta_data.iloc[0:0].copy()

    if not hasattr(dataset, "meta_data"):
        logger.warning("Dataset has no metadata. Skipping offline augmentation.")
        return np.array(train_range, copy=True), pd.DataFrame()

    if isinstance(underrepresented_group_columns, str):
        group_columns = [underrepresented_group_columns]
    elif isinstance(underrepresented_group_columns, list):
        group_columns = underrepresented_group_columns
    else:
        logger.warning(
            "underrepresented_group_columns must be str or list. Skipping offline augmentation."
        )
        return np.array(train_range, copy=True), dataset.meta_data.iloc[0:0].copy()

    missing_cols = [col for col in group_columns if col not in dataset.meta_data.columns]
    if missing_cols:
        logger.warning(
            f"Underrepresented group columns {missing_cols} not found in metadata. "
            f"Available columns: {dataset.meta_data.columns.tolist()}"
        )
        return np.array(train_range, copy=True), dataset.meta_data.iloc[0:0].copy()

    if balance_reference_range is None:
        balance_reference_range = train_range

    reference_meta = dataset.meta_data.iloc[balance_reference_range].copy()
    reference_group_counts = _group_series(reference_meta, group_columns).value_counts()

    if balance_target == "reference_group":
        if len(group_columns) != 1 or balance_target_group_value is None:
            logger.warning(
                "balance_target='reference_group' requires a single group column and "
                "`balance_target_group_value`. Falling back to median."
            )
            target_size = int(np.median(reference_group_counts.values))
        else:
            target_size = None
            for group_value, count in reference_group_counts.items():
                if str(group_value) == str(balance_target_group_value):
                    target_size = int(count)
                    break

            if target_size is None:
                logger.warning(
                    f"Reference group value '{balance_target_group_value}' not found in "
                    f"group counts {reference_group_counts.to_dict()}. Falling back to median."
                )
                target_size = int(np.median(reference_group_counts.values))
    elif balance_target == "median":
        target_size = int(np.median(reference_group_counts.values))
    elif balance_target == "max":
        target_size = int(reference_group_counts.max())
    elif isinstance(balance_target, (int, float)):
        target_size = int(balance_target)
    else:
        logger.warning(f"Invalid balance_target '{balance_target}'. Using median.")
        target_size = int(np.median(reference_group_counts.values))

    if debug:
        logger.info(
            f"Generating offline color jitter copies for {group_columns} to target size {target_size}. "
            f"Reference group distribution: {reference_group_counts.to_dict()}"
        )

    train_meta = dataset.meta_data.iloc[train_range].copy()
    train_group_series = _group_series(train_meta, group_columns)
    train_group_counts = train_group_series.value_counts()

    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    torch.manual_seed(seed)
    img_col = dataset.IMG_COL
    augmented_rows = []
    balanced_train_indices = []

    for group, current_size in train_group_counts.items():
        group_mask = train_group_series == group
        group_indices = train_range[group_mask.values]

        if current_size > target_size:
            kept_indices = rng.choice(group_indices, size=target_size, replace=False)
        else:
            kept_indices = np.array(group_indices, copy=True)
        balanced_train_indices.extend(int(idx) for idx in kept_indices)

        if current_size >= target_size:
            continue

        n_samples_needed = target_size - current_size
        sampled_indices = rng.choice(group_indices, size=n_samples_needed, replace=True)

        for dup_id, source_index in enumerate(sampled_indices):
            row = dataset.meta_data.iloc[int(source_index)].copy()
            source_path = Path(row[img_col])
            if jitter_implementation == "paper_plain":
                source_image = cv2.imread(str(source_path))
                if source_image is None:
                    logger.warning(f"Unable to load image for augmentation: {source_path}")
                    continue
                augmented_image = _apply_paper_plain_color_jitter(
                    image=source_image,
                    rng=py_rng,
                )
            else:
                with Image.open(source_path) as img:
                    source_image = img.convert("RGB")
                augmented_image = _apply_random_color_jitter(
                    image=source_image,
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                )

            output_suffix = source_path.suffix if source_path.suffix else ".jpg"
            safe_group = _sanitize_for_filename(str(group))
            output_name = (
                f"{source_path.stem}__offline_jitter__{safe_group}__src{int(source_index)}__dup{dup_id}"
                f"{output_suffix}"
            )
            output_path = output_dir / output_name
            _save_augmented_image(augmented_image, output_path)

            row[img_col] = str(output_path)
            if "img_name" in row.index:
                row["img_name"] = output_path.stem
            augmented_rows.append(row)

    if not augmented_rows:
        return np.asarray(balanced_train_indices, dtype=int), dataset.meta_data.iloc[0:0].copy()

    augmented_meta = pd.DataFrame(augmented_rows)
    augmented_meta = augmented_meta.reindex(columns=dataset.meta_data.columns)

    if debug:
        logger.info(
            f"Generated {len(augmented_meta)} offline augmented training rows in {output_dir}."
        )

    return np.asarray(balanced_train_indices, dtype=int), augmented_meta
