from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import fairlearn
from fairlearn.postprocessing import ThresholdOptimizer


DEFAULT_OBJECTIVE = "balanced_accuracy_score"
DEFAULT_GRID_SIZE = 1000
DEFAULT_FLIP = False


def _normalize_strength(strength: Optional[float]) -> float:
    if strength is None:
        return 1.0

    normalized = float(strength)
    if normalized < 0.0 or normalized > 1.0:
        raise ValueError(f"Post-processing strength must be in [0, 1], got {strength}.")
    return normalized


def _normalize_group_names(sensitive_features: np.ndarray) -> np.ndarray:
    return np.asarray([str(value).strip() for value in sensitive_features])


def _serialize_scalar(value):
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return float(value)


def _serialize_threshold_operation(operation) -> dict:
    return {
        "operator": str(operation.operator),
        "threshold": _serialize_scalar(operation.threshold),
    }


def _serialize_interpolation(interpolation) -> dict:
    serialized = {
        "p0": float(interpolation["p0"]),
        "operation0": _serialize_threshold_operation(interpolation["operation0"]),
        "p1": float(interpolation["p1"]),
        "operation1": _serialize_threshold_operation(interpolation["operation1"]),
    }
    if "p_ignore" in interpolation:
        serialized["p_ignore"] = float(interpolation["p_ignore"])
    if "prediction_constant" in interpolation:
        serialized["prediction_constant"] = float(
            interpolation["prediction_constant"]
        )
    return serialized


def _score_vector_to_probability_matrix(scores: np.ndarray) -> np.ndarray:
    positive_scores = np.asarray(scores, dtype=float).reshape(-1)
    positive_scores = np.clip(positive_scores, 0.0, 1.0)
    return np.column_stack([1.0 - positive_scores, positive_scores])


def _mix_predictions_with_strength(
    original_predictions: np.ndarray,
    adjusted_predictions: np.ndarray,
    strength: float,
    seed: Optional[int],
) -> tuple[np.ndarray, dict]:
    original_predictions = np.asarray(original_predictions)
    adjusted_predictions = np.asarray(adjusted_predictions)
    changed_mask = adjusted_predictions != original_predictions
    n_changed_full = int(changed_mask.sum())

    if strength <= 0.0 or n_changed_full == 0:
        return np.array(original_predictions, copy=True), {
            "strength": strength,
            "application": "disabled" if strength <= 0.0 else "no_prediction_changes",
            "n_changed_at_full_strength": n_changed_full,
            "n_changed_after_strength": 0,
        }

    if strength >= 1.0:
        return np.array(adjusted_predictions, copy=True), {
            "strength": strength,
            "application": "full_adjustment",
            "n_changed_at_full_strength": n_changed_full,
            "n_changed_after_strength": n_changed_full,
        }

    rng = np.random.default_rng(seed)
    apply_adjustment_mask = changed_mask & (rng.random(changed_mask.shape[0]) < strength)
    blended_predictions = np.array(original_predictions, copy=True)
    blended_predictions[apply_adjustment_mask] = adjusted_predictions[
        apply_adjustment_mask
    ]
    return blended_predictions, {
        "strength": strength,
        "application": "sample_level_randomized_mix",
        "n_changed_at_full_strength": n_changed_full,
        "n_changed_after_strength": int(apply_adjustment_mask.sum()),
    }


class _ProbabilityPassthroughEstimator(BaseEstimator, ClassifierMixin):
    """Minimal estimator wrapper so Fairlearn can reuse existing scores."""

    def __init__(self):
        self.is_fitted_ = True
        self.classes_ = np.asarray([0, 1], dtype=int)

    def fit(self, X, y=None):
        self.is_fitted_ = True
        self.classes_ = np.asarray([0, 1], dtype=int)
        return self

    def predict_proba(self, X):
        probability_matrix = np.asarray(X, dtype=float)
        if probability_matrix.ndim != 2 or probability_matrix.shape[1] != 2:
            raise ValueError(
                "Expected a binary probability matrix with shape (n_samples, 2), "
                f"got {probability_matrix.shape}."
            )
        return np.clip(probability_matrix, 0.0, 1.0)


def _validate_binary_calibration_data(
    scores: np.ndarray,
    targets: np.ndarray,
    sensitive_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, int]]]:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    sensitive_features = np.asarray(sensitive_features).reshape(-1)

    if len(scores) != len(targets) or len(scores) != len(sensitive_features):
        raise ValueError(
            "scores, targets, and sensitive_features must have the same length."
        )
    if len(scores) == 0:
        raise ValueError(
            "Cannot fit equalized-odds post-processing on an empty split."
        )

    unique_labels = sorted(int(label) for label in np.unique(targets))
    if unique_labels != [0, 1]:
        raise ValueError(
            "Equalized-odds post-processing requires binary labels encoded as "
            f"{{0, 1}}. Got {unique_labels}."
        )

    group_names = _normalize_group_names(sensitive_features)
    unique_groups = sorted({group for group in group_names if group != ""})
    if len(unique_groups) < 2:
        raise ValueError(
            "Equalized-odds post-processing requires at least two observed "
            "sensitive groups."
        )

    group_counts = {}
    for group_name in unique_groups:
        group_mask = group_names == group_name
        group_targets = targets[group_mask]
        n_positive = int(group_targets.sum())
        n_negative = int((1 - group_targets).sum())
        if n_positive == 0 or n_negative == 0:
            raise ValueError(
                "ThresholdOptimizer requires both outcome classes inside each "
                f"calibration group. Group '{group_name}' has positives="
                f"{n_positive}, negatives={n_negative}."
            )
        group_counts[group_name] = {
            "n_samples": int(group_mask.sum()),
            "n_positive": n_positive,
            "n_negative": n_negative,
        }

    return scores, group_names, group_counts


def apply_binary_hardt_equalized_odds(
    calibration_scores: np.ndarray,
    calibration_targets: np.ndarray,
    calibration_groups: np.ndarray,
    evaluation_scores: np.ndarray,
    evaluation_groups: np.ndarray,
    strength: float = 1.0,
    seed: Optional[int] = None,
) -> dict:
    """Apply Fairlearn ThresholdOptimizer for binary equalized odds."""
    strength = _normalize_strength(strength)
    calibration_scores, calibration_group_names, group_counts = (
        _validate_binary_calibration_data(
            scores=calibration_scores,
            targets=calibration_targets,
            sensitive_features=calibration_groups,
        )
    )
    evaluation_scores = np.asarray(evaluation_scores, dtype=float).reshape(-1)
    evaluation_group_names = _normalize_group_names(
        np.asarray(evaluation_groups).reshape(-1)
    )
    if len(evaluation_scores) != len(evaluation_group_names):
        raise ValueError(
            "evaluation_scores and evaluation_groups must have the same length."
        )

    missing_groups = sorted(
        {group for group in evaluation_group_names if group not in group_counts}
    )
    if missing_groups:
        raise ValueError(
            f"Encountered unseen groups at transform time: {missing_groups}."
        )

    calibration_probability_matrix = _score_vector_to_probability_matrix(
        calibration_scores
    )
    estimator = _ProbabilityPassthroughEstimator().fit(
        calibration_probability_matrix,
        calibration_targets,
    )
    threshold_optimizer = ThresholdOptimizer(
        estimator=estimator,
        constraints="equalized_odds",
        objective=DEFAULT_OBJECTIVE,
        grid_size=DEFAULT_GRID_SIZE,
        flip=DEFAULT_FLIP,
        prefit=True,
        predict_method="predict_proba",
    )
    threshold_optimizer.fit(
        calibration_probability_matrix,
        calibration_targets,
        sensitive_features=calibration_group_names,
    )

    predicted_positive = threshold_optimizer.predict(
        _score_vector_to_probability_matrix(evaluation_scores),
        sensitive_features=evaluation_group_names,
        random_state=seed,
    ).astype(int)
    original_predictions = (evaluation_scores > 0.5).astype(int)
    blended_predictions, strength_summary = _mix_predictions_with_strength(
        original_predictions=original_predictions,
        adjusted_predictions=predicted_positive,
        strength=strength,
        seed=seed,
    )

    interpolation_dict = threshold_optimizer.interpolated_thresholder_.interpolation_dict
    return {
        "predictions": blended_predictions.astype(int),
        "summary": {
            "implementation": "fairlearn_threshold_optimizer_binary",
            "library": "fairlearn",
            "library_version": fairlearn.__version__,
            "constraints": "equalized_odds",
            "objective": DEFAULT_OBJECTIVE,
            "grid_size": DEFAULT_GRID_SIZE,
            "flip": DEFAULT_FLIP,
            "decision_output": "hard_labels_only",
            "strength_application": strength_summary,
            "groups": {
                group_name: {
                    **group_counts[group_name],
                    "interpolation": _serialize_interpolation(
                        interpolation_dict[group_name]
                    ),
                }
                for group_name in sorted(group_counts)
            },
        },
    }


def apply_multiclass_hardt_equalized_odds_ovr(
    calibration_score_matrix: np.ndarray,
    calibration_targets: np.ndarray,
    calibration_groups: np.ndarray,
    evaluation_score_matrix: np.ndarray,
    evaluation_groups: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    strength: float = 1.0,
    seed: Optional[int] = None,
) -> dict:
    """Apply binary ThresholdOptimizer one-vs-rest with hard-label resolution."""
    strength = _normalize_strength(strength)
    calibration_score_matrix = np.asarray(calibration_score_matrix, dtype=float)
    evaluation_score_matrix = np.asarray(evaluation_score_matrix, dtype=float)
    calibration_targets = np.asarray(calibration_targets).reshape(-1)
    calibration_groups = np.asarray(calibration_groups).reshape(-1)
    evaluation_groups = np.asarray(evaluation_groups).reshape(-1)

    if calibration_score_matrix.ndim != 2 or evaluation_score_matrix.ndim != 2:
        raise ValueError(
            "Expected score matrices with shape (n_samples, n_classes)."
        )
    if calibration_score_matrix.shape[1] != evaluation_score_matrix.shape[1]:
        raise ValueError(
            "Calibration and evaluation score matrices must have the same number "
            "of classes."
        )
    if len(calibration_score_matrix) != len(calibration_targets):
        raise ValueError(
            "calibration_score_matrix and calibration_targets must have the same "
            "length."
        )
    if len(calibration_score_matrix) != len(calibration_groups):
        raise ValueError(
            "calibration_score_matrix and calibration_groups must have the same "
            "length."
        )
    if len(evaluation_score_matrix) != len(evaluation_groups):
        raise ValueError(
            "evaluation_score_matrix and evaluation_groups must have the same "
            "length."
        )

    n_classes = int(calibration_score_matrix.shape[1])
    resolved_class_names = (
        list(class_names)
        if class_names is not None
        else [f"class_{class_idx}" for class_idx in range(n_classes)]
    )

    original_predictions = evaluation_score_matrix.argmax(axis=1)
    adjusted_binary_predictions = np.zeros_like(evaluation_score_matrix, dtype=int)
    class_status = {}
    for class_idx in range(n_classes):
        binary_targets = (calibration_targets == class_idx).astype(int)
        try:
            binary_result = apply_binary_hardt_equalized_odds(
                calibration_scores=calibration_score_matrix[:, class_idx],
                calibration_targets=binary_targets,
                calibration_groups=calibration_groups,
                evaluation_scores=evaluation_score_matrix[:, class_idx],
                evaluation_groups=evaluation_groups,
                strength=1.0,
                seed=seed,
            )
        except ValueError as exc:
            class_status[str(class_idx)] = {
                "class_name": resolved_class_names[class_idx],
                "status": "skipped",
                "reason": str(exc),
            }
            continue

        adjusted_binary_predictions[:, class_idx] = binary_result["predictions"]
        class_status[str(class_idx)] = {
            "class_name": resolved_class_names[class_idx],
            "status": "applied",
            "summary": binary_result["summary"],
        }

    predictions = np.array(original_predictions, copy=True)
    positive_counts = adjusted_binary_predictions.sum(axis=1)

    exactly_one_positive = positive_counts == 1
    if exactly_one_positive.any():
        predictions[exactly_one_positive] = adjusted_binary_predictions[
            exactly_one_positive
        ].argmax(axis=1)

    multiple_positive = positive_counts > 1
    if multiple_positive.any():
        multi_indices = np.where(multiple_positive)[0]
        for sample_idx in multi_indices:
            active_classes = np.flatnonzero(adjusted_binary_predictions[sample_idx])
            best_active_class = active_classes[
                evaluation_score_matrix[sample_idx, active_classes].argmax()
            ]
            predictions[sample_idx] = int(best_active_class)
    blended_predictions, strength_summary = _mix_predictions_with_strength(
        original_predictions=original_predictions,
        adjusted_predictions=predictions,
        strength=strength,
        seed=seed,
    )

    return {
        "predictions": blended_predictions.astype(int),
        "summary": {
            "implementation": "fairlearn_threshold_optimizer_multiclass_one_vs_rest",
            "library": "fairlearn",
            "library_version": fairlearn.__version__,
            "constraints": "equalized_odds",
            "objective": DEFAULT_OBJECTIVE,
            "grid_size": DEFAULT_GRID_SIZE,
            "flip": DEFAULT_FLIP,
            "n_classes": n_classes,
            "decision_output": "hard_labels_only",
            "strength_application": strength_summary,
            "multiclass_resolution": {
                "exactly_one_positive": "select_that_class",
                "multiple_positive": "break_ties_with_original_score",
                "no_positive": "fallback_to_original_argmax",
            },
            "class_status": class_status,
        },
    }


def save_postprocessing_summary(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
