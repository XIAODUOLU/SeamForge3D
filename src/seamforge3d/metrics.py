from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def seam_field_metrics(prediction: dict[str, np.ndarray], target: dict[str, np.ndarray], threshold: float = 0.3) -> dict[str, float]:
    probability = 1 / (1 + np.exp(-np.clip(prediction["seam_logit"], -30, 30)))
    predicted = probability >= threshold
    truth = target["seam_heat"] >= threshold
    tp = np.count_nonzero(predicted & truth)
    predicted_count, truth_count = np.count_nonzero(predicted), np.count_nonzero(truth)
    precision = tp / predicted_count if predicted_count else 1.0
    recall = tp / truth_count if truth_count else 1.0
    false_positive_rate = np.count_nonzero(predicted & ~truth) / max(np.count_nonzero(~truth), 1)
    active = target["seam_distance"] < 0.012
    offset_error = np.linalg.norm(prediction["seam_offset"][active] - target["seam_offset"][active], axis=1).mean() if np.any(active) else np.nan
    dot = np.abs(np.sum(prediction["seam_tangent"][active] * target["seam_tangent"][active], axis=1)).clip(0, 1)
    tangent_error = np.rad2deg(np.arccos(dot)).mean() if np.any(active) else np.nan
    return {"heat_precision": float(precision), "heat_recall": float(recall), "heat_false_positive_rate": float(false_positive_rate), "offset_error_m": float(offset_error), "tangent_error_deg": float(tangent_error)}


def trajectory_metrics(predicted: list[np.ndarray], ground_truth: list[np.ndarray], tolerances: tuple[float, ...] = (0.003, 0.005)) -> dict[str, float]:
    if not predicted and not ground_truth:
        result = {"component_count_error": 0.0, "chamfer_m": 0.0, "hausdorff95_m": 0.0}
        for tolerance in tolerances:
            mm = int(round(tolerance * 1000))
            result.update({f"precision_{mm}mm": 1.0, f"recall_{mm}mm": 1.0, f"false_seam_fraction_{mm}mm": 0.0})
        return result
    if not predicted or not ground_truth:
        result = {"component_count_error": float(abs(len(predicted) - len(ground_truth))), "chamfer_m": float("inf"), "hausdorff95_m": float("inf")}
        for tolerance in tolerances:
            mm = int(round(tolerance * 1000))
            result.update({
                f"precision_{mm}mm": 0.0 if predicted else 1.0,
                f"recall_{mm}mm": 0.0 if ground_truth else 1.0,
                f"false_seam_fraction_{mm}mm": 1.0 if predicted and not ground_truth else 0.0,
            })
        return result
    pred = np.concatenate(predicted)
    gt = np.concatenate(ground_truth)
    pred_to_gt = cKDTree(gt).query(pred, k=1)[0]
    gt_to_pred = cKDTree(pred).query(gt, k=1)[0]
    result = {
        "chamfer_m": float(0.5 * (pred_to_gt.mean() + gt_to_pred.mean())),
        "hausdorff95_m": float(max(np.quantile(pred_to_gt, 0.95), np.quantile(gt_to_pred, 0.95))),
        "component_count_error": float(abs(len(predicted) - len(ground_truth))),
    }
    for tolerance in tolerances:
        mm = int(round(tolerance * 1000))
        result[f"precision_{mm}mm"] = float(np.mean(pred_to_gt <= tolerance))
        result[f"recall_{mm}mm"] = float(np.mean(gt_to_pred <= tolerance))
        result[f"false_seam_fraction_{mm}mm"] = float(np.mean(pred_to_gt > tolerance))
    return result
