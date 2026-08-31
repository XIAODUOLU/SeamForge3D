from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def seam_field_metrics(prediction: dict[str, np.ndarray], target: dict[str, np.ndarray], threshold: float = 0.3) -> dict[str, float]:
    probability = 1 / (1 + np.exp(-np.clip(prediction["seam_logit"], -30, 30)))
    predicted = probability >= threshold
    truth = target["seam_heat"] >= threshold
    tp = np.count_nonzero(predicted & truth)
    precision = tp / max(np.count_nonzero(predicted), 1)
    recall = tp / max(np.count_nonzero(truth), 1)
    active = target["seam_distance"] < 0.012
    offset_error = np.linalg.norm(prediction["seam_offset"][active] - target["seam_offset"][active], axis=1).mean() if np.any(active) else np.nan
    dot = np.abs(np.sum(prediction["seam_tangent"][active] * target["seam_tangent"][active], axis=1)).clip(0, 1)
    tangent_error = np.rad2deg(np.arccos(dot)).mean() if np.any(active) else np.nan
    return {"heat_precision": float(precision), "heat_recall": float(recall), "offset_error_m": float(offset_error), "tangent_error_deg": float(tangent_error)}


def trajectory_metrics(predicted: list[np.ndarray], ground_truth: list[np.ndarray], tolerances: tuple[float, ...] = (0.003, 0.005)) -> dict[str, float]:
    if not predicted or not ground_truth:
        return {"component_count_error": float(abs(len(predicted) - len(ground_truth))), "chamfer_m": float("inf"), "hausdorff95_m": float("inf")}
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

