from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _kmeans2(features: np.ndarray, iterations: int = 15) -> tuple[np.ndarray, np.ndarray]:
    first = 0
    second = int(np.argmax(np.linalg.norm(features - features[first], axis=1)))
    centers = np.stack((features[first], features[second])).astype(np.float64)
    labels = np.zeros(len(features), dtype=np.int32)
    for _ in range(iterations):
        labels = np.argmin(np.linalg.norm(features[:, None] - centers[None], axis=2), axis=1)
        if len(np.unique(labels)) < 2:
            break
        updated = np.stack([features[labels == k].mean(axis=0) for k in range(2)])
        if np.allclose(updated, centers):
            break
        centers = updated
    return labels, centers


def refine_trajectory_on_dense_cloud(
    coarse: np.ndarray,
    raw_coord: np.ndarray,
    net_coord: np.ndarray,
    net_embedding: np.ndarray,
    roi_radius: float = 0.008,
    min_points_per_part: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine a coarse seam with local intersections of two embedding-separated surfaces."""
    net_tree = cKDTree(net_coord)
    raw_embedding = net_embedding[net_tree.query(raw_coord, k=1)[1]]
    raw_tree = cKDTree(raw_coord)
    refined = coarse.astype(np.float64).copy()
    confidence = np.zeros(len(coarse), dtype=np.float32)
    for index, point in enumerate(coarse):
        neighbors = np.asarray(raw_tree.query_ball_point(point, roi_radius), dtype=np.int64)
        if len(neighbors) < 2 * min_points_per_part:
            continue
        labels, centers = _kmeans2(raw_embedding[neighbors])
        groups = [raw_coord[neighbors[labels == group]].astype(np.float64) for group in range(2)]
        if min(map(len, groups)) < min_points_per_part:
            continue
        plane_normals, plane_offsets, planarity = [], [], []
        for group in groups:
            centroid = group.mean(axis=0)
            covariance = np.cov((group - centroid).T)
            values, vectors = np.linalg.eigh(covariance)
            normal = vectors[:, 0]
            plane_normals.append(normal)
            plane_offsets.append(float(normal @ centroid))
            planarity.append(float(1 - values[0] / max(values.sum(), 1e-12)))
        cross = np.cross(plane_normals[0], plane_normals[1])
        angle_strength = np.linalg.norm(cross)
        if angle_strength < 0.08:
            continue
        # Orthogonally project the coarse point onto the intersection of both local planes.
        matrix = np.stack(plane_normals)
        residual = np.asarray(plane_offsets) - matrix @ point
        correction = matrix.T @ np.linalg.solve(matrix @ matrix.T + np.eye(2) * 1e-8, residual)
        if np.linalg.norm(correction) <= roi_radius:
            refined[index] = point + correction
            embedding_separation = np.linalg.norm(centers[0] - centers[1])
            confidence[index] = float(np.clip(angle_strength * min(planarity) * embedding_separation, 0, 1))
    # Confidence-weighted three-tap smoothing only on the correction, retaining corners in the coarse curve.
    correction = refined - coarse
    if len(correction) >= 3:
        smooth = correction.copy()
        smooth[1:-1] = 0.25 * correction[:-2] + 0.5 * correction[1:-1] + 0.25 * correction[2:]
        refined = coarse + smooth * confidence[:, None]
    return refined.astype(np.float32), confidence

