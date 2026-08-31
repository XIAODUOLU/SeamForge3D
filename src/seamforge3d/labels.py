from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry.assets import SeamCurve


@dataclass(frozen=True)
class SeamLabels:
    distance: np.ndarray
    heat: np.ndarray
    offset: np.ndarray
    tangent: np.ndarray
    nearest_trajectory: np.ndarray
    endpoint_heat: np.ndarray
    junction_heat: np.ndarray


def _nearest_polyline(points: np.ndarray, polyline: np.ndarray, chunk_size: int = 8192) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = polyline[:-1].astype(np.float64)
    vectors = (polyline[1:] - polyline[:-1]).astype(np.float64)
    lengths2 = np.einsum("ij,ij->i", vectors, vectors).clip(1e-16)
    best_distance2 = np.full(len(points), np.inf, dtype=np.float64)
    best_point = np.zeros((len(points), 3), dtype=np.float64)
    best_tangent = np.zeros((len(points), 3), dtype=np.float64)
    unit_tangent = vectors / np.sqrt(lengths2)[:, None]
    for begin in range(0, len(points), chunk_size):
        query = points[begin : begin + chunk_size].astype(np.float64)
        delta = query[:, None, :] - starts[None, :, :]
        u = np.einsum("csd,sd->cs", delta, vectors) / lengths2[None, :]
        u = np.clip(u, 0.0, 1.0)
        projected = starts[None, :, :] + u[..., None] * vectors[None, :, :]
        distance2 = np.einsum("csd,csd->cs", query[:, None, :] - projected, query[:, None, :] - projected)
        segment = np.argmin(distance2, axis=1)
        row = np.arange(len(query))
        sl = slice(begin, begin + len(query))
        best_distance2[sl] = distance2[row, segment]
        best_point[sl] = projected[row, segment]
        best_tangent[sl] = unit_tangent[segment]
    return np.sqrt(best_distance2), best_point, best_tangent


def build_seam_labels(points: np.ndarray, seams: tuple[SeamCurve, ...], sigma: float, supervision_radius: float) -> SeamLabels:
    if not seams:
        zeros = np.zeros(len(points), dtype=np.float32)
        return SeamLabels(np.full(len(points), np.inf, np.float32), zeros, np.zeros((len(points), 3), np.float32), np.zeros((len(points), 3), np.float32), np.full(len(points), -1, np.int32), zeros, zeros)
    all_results = [_nearest_polyline(points, seam.points) for seam in seams]
    distances = np.stack([result[0] for result in all_results], axis=1)
    nearest = np.argmin(distances, axis=1)
    row = np.arange(len(points))
    distance = distances[row, nearest]
    projected = np.stack([result[1] for result in all_results], axis=1)[row, nearest]
    tangent = np.stack([result[2] for result in all_results], axis=1)[row, nearest]
    active = distance < supervision_radius
    offset = projected - points
    offset[~active] = 0.0
    tangent[~active] = 0.0
    heat = np.exp(-(distance ** 2) / (2 * sigma ** 2))
    heat[distance > supervision_radius] = 0.0

    endpoints = np.concatenate([np.stack((s.points[0], s.points[-1])) for s in seams if s.topology == "open"], axis=0)
    if len(endpoints):
        endpoint_distance = np.linalg.norm(points[:, None, :] - endpoints[None, :, :], axis=-1).min(axis=1)
        endpoint_heat = np.exp(-(endpoint_distance ** 2) / (2 * sigma ** 2))
    else:
        endpoint_heat = np.zeros(len(points))
    # Seed assembly has no junction. This field remains part of the stable schema.
    junction_heat = np.zeros(len(points))
    return SeamLabels(
        distance.astype(np.float32), heat.astype(np.float32), offset.astype(np.float32),
        tangent.astype(np.float32), nearest.astype(np.int32), endpoint_heat.astype(np.float32),
        junction_heat.astype(np.float32),
    )


def encode_seams(seams: tuple[SeamCurve, ...]) -> dict[str, np.ndarray]:
    lengths = np.array([len(seam.points) for seam in seams], dtype=np.int32)
    offsets = np.r_[0, np.cumsum(lengths)].astype(np.int32)
    return {
        "seam_points": np.concatenate([seam.points for seam in seams], axis=0).astype(np.float32),
        "seam_offsets": offsets,
        "seam_topology": np.array([0 if seam.topology == "open" else 1 for seam in seams], dtype=np.int8),
        "seam_part_pairs": np.array([seam.part_pair for seam in seams], dtype=np.int32),
        "seam_trajectory_ids": np.array([seam.trajectory_id for seam in seams], dtype=np.int32),
    }

