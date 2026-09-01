from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .assets import SeamCurve


def _fill_short_false_runs(mask: np.ndarray, max_gap: int, closed: bool) -> np.ndarray:
    result = mask.copy()
    if max_gap <= 0 or not np.any(result):
        return result
    work = np.tile(result, 3) if closed else result
    start = None
    for index, value in enumerate(np.r_[work, True]):
        if not value and start is None:
            start = index
        elif value and start is not None:
            if index - start <= max_gap and start > 0 and index < len(work):
                work[start:index] = True
            start = None
    return work[len(result):2 * len(result)] if closed else work


def _visible_runs(mask: np.ndarray, closed: bool) -> list[np.ndarray]:
    if not np.any(mask):
        return []
    if closed and np.all(mask):
        return [np.arange(len(mask), dtype=np.int32)]
    if closed:
        first_false = int(np.flatnonzero(~mask)[0])
        order = np.roll(np.arange(len(mask)), -(first_false + 1))
        ordered_mask = mask[order]
    else:
        order = np.arange(len(mask))
        ordered_mask = mask
    changes = np.diff(np.r_[False, ordered_mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    return [order[start:end] for start, end in zip(starts, ends)]


def filter_visible_seams(
    seams: tuple[SeamCurve, ...], raw_coord: np.ndarray, raw_instance_id: np.ndarray,
    support_radius: float, min_visible_length: float, max_gap_points: int = 4,
) -> tuple[tuple[SeamCurve, ...], np.ndarray]:
    """Keep only trajectory runs supported by observations from both workpieces.

    This intentionally evaluates visibility in the rendered point cloud rather
    than against complete CAD geometry. It therefore captures camera occlusion,
    field-of-view clipping, grazing dropout, and synthetic patch dropout.
    """
    instance_trees = {
        int(instance): cKDTree(raw_coord[raw_instance_id == instance])
        for instance in np.unique(raw_instance_id)
        if np.any(raw_instance_id == instance)
    }
    visible: list[SeamCurve] = []
    fractions = np.zeros(len(seams), dtype=np.float32)
    next_id = 0
    for seam_index, seam in enumerate(seams):
        trees = [instance_trees.get(int(instance)) for instance in seam.part_pair]
        if any(tree is None for tree in trees):
            continue
        supported = np.ones(len(seam.points), dtype=bool)
        for tree in trees:
            supported &= tree.query(seam.points, k=1, workers=-1)[0] <= support_radius
        supported = _fill_short_false_runs(supported, max_gap_points, seam.topology == "closed")
        segment_length = np.linalg.norm(np.diff(seam.points, axis=0), axis=1)
        if len(segment_length):
            edge_supported = supported[:-1] & supported[1:]
            fractions[seam_index] = float(segment_length[edge_supported].sum() / max(segment_length.sum(), 1e-12))
        for indices in _visible_runs(supported, seam.topology == "closed"):
            if len(indices) < 2:
                continue
            points = seam.points[indices]
            length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            if length < min_visible_length:
                continue
            complete_loop = seam.topology == "closed" and len(indices) == len(seam.points)
            topology = "closed" if complete_loop else "open"
            visible.append(SeamCurve(
                points.astype(np.float32), topology, seam.part_pair, next_id,
                seam.trajectory_id if seam.source_trajectory_id is None else seam.source_trajectory_id,
            ))
            next_id += 1
    return tuple(visible), fractions
