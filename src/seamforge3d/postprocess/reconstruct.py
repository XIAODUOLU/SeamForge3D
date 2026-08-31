from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ReconstructionConfig:
    probability_threshold: float = 0.30
    merge_voxel: float = 0.002
    graph_radius: float = 0.006
    tangent_angle_deg: float = 30.0
    min_component_length: float = 0.015
    bridge_gap: float = 0.010
    sample_spacing: float = 0.002


@dataclass
class Trajectory:
    trajectory_id: int
    confidence: float
    topology: str
    part_pair: tuple[int, int] | None
    points: np.ndarray
    graph_id: int = 0
    junction_start_id: int | None = None
    junction_end_id: int | None = None


def _merge_votes(coord: np.ndarray, probability: np.ndarray, tangent: np.ndarray, embedding: np.ndarray, voxel: float):
    origin = coord.min(axis=0)
    key = np.floor((coord - origin) / voxel).astype(np.int64)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    count = np.bincount(inverse)
    weight = probability.clip(1e-4)
    weight_sum = np.bincount(inverse, weights=weight)
    merged_coord = np.stack([np.bincount(inverse, weights=coord[:, d] * weight) / weight_sum for d in range(3)], axis=1)
    merged_probability = np.bincount(inverse, weights=probability) / count
    # Tangents are unoriented. Align each voxel's vectors to its first vector before averaging.
    merged_tangent = np.zeros((len(count), 3), dtype=np.float64)
    merged_embedding = np.zeros((len(count), embedding.shape[1]), dtype=np.float64)
    for group in range(len(count)):
        indices = np.flatnonzero(inverse == group)
        signs = np.sign(tangent[indices] @ tangent[indices[0]])
        signs[signs == 0] = 1
        merged_tangent[group] = np.average(tangent[indices] * signs[:, None], axis=0, weights=weight[indices])
        merged_embedding[group] = np.average(embedding[indices], axis=0, weights=weight[indices])
    merged_tangent /= np.linalg.norm(merged_tangent, axis=1, keepdims=True).clip(1e-8)
    return merged_coord, merged_probability, merged_tangent, merged_embedding


def _edges(coord: np.ndarray, tangent: np.ndarray, radius: float, angle_deg: float, bridge_gap: float) -> set[tuple[int, int]]:
    cosine = np.cos(np.deg2rad(angle_deg))
    tree = cKDTree(coord)
    edges: set[tuple[int, int]] = set()
    for i, neighbors in enumerate(tree.query_ball_point(coord, bridge_gap)):
        candidates = []
        for j in neighbors:
            if j == i:
                continue
            delta = coord[j] - coord[i]
            distance = np.linalg.norm(delta)
            if abs(np.dot(tangent[i], tangent[j])) < cosine:
                continue
            direction_alignment = abs(np.dot(delta / max(distance, 1e-12), tangent[i]))
            if distance <= radius or direction_alignment > cosine:
                candidates.append((j, distance, np.dot(delta, tangent[i])))
        # One closest continuation in each tangent half-space suppresses thick vote-cloud cross edges.
        for sign in (-1, 1):
            side = [item for item in candidates if item[2] * sign > 0]
            if side:
                j = min(side, key=lambda item: item[1])[0]
                edges.add((min(i, j), max(i, j)))
    return edges


def _components(n: int, edges: set[tuple[int, int]]) -> list[np.ndarray]:
    parent = np.arange(n)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for a, b in edges:
        union(a, b)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [np.asarray(value, dtype=np.int32) for value in groups.values()]


def _adjacency(nodes: np.ndarray, edges: set[tuple[int, int]]) -> dict[int, list[int]]:
    node_set = set(map(int, nodes))
    adjacency = {int(i): [] for i in nodes}
    for a, b in edges:
        if a in node_set and b in node_set:
            adjacency[a].append(b); adjacency[b].append(a)
    return adjacency


def _trace_segments(adjacency: dict[int, list[int]]) -> tuple[str, list[list[int]]]:
    active = {node: neighbors for node, neighbors in adjacency.items() if neighbors}
    if not active:
        return "open", []
    degree = {node: len(neighbors) for node, neighbors in active.items()}
    keys = {node for node, value in degree.items() if value != 2}
    visited: set[tuple[int, int]] = set()
    segments: list[list[int]] = []
    if not keys:
        start = next(iter(active))
        path = [start]
        previous, current = -1, start
        while True:
            choices = [n for n in active[current] if n != previous]
            if not choices:
                break
            nxt = choices[0]
            edge = (min(current, nxt), max(current, nxt))
            if edge in visited:
                break
            visited.add(edge); path.append(nxt)
            previous, current = current, nxt
            if current == start:
                break
        return "closed", [path]
    for start in keys:
        for neighbor in active[start]:
            edge = (min(start, neighbor), max(start, neighbor))
            if edge in visited:
                continue
            visited.add(edge)
            path = [start, neighbor]
            previous, current = start, neighbor
            while current not in keys:
                choices = [n for n in active[current] if n != previous]
                if not choices:
                    break
                nxt = choices[0]
                edge = (min(current, nxt), max(current, nxt))
                if edge in visited:
                    break
                visited.add(edge); path.append(nxt)
                previous, current = current, nxt
            segments.append(path)
    topology = "open" if len(keys) == 2 and all(degree[k] == 1 for k in keys) else "segment"
    return topology, segments


def _path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def _fit_resample(points: np.ndarray, spacing: float, closed: bool) -> np.ndarray:
    if closed and np.linalg.norm(points[0] - points[-1]) > 1e-9:
        points = np.vstack((points, points[0]))
    length = _path_length(points)
    count = max(2, int(np.ceil(length / spacing)) + (0 if closed else 1))
    if len(points) < 4 or length < spacing:
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        target = np.linspace(0, distance[-1], count, endpoint=not closed)
        return np.column_stack([np.interp(target, distance, points[:, d]) for d in range(3)]).astype(np.float32)
    try:
        smooth = max(len(points) * (spacing * 0.15) ** 2, 1e-12)
        spline, _ = splprep(points.T, s=smooth, k=min(3, len(points) - 1), per=closed)
        sampled = np.column_stack(splev(np.linspace(0, 1, count, endpoint=not closed), spline))
        # Second pass enforces physical arc-length spacing after spline parameterization.
        cumulative = np.r_[0, np.cumsum(np.linalg.norm(np.diff(sampled, axis=0), axis=1))]
        target = np.arange(0, cumulative[-1] + (0 if closed else spacing * 0.5), spacing)
        return np.column_stack([np.interp(target, cumulative, sampled[:, d]) for d in range(3)]).astype(np.float32)
    except (ValueError, TypeError):
        return points.astype(np.float32)


def _part_consistency(embedding: np.ndarray) -> tuple[float, tuple[int, int] | None]:
    if len(embedding) < 4:
        return 0.5, None
    centers = np.stack((embedding[0], embedding[-1])).astype(np.float64)
    assignment = np.zeros(len(embedding), dtype=np.int32)
    for _ in range(12):
        distance = np.linalg.norm(embedding[:, None, :] - centers[None, :, :], axis=2)
        assignment = np.argmin(distance, axis=1)
        if len(np.unique(assignment)) < 2:
            return 0.35, None
        new = np.stack([embedding[assignment == k].mean(axis=0) for k in range(2)])
        if np.allclose(new, centers):
            break
        centers = new
    separation = np.linalg.norm(centers[0] - centers[1])
    spread = np.mean([np.linalg.norm(embedding[assignment == k] - centers[k], axis=1).mean() for k in range(2)])
    confidence = float(1 / (1 + np.exp(-(separation / max(spread, 1e-6) - 1))))
    return confidence, (0, 1)


def _merge_aligned_segments(trajectories: list[Trajectory], gap_limit: float, angle_deg: float, spacing: float) -> list[Trajectory]:
    """Bridge short missing-data gaps without joining parallel adjacent seams."""
    cosine = np.cos(np.deg2rad(angle_deg))
    items = list(trajectories)
    while True:
        best: tuple[float, int, int, np.ndarray, np.ndarray] | None = None
        for i in range(len(items)):
            if len(items[i].points) < 2 or items[i].topology == "closed":
                continue
            for j in range(i + 1, len(items)):
                if len(items[j].points) < 2 or items[j].topology == "closed":
                    continue
                for reverse_i in (False, True):
                    left = items[i].points[::-1] if reverse_i else items[i].points
                    for reverse_j in (False, True):
                        right = items[j].points[::-1] if reverse_j else items[j].points
                        delta = right[0] - left[-1]
                        distance = np.linalg.norm(delta)
                        if distance <= 1e-9 or distance > gap_limit:
                            continue
                        direction = delta / distance
                        left_tangent = left[-1] - left[-2]
                        right_tangent = right[1] - right[0]
                        left_tangent /= np.linalg.norm(left_tangent) + 1e-12
                        right_tangent /= np.linalg.norm(right_tangent) + 1e-12
                        if np.dot(left_tangent, direction) < cosine or np.dot(right_tangent, direction) < cosine:
                            continue
                        candidate = (float(distance), i, j, left, right)
                        if best is None or candidate[0] < best[0]:
                            best = candidate
        if best is None:
            break
        _, i, j, left, right = best
        joined = _fit_resample(np.vstack((left, right)), spacing, closed=False)
        total_points = len(items[i].points) + len(items[j].points)
        confidence = (items[i].confidence * len(items[i].points) + items[j].confidence * len(items[j].points)) / total_points
        merged = Trajectory(
            trajectory_id=0, confidence=float(confidence), topology="open",
            part_pair=items[i].part_pair or items[j].part_pair, points=joined,
            graph_id=min(items[i].graph_id, items[j].graph_id),
        )
        items = [item for index, item in enumerate(items) if index not in (i, j)] + [merged]
    for trajectory_id, trajectory in enumerate(items):
        trajectory.trajectory_id = trajectory_id
    return items


class SeamGraphReconstructor:
    def __init__(self, config: ReconstructionConfig):
        self.config = config

    def __call__(self, coord: np.ndarray, prediction: dict[str, np.ndarray]) -> list[Trajectory]:
        probability = 1 / (1 + np.exp(-np.clip(prediction["seam_logit"], -30, 30)))
        junction_present = False
        if "junction_logit" in prediction:
            junction_probability = 1 / (1 + np.exp(-np.clip(prediction["junction_logit"], -30, 30)))
            junction_present = bool(np.any(junction_probability >= 0.5))
        mask = probability >= self.config.probability_threshold
        if np.count_nonzero(mask) < 2:
            return []
        votes = coord[mask] + prediction["seam_offset"][mask]
        votes, confidence, tangent, embedding = _merge_votes(
            votes, probability[mask], prediction["seam_tangent"][mask], prediction["part_embed"][mask], self.config.merge_voxel
        )
        edges = _edges(votes, tangent, self.config.graph_radius, self.config.tangent_angle_deg, self.config.bridge_gap)
        trajectories: list[Trajectory] = []
        for graph_id, component in enumerate(_components(len(votes), edges)):
            adjacency = _adjacency(component, edges)
            topology, segments = _trace_segments(adjacency)
            for path in segments:
                points = votes[path]
                length = _path_length(points)
                if length < self.config.min_component_length:
                    continue
                closed = topology == "closed"
                fitted = _fit_resample(points, self.config.sample_spacing, closed)
                part_conf, pair = _part_consistency(embedding[path])
                score = float(np.mean(confidence[path]) * (0.5 + 0.5 * part_conf))
                resolved_topology = "open" if topology == "segment" and not junction_present else topology
                trajectories.append(Trajectory(
                    trajectory_id=len(trajectories), confidence=score, topology=resolved_topology,
                    part_pair=pair, points=fitted, graph_id=graph_id,
                    junction_start_id=path[0] if resolved_topology == "segment" else None,
                    junction_end_id=path[-1] if resolved_topology == "segment" else None,
                ))
        return _merge_aligned_segments(
            trajectories,
            gap_limit=self.config.bridge_gap * 1.5,
            # Endpoint tangents from short fitted fragments are noisier than
            # point-wise graph tangents, so use a moderately wider bridge cone.
            angle_deg=min(60.0, self.config.tangent_angle_deg * 1.5),
            spacing=self.config.sample_spacing,
        )
