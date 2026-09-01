from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MeshData:
    name: str
    vertices: np.ndarray
    triangles: np.ndarray
    instance_id: int

    def transformed(self, transform: np.ndarray) -> "MeshData":
        homogeneous = np.c_[self.vertices, np.ones(len(self.vertices))]
        vertices = (homogeneous @ transform.T)[:, :3].astype(np.float32)
        return MeshData(self.name, vertices, self.triangles.copy(), self.instance_id)

    def write_obj(self, path: str | Path) -> None:
        path = Path(path)
        lines = [f"o {self.name}"]
        lines.extend(f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in self.vertices)
        lines.extend("f " + " ".join(str(int(i) + 1) for i in face) for face in self.triangles)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class SeamCurve:
    points: np.ndarray
    topology: str
    part_pair: tuple[int, int]
    trajectory_id: int
    source_trajectory_id: int | None = None


@dataclass(frozen=True)
class Assembly:
    meshes: tuple[MeshData, ...]
    seams: tuple[SeamCurve, ...]
    transform: np.ndarray
    family: str = "t_joint"

    def transformed(self, transform: np.ndarray) -> "Assembly":
        composed = transform @ self.transform
        meshes = tuple(mesh.transformed(transform) for mesh in self.meshes)
        seams = []
        for seam in self.seams:
            h = np.c_[seam.points, np.ones(len(seam.points))]
            points = (h @ transform.T)[:, :3].astype(np.float32)
            seams.append(SeamCurve(points, seam.topology, seam.part_pair, seam.trajectory_id, seam.source_trajectory_id))
        return Assembly(meshes, tuple(seams), composed, self.family)


def cuboid(name: str, size: tuple[float, float, float], center: tuple[float, float, float], instance_id: int) -> MeshData:
    sx, sy, sz = np.asarray(size, dtype=np.float32) / 2
    cx, cy, cz = np.asarray(center, dtype=np.float32)
    vertices = np.array(
        [
            [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
            [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz],
        ], dtype=np.float32,
    ) + np.array([cx, cy, cz], dtype=np.float32)
    triangles = np.array(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ], dtype=np.int32,
    )
    return MeshData(name, vertices, triangles, instance_id)


def create_seed_assembly(seam_samples: int = 321) -> Assembly:
    """Create two independent watertight meshes forming a double-sided T joint.

    The base top is z=0 and the web bottom is z=0. The two physical fillet seam
    centerlines are explicitly defined at the web/base intersection boundaries.
    """
    base = cuboid("base_plate", (0.220, 0.140, 0.010), (0.0, 0.0, -0.005), 0)
    web = cuboid("vertical_web", (0.170, 0.008, 0.085), (0.0, 0.0, 0.0425), 1)
    x = np.linspace(-0.085, 0.085, seam_samples, dtype=np.float32)
    left = np.column_stack((x, np.full_like(x, -0.004), np.zeros_like(x)))
    right = np.column_stack((x, np.full_like(x, 0.004), np.zeros_like(x)))
    seams = (
        SeamCurve(left, "open", (0, 1), 0),
        SeamCurve(right, "open", (0, 1), 1),
    )
    return Assembly((base, web), seams, np.eye(4, dtype=np.float32), "t_joint")


def concatenate_meshes(name: str, meshes: tuple[MeshData, ...], instance_id: int) -> MeshData:
    vertices, triangles, offset = [], [], 0
    for mesh in meshes:
        vertices.append(mesh.vertices)
        triangles.append(mesh.triangles + offset)
        offset += len(mesh.vertices)
    return MeshData(name, np.concatenate(vertices), np.concatenate(triangles), instance_id)


def triangular_prism(name: str, length: float, width: float, height: float, instance_id: int) -> MeshData:
    x0, x1 = -length / 2, length / 2
    yz = np.array([[-width / 2, 0], [width / 2, 0], [0, height]], dtype=np.float32)
    vertices = np.vstack((np.c_[np.full(3, x0), yz], np.c_[np.full(3, x1), yz])).astype(np.float32)
    triangles = np.array([
        [0, 2, 1], [3, 4, 5],
        [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5],
    ], dtype=np.int32)
    return MeshData(name, vertices, triangles, instance_id)


def cylinder(name: str, radius: float, height: float, center_xy: tuple[float, float], instance_id: int, segments: int = 64) -> MeshData:
    angle = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.column_stack((center_xy[0] + radius * np.cos(angle), center_xy[1] + radius * np.sin(angle)))
    bottom = np.c_[ring, np.zeros(segments)]
    top = np.c_[ring, np.full(segments, height)]
    vertices = np.vstack((bottom, top, [[center_xy[0], center_xy[1], 0]], [[center_xy[0], center_xy[1], height]])).astype(np.float32)
    bottom_center, top_center = 2 * segments, 2 * segments + 1
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.extend(([i, j, segments + j], [i, segments + j, segments + i], [bottom_center, j, i], [top_center, segments + i, segments + j]))
    return MeshData(name, vertices, np.asarray(faces, dtype=np.int32), instance_id)


def cone(name: str, radius: float, height: float, center_xy: tuple[float, float], instance_id: int, segments: int = 64) -> MeshData:
    angle = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.column_stack((center_xy[0] + radius * np.cos(angle), center_xy[1] + radius * np.sin(angle), np.zeros(segments)))
    vertices = np.vstack((ring, [[center_xy[0], center_xy[1], 0]], [[center_xy[0], center_xy[1], height]])).astype(np.float32)
    base_center, apex = segments, segments + 1
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.extend(([base_center, j, i], [i, j, apex]))
    return MeshData(name, vertices, np.asarray(faces, dtype=np.int32), instance_id)


def i_beam(name: str, length: float, width: float, height: float, flange: float, web: float, instance_id: int) -> MeshData:
    pieces = (
        cuboid("bottom_flange", (length, width, flange), (0, 0, flange / 2), instance_id),
        cuboid("web", (length, web, height - 2 * flange), (0, 0, height / 2), instance_id),
        cuboid("top_flange", (length, width, flange), (0, 0, height - flange / 2), instance_id),
    )
    return concatenate_meshes(name, pieces, instance_id)


def l_angle(name: str, length: float, width: float, height: float, thickness: float, instance_id: int) -> MeshData:
    pieces = (
        cuboid("angle_foot", (length, width, thickness), (0, 0, thickness / 2), instance_id),
        cuboid("angle_web", (length, thickness, height), (0, -width / 2 + thickness / 2, height / 2), instance_id),
    )
    return concatenate_meshes(name, pieces, instance_id)


def _line_pair(length: float, half_width: float, samples: int, pair: tuple[int, int]) -> tuple[SeamCurve, SeamCurve]:
    x = np.linspace(-length / 2, length / 2, samples, dtype=np.float32)
    return (
        SeamCurve(np.column_stack((x, np.full_like(x, -half_width), np.zeros_like(x))), "open", pair, 0, 0),
        SeamCurve(np.column_stack((x, np.full_like(x, half_width), np.zeros_like(x))), "open", pair, 1, 1),
    )


def _closed_circle(radius: float, center_xy: tuple[float, float], samples: int, pair: tuple[int, int]) -> tuple[SeamCurve, ...]:
    angle = np.linspace(0, 2 * np.pi, samples, endpoint=True, dtype=np.float32)
    points = np.column_stack((center_xy[0] + radius * np.cos(angle), center_xy[1] + radius * np.sin(angle), np.zeros_like(angle)))
    return (SeamCurve(points.astype(np.float32), "closed", pair, 0, 0),)


def _yaw_part(mesh: MeshData, seams: tuple[SeamCurve, ...], angle: float) -> tuple[MeshData, tuple[SeamCurve, ...]]:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], dtype=np.float32)
    transformed_mesh = mesh.transformed(transform)
    transformed_seams = []
    for seam in seams:
        points = (np.c_[seam.points, np.ones(len(seam.points))] @ transform.T)[:, :3].astype(np.float32)
        transformed_seams.append(SeamCurve(points, seam.topology, seam.part_pair, seam.trajectory_id, seam.source_trajectory_id))
    return transformed_mesh, tuple(transformed_seams)


PROCEDURAL_FAMILIES = ("t_joint", "triangular_prism", "cylinder", "cone", "i_beam", "l_angle", "box")


def create_procedural_assembly(rng: np.random.Generator, families: tuple[str, ...] = PROCEDURAL_FAMILIES, seam_samples: int = 321) -> Assembly:
    """Create a dimension-randomized two-workpiece assembly in metres."""
    family = str(rng.choice(families))
    base_length = float(rng.uniform(0.22, 0.34))
    base_width = float(rng.uniform(0.16, 0.26))
    base_thickness = float(rng.uniform(0.008, 0.018))
    base = cuboid("base_plate", (base_length, base_width, base_thickness), (0, 0, -base_thickness / 2), 0)
    pair = (0, 1)
    length = float(rng.uniform(base_length * 0.48, base_length * 0.82))
    width = float(rng.uniform(0.026, min(0.075, base_width * 0.42)))
    height = float(rng.uniform(0.045, 0.125))
    if family == "t_joint":
        thickness = float(rng.uniform(0.005, 0.014))
        part = cuboid("vertical_web", (length, thickness, height), (0, 0, height / 2), 1)
        seams = _line_pair(length, thickness / 2, seam_samples, pair)
    elif family == "triangular_prism":
        part = triangular_prism("triangular_prism", length, width, height, 1)
        seams = _line_pair(length, width / 2, seam_samples, pair)
    elif family == "i_beam":
        flange = float(rng.uniform(0.005, min(0.014, height * 0.16)))
        web = float(rng.uniform(0.004, min(0.012, width * 0.28)))
        part = i_beam("i_beam", length, width, height, flange, web, 1)
        seams = _line_pair(length, width / 2, seam_samples, pair)
    elif family == "l_angle":
        thickness = float(rng.uniform(0.004, min(0.012, width * 0.28)))
        part = l_angle("l_angle", length, width, height, thickness, 1)
        seams = _line_pair(length, width / 2, seam_samples, pair)
    elif family in ("cylinder", "cone"):
        radius = float(rng.uniform(0.022, min(0.055, base_width * 0.28)))
        center_xy = (float(rng.uniform(-base_length * 0.10, base_length * 0.10)), float(rng.uniform(-base_width * 0.10, base_width * 0.10)))
        part = cylinder("cylinder", radius, height, center_xy, 1) if family == "cylinder" else cone("cone", radius, height, center_xy, 1)
        seams = _closed_circle(radius, center_xy, seam_samples, pair)
    elif family == "box":
        part = cuboid("upright_box", (length, width, height), (0, 0, height / 2), 1)
        seams = _line_pair(length, width / 2, seam_samples, pair)
    else:
        raise ValueError(f"Unknown procedural family: {family}")
    if family not in ("cylinder", "cone"):
        part, seams = _yaw_part(part, seams, float(rng.uniform(-np.pi, np.pi)))
    return Assembly((base, part), seams, np.eye(4, dtype=np.float32), family)


def random_rigid_transform(rng: np.random.Generator, max_translation: float = 0.04, rotation_mode: str = "full") -> np.ndarray:
    if rotation_mode == "full":
        angles = rng.uniform(-np.pi, np.pi, size=3)
    elif rotation_mode == "yaw":
        angles = np.array([0.0, 0.0, rng.uniform(-np.pi, np.pi)])
    elif rotation_mode == "none":
        angles = np.zeros(3)
    else:
        raise ValueError(f"Unsupported assembly_rotation_mode: {rotation_mode}")
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = (rz @ ry @ rx).astype(np.float32)
    out[:3, 3] = rng.uniform(-max_translation, max_translation, size=3) if max_translation > 0 else 0
    return out


def export_seed_assets(output_dir: str | Path) -> Assembly:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assembly = create_seed_assembly()
    for mesh in assembly.meshes:
        mesh.write_obj(output_dir / f"{mesh.name}.obj")
    for seam in assembly.seams:
        np.save(output_dir / f"seam_{seam.trajectory_id:02d}.npy", seam.points)
    return assembly
