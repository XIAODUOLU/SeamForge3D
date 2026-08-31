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


@dataclass(frozen=True)
class Assembly:
    meshes: tuple[MeshData, ...]
    seams: tuple[SeamCurve, ...]
    transform: np.ndarray

    def transformed(self, transform: np.ndarray) -> "Assembly":
        composed = transform @ self.transform
        meshes = tuple(mesh.transformed(transform) for mesh in self.meshes)
        seams = []
        for seam in self.seams:
            h = np.c_[seam.points, np.ones(len(seam.points))]
            points = (h @ transform.T)[:, :3].astype(np.float32)
            seams.append(SeamCurve(points, seam.topology, seam.part_pair, seam.trajectory_id))
        return Assembly(meshes, tuple(seams), composed)


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
    return Assembly((base, web), seams, np.eye(4, dtype=np.float32))


def random_rigid_transform(rng: np.random.Generator, max_translation: float = 0.04) -> np.ndarray:
    angles = rng.uniform(-np.pi, np.pi, size=3)
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = (rz @ ry @ rx).astype(np.float32)
    out[:3, 3] = rng.uniform(-max_translation, max_translation, size=3)
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

