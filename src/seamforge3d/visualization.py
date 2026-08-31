from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry.assets import Assembly
from .io import heat_colors, write_point_ply


def _equal_axes(ax, points: np.ndarray) -> None:
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) / 2
    radius = max(hi - lo) / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def visualize_assembly(assembly: Assembly, output: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["#6baed6", "#fd8d3c"]
    all_points = []
    for mesh, color in zip(assembly.meshes, colors):
        polygons = mesh.vertices[mesh.triangles]
        ax.add_collection3d(Poly3DCollection(polygons, facecolor=color, alpha=0.38, edgecolor="#333333", linewidth=0.25))
        all_points.append(mesh.vertices)
    for seam in assembly.seams:
        ax.plot(*seam.points.T, color="#e31a1c", linewidth=4, label=f"seam {seam.trajectory_id}")
    _equal_axes(ax, np.concatenate(all_points))
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.legend(); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def visualize_labels(sample: dict[str, np.ndarray], output: str | Path, max_points: int = 60000) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coord, heat = sample["coord"], sample["seam_heat"]
    if len(coord) > max_points:
        index = np.linspace(0, len(coord) - 1, max_points).astype(int)
        coord, heat = coord[index], heat[index]
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot = ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c=heat, cmap="turbo", s=2, vmin=0, vmax=1)
    if "seam_points" in sample:
        offsets = sample["seam_offsets"]
        for i in range(len(offsets) - 1):
            seam = sample["seam_points"][offsets[i] : offsets[i + 1]]
            ax.plot(*seam.T, color="black", linewidth=2)
    _equal_axes(ax, coord)
    fig.colorbar(plot, ax=ax, label="GT seam heat")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def export_label_ply(sample: dict[str, np.ndarray], output: str | Path) -> None:
    write_point_ply(
        output, sample["coord"], sample["normal"], heat_colors(sample["seam_heat"]),
        {"seam_heat": sample["seam_heat"], "seam_distance": sample["seam_distance"], "instance_id": sample["instance_id"]},
    )

