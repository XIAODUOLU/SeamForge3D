from __future__ import annotations

from pathlib import Path

import numpy as np


def write_point_ply(path: str | Path, coord: np.ndarray, normal: np.ndarray | None = None, colors: np.ndarray | None = None, scalars: dict[str, np.ndarray] | None = None) -> None:
    path = Path(path)
    n = len(coord)
    normal = np.zeros_like(coord) if normal is None else normal
    colors = np.full((n, 3), 180, dtype=np.uint8) if colors is None else np.clip(colors, 0, 255).astype(np.uint8)
    scalars = scalars or {}
    header = [
        "ply", "format ascii 1.0", f"element vertex {n}",
        "property float x", "property float y", "property float z",
        "property float nx", "property float ny", "property float nz",
        "property uchar red", "property uchar green", "property uchar blue",
    ]
    for name in scalars:
        header.append(f"property float {name}")
    header.append("end_header")
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(header) + "\n")
        values = np.column_stack((coord, normal, colors, *[v.reshape(-1, 1) for v in scalars.values()]))
        format_spec = ["%.8f"] * 6 + ["%d"] * 3 + ["%.8f"] * len(scalars)
        np.savetxt(handle, values, fmt=format_spec)


def heat_colors(heat: np.ndarray) -> np.ndarray:
    heat = np.clip(heat.reshape(-1), 0.0, 1.0)
    return np.column_stack((255 * heat, 40 * (1 - heat), 255 * (1 - heat))).astype(np.uint8)

