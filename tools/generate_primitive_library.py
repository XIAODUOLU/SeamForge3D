#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/seamforge3d-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/seamforge3d-cache")

import numpy as np

from seamforge3d.geometry.assets import PROCEDURAL_FAMILIES, create_procedural_assembly
from seamforge3d.visualization import visualize_assembly


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a dimension-randomized primitive workpiece gallery")
    parser.add_argument("--output", default="assets/procedural", help="OBJ/NPY/PNG output directory")
    parser.add_argument("--variants", type=int, default=2, help="Random dimension variants per primitive family")
    parser.add_argument("--seed", type=int, default=20260901, help="Deterministic generator seed")
    args = parser.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for family in PROCEDURAL_FAMILIES:
        for variant in range(args.variants):
            assembly = create_procedural_assembly(rng, families=(family,))
            target = root / family / f"variant_{variant:02d}"
            target.mkdir(parents=True, exist_ok=True)
            for mesh_index, mesh in enumerate(assembly.meshes):
                mesh.write_obj(target / f"part_{mesh.instance_id:02d}_{mesh_index:02d}_{mesh.name}.obj")
            for seam in assembly.seams:
                np.save(target / f"seam_{seam.trajectory_id:02d}.npy", seam.points)
            visualize_assembly(assembly, target / "preview.png")


if __name__ == "__main__":
    main()
