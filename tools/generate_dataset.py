#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/seamforge3d-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/seamforge3d-cache")

import numpy as np
from tqdm import tqdm

from seamforge3d.config import load_config
from seamforge3d.data.dataset import generate_scene, save_scene
from seamforge3d.geometry.assets import export_seed_assets
from seamforge3d.visualization import export_label_ply, visualize_assembly, visualize_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate raycast two-mesh weld seam training scenes")
    parser.add_argument("--config", default="configs/overfit_two_meshes.yaml")
    parser.add_argument("--visualizations", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    asset_dir = Path("assets/seed")
    assembly = export_seed_assets(asset_dir)
    root.mkdir(parents=True, exist_ok=True)
    visualize_assembly(assembly, root / "seed_meshes_and_seams.png")

    manifest = {"generator": "two_mesh_t_joint_raycast_v1", "unit": "meter", "config": data_cfg, "splits": {}}
    for split, count, offset in (("train", int(data_cfg["train_scenes"]), 0), ("val", int(data_cfg["val_scenes"]), 10_000_000)):
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest["splits"][split] = []
        for index in tqdm(range(count), desc=f"generate {split}"):
            path = split_dir / f"scene_{index:06d}.npz"
            seed = int(cfg["seed"]) + offset + index
            if args.overwrite or not path.exists():
                sample = generate_scene(data_cfg, seed)
                save_scene(path, sample)
            else:
                with np.load(path, allow_pickle=False) as archive:
                    sample = {key: archive[key] for key in archive.files}
            manifest["splits"][split].append({"file": str(path), "seed": seed, "points": int(len(sample["coord"]))})
            if index < args.visualizations:
                visualize_labels(sample, root / f"{split}_{index:02d}_labels.png")
                export_label_ply(sample, root / f"{split}_{index:02d}_labels.ply")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
