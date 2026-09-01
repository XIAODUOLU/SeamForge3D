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
    parser = argparse.ArgumentParser(description="Generate raycast procedural weld-seam scenes")
    parser.add_argument("--config", default="configs/overfit_two_meshes.yaml")
    parser.add_argument("--visualizations", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    asset_dir = Path("assets/seed")
    if data_cfg.get("assembly_generator", "seed_t_joint") == "seed_t_joint":
        assembly = export_seed_assets(asset_dir)
        root.mkdir(parents=True, exist_ok=True)
        visualize_assembly(assembly, root / "seed_meshes_and_seams.png")

    manifest = {"generator": "procedural_raycast_v2", "unit": "meter", "config": data_cfg, "splits": {}}
    split_specs = (
        ("train", int(data_cfg["train_scenes"]), 0),
        ("val", int(data_cfg["val_scenes"]), 10_000_000),
        ("test", int(data_cfg.get("test_scenes", 0)), 20_000_000),
    )
    for split, count, offset in split_specs:
        if count <= 0:
            continue
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
            manifest["splits"][split].append({
                "file": str(path), "seed": seed, "points": int(len(sample["coord"])),
                "family": str(sample["assembly_family"]),
                "rendered_views": int(sample["num_rendered_views"]),
                "full_seams": int(len(sample["full_seam_offsets"]) - 1),
                "visible_seams": int(len(sample["seam_offsets"]) - 1),
                "visible_fraction": sample["full_seam_visible_fraction"].tolist(),
            })
            if index < args.visualizations:
                visualize_labels(sample, root / f"{split}_{index:02d}_labels.png")
                export_label_ply(sample, root / f"{split}_{index:02d}_labels.ply")
    statistics = {}
    for split, entries in manifest["splits"].items():
        family_counts: dict[str, int] = {}
        for entry in entries:
            family_counts[entry["family"]] = family_counts.get(entry["family"], 0) + 1
        statistics[split] = {
            "scenes": len(entries), "family_counts": family_counts,
            "single_view_scenes": sum(entry["rendered_views"] == 1 for entry in entries),
            "empty_visible_truth_scenes": sum(entry["visible_seams"] == 0 for entry in entries),
            "mean_network_points": float(np.mean([entry["points"] for entry in entries])) if entries else 0.0,
            "full_trajectories": sum(entry["full_seams"] for entry in entries),
            "visible_trajectories": sum(entry["visible_seams"] for entry in entries),
        }
    manifest["statistics"] = statistics
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
