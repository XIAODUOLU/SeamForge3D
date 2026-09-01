from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..geometry.assets import PROCEDURAL_FAMILIES, create_procedural_assembly, create_seed_assembly, random_rigid_transform
from ..geometry.scanner import ScanConfig, scan_assembly
from ..geometry.visibility import filter_visible_seams
from ..labels import build_seam_labels, encode_seams


def scan_config_from_dict(config: dict[str, Any]) -> ScanConfig:
    return ScanConfig(
        num_views=tuple(config["num_views"]), width=int(config["image_width"]), height=int(config["image_height"]),
        fov_deg=float(config.get("fov_deg", 60.0)),
        camera_distance=tuple(config.get("camera_distance", (0.25, 0.42))),
        elevation_deg=tuple(config.get("elevation_deg", (18.0, 72.0))),
        voxel_size=float(config["voxel_size"]), xyz_noise=tuple(config["xyz_noise"]),
        normal_noise_deg=tuple(config["normal_noise_deg"]),
        grazing_dropout_strength=float(config["grazing_dropout_strength"]),
        edge_noise=float(config["edge_noise"]),
        flying_point_probability=float(config["flying_point_probability"]),
        pose_translation_noise=float(config["pose_translation_noise"]),
        pose_rotation_noise_deg=float(config["pose_rotation_noise_deg"]),
        point_dropout=tuple(config["point_dropout"]),
        patch_dropout_probability=float(config["patch_dropout_probability"]),
        view_mode=str(config.get("view_mode", "multi")),
        single_view_probability=float(config.get("single_view_probability", 0.0)),
        azimuth_deg=tuple(config.get("azimuth_deg", (0.0, 360.0))),
    )


def generate_scene(config: dict[str, Any], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    generator = str(config.get("assembly_generator", "seed_t_joint"))
    if generator == "seed_t_joint":
        assembly = create_seed_assembly()
    elif generator == "procedural":
        families = tuple(config.get("assembly_families", PROCEDURAL_FAMILIES))
        selected_families = (families[seed % len(families)],) if bool(config.get("balanced_family_sampling", False)) else families
        assembly = create_procedural_assembly(rng, families=selected_families, seam_samples=int(config.get("seam_samples", 321)))
    else:
        raise ValueError(f"Unknown assembly_generator: {generator}")
    assembly = assembly.transformed(random_rigid_transform(
        rng, max_translation=float(config.get("assembly_max_translation", 0.04)),
        rotation_mode=str(config.get("assembly_rotation_mode", "full")),
    ))
    sample = scan_assembly(assembly, scan_config_from_dict(config), rng)
    full_seams = assembly.seams
    if bool(config.get("filter_invisible_seams", False)):
        supervised_seams, visible_fraction = filter_visible_seams(
            full_seams, sample["raw_coord"], sample["raw_instance_id"],
            support_radius=float(config.get("visibility_support_radius", 0.007)),
            min_visible_length=float(config.get("min_visible_seam_length", 0.012)),
            max_gap_points=int(config.get("visibility_max_gap_points", 4)),
        )
    else:
        supervised_seams = full_seams
        visible_fraction = np.ones(len(full_seams), dtype=np.float32)
    labels = build_seam_labels(
        sample["coord"], supervised_seams, float(config["seam_sigma"]), float(config["supervision_radius"])
    )
    sample.update({
        "seam_distance": labels.distance,
        "seam_heat": labels.heat,
        "seam_offset": labels.offset,
        "seam_tangent": labels.tangent,
        "nearest_trajectory": labels.nearest_trajectory,
        "endpoint_heat": labels.endpoint_heat,
        "junction_heat": labels.junction_heat,
        "assembly_transform": assembly.transform.astype(np.float32),
        "scene_seed": np.array(seed, dtype=np.int64),
        "assembly_family": np.array(assembly.family),
        "full_seam_visible_fraction": visible_fraction,
    })
    sample.update(encode_seams(supervised_seams))
    sample.update(encode_seams(full_seams, prefix="full_seam_"))
    return sample


def save_scene(path: str | Path, sample: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **sample)


class SeamDataset:
    def __init__(self, config: dict[str, Any], split: str, seed: int, dynamic: bool | None = None):
        self.config = config
        self.split = split
        self.seed = int(seed)
        default_dynamic = config["dynamic"] if split == "train" else config.get(f"dynamic_{split}", False)
        self.dynamic = bool(default_dynamic if dynamic is None else dynamic)
        self.count = int(config[f"{split}_scenes"])
        self.root = Path(config["root"]) / split
        all_files = sorted(self.root.glob("*.npz"))
        start = int(config.get(f"{split}_start_index", 0))
        self.files = all_files[start:start + self.count]
        self.epoch = 0
        if not self.dynamic and len(self.files) < self.count:
            raise FileNotFoundError(
                f"Expected {self.count} scenes from index {start} under {self.root}, "
                f"found {len(self.files)}; run tools/generate_dataset.py"
            )

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.dynamic:
            # Large coprime epoch/index spacing prevents repeated augmentation sequences.
            sample = generate_scene(self.config, self.seed + self.epoch * 1_000_003 + index * 104729)
        else:
            with np.load(self.files[index], allow_pickle=False) as archive:
                sample = {key: archive[key] for key in archive.files}
        rng = np.random.default_rng(self.seed + index * 1009)
        if self.split == "train" and rng.random() < float(self.config["normal_sign_flip_probability"]):
            sample["normal"] = -sample["normal"]
        return sample

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def collate_seam_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    lengths = np.array([len(sample["coord"]) for sample in samples], dtype=np.int64)
    batch = np.repeat(np.arange(len(samples)), lengths)
    result: dict[str, Any] = {
        "coord": torch.from_numpy(np.concatenate([s["coord"] for s in samples])).float(),
        "normal": torch.from_numpy(np.concatenate([s["normal"] for s in samples])).float(),
        "instance_id": torch.from_numpy(np.concatenate([s["instance_id"] for s in samples])).long(),
        "seam_heat": torch.from_numpy(np.concatenate([s["seam_heat"] for s in samples])).float(),
        "seam_distance": torch.from_numpy(np.concatenate([s["seam_distance"] for s in samples])).float(),
        "seam_offset": torch.from_numpy(np.concatenate([s["seam_offset"] for s in samples])).float(),
        "seam_tangent": torch.from_numpy(np.concatenate([s["seam_tangent"] for s in samples])).float(),
        "endpoint_heat": torch.from_numpy(np.concatenate([s["endpoint_heat"] for s in samples])).float(),
        "junction_heat": torch.from_numpy(np.concatenate([s["junction_heat"] for s in samples])).float(),
        "batch": torch.from_numpy(batch).long(),
        "offset": torch.from_numpy(np.cumsum(lengths)).long(),
        "samples": samples,
    }
    return result
