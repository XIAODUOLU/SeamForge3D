from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .assets import Assembly


@dataclass(frozen=True)
class ScanConfig:
    num_views: tuple[int, int] = (8, 16)
    width: int = 320
    height: int = 240
    fov_deg: float = 60.0
    camera_distance: tuple[float, float] = (0.25, 0.42)
    elevation_deg: tuple[float, float] = (18.0, 72.0)
    voxel_size: float = 0.003
    xyz_noise: tuple[float, float] = (0.0002, 0.0010)
    normal_noise_deg: tuple[float, float] = (2.0, 8.0)
    grazing_dropout_strength: float = 0.60
    edge_noise: float = 0.0015
    flying_point_probability: float = 0.02
    pose_translation_noise: float = 0.001
    pose_rotation_noise_deg: float = 0.5
    point_dropout: tuple[float, float] = (0.0, 0.20)
    patch_dropout_probability: float = 0.35
    view_mode: str = "multi"
    single_view_probability: float = 0.0
    azimuth_deg: tuple[float, float] = (0.0, 360.0)


def _look_at(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-12
    right = np.cross(forward, up_hint)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, forward)
    rotation = np.stack((right, up, forward), axis=1)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation.T
    extrinsic[:3, 3] = -rotation.T @ eye
    return extrinsic.astype(np.float32)


def _perturb_extrinsic(extrinsic: np.ndarray, rng: np.random.Generator, translation: float, rotation_deg: float) -> np.ndarray:
    delta = np.eye(4, dtype=np.float32)
    angle = np.deg2rad(rng.normal(0.0, rotation_deg))
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-12
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    delta[:3, :3] = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
    delta[:3, 3] = rng.normal(0.0, translation, size=3)
    return delta @ extrinsic


def _voxel_reduce(coord: np.ndarray, normal: np.ndarray, instance: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = coord.min(axis=0)
    key = np.floor((coord - origin) / voxel).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_inverse=True, return_index=True)
    counts = np.bincount(inverse)
    out_coord = np.stack([np.bincount(inverse, weights=coord[:, d]) / counts for d in range(3)], axis=1)
    out_normal = np.stack([np.bincount(inverse, weights=normal[:, d]) / counts for d in range(3)], axis=1)
    out_normal /= np.linalg.norm(out_normal, axis=1, keepdims=True).clip(1e-8)
    out_instance = instance[first]
    return out_coord.astype(np.float32), out_normal.astype(np.float32), out_instance.astype(np.int32)


def scan_assembly(assembly: Assembly, config: ScanConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Raycast a multi-view point cloud and retain geometry-derived instance IDs."""
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required for physical scan simulation; run deploy.sh") from exc

    scene = o3d.t.geometry.RaycastingScene()
    geometry_to_instance: dict[int, int] = {}
    for mesh in assembly.meshes:
        legacy = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64)),
            o3d.utility.Vector3iVector(mesh.triangles.astype(np.int32)),
        )
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
        geometry_id = scene.add_triangles(tensor_mesh)
        geometry_to_instance[int(geometry_id)] = mesh.instance_id

    if config.view_mode not in ("single", "multi", "mixed"):
        raise ValueError(f"Unsupported view_mode: {config.view_mode}")
    single_view = config.view_mode == "single" or (config.view_mode == "mixed" and rng.random() < config.single_view_probability)
    n_views = 1 if single_view else int(rng.integers(config.num_views[0], config.num_views[1] + 1))
    coords: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    instances: list[np.ndarray] = []
    view_ids: list[np.ndarray] = []
    camera_origins: list[np.ndarray] = []
    camera_extrinsics: list[np.ndarray] = []
    center = np.mean(np.concatenate([m.vertices for m in assembly.meshes]), axis=0)

    intrinsic = np.array(
        [[0.5 * config.width / np.tan(np.deg2rad(config.fov_deg) / 2), 0, config.width / 2],
         [0, 0.5 * config.width / np.tan(np.deg2rad(config.fov_deg) / 2), config.height / 2],
         [0, 0, 1]], dtype=np.float32,
    )
    for view_id in range(n_views):
        azimuth = np.deg2rad(rng.uniform(*config.azimuth_deg))
        elevation = np.deg2rad(rng.uniform(*config.elevation_deg))
        distance = rng.uniform(*config.camera_distance)
        eye = center + distance * np.array(
            [np.cos(azimuth) * np.cos(elevation), np.sin(azimuth) * np.cos(elevation), np.sin(elevation)]
        )
        extrinsic = _look_at(eye, center, np.array([0.0, 0.0, 1.0]))
        extrinsic = _perturb_extrinsic(extrinsic, rng, config.pose_translation_noise, config.pose_rotation_noise_deg)
        camera_origins.append(eye.astype(np.float32))
        camera_extrinsics.append(extrinsic.astype(np.float32))
        rays = scene.create_rays_pinhole(
            o3d.core.Tensor(intrinsic), o3d.core.Tensor(extrinsic), config.width, config.height
        )
        hits = scene.cast_rays(rays)
        t_hit = hits["t_hit"].numpy().reshape(-1)
        rays_np = rays.numpy().reshape(-1, 6)
        valid = np.isfinite(t_hit)
        if not np.any(valid):
            continue
        xyz = rays_np[valid, :3] + rays_np[valid, 3:] * t_hit[valid, None]
        primitive_normal = hits["primitive_normals"].numpy().reshape(-1, 3)[valid]
        primitive_uv = hits["primitive_uvs"].numpy().reshape(-1, 2)[valid]
        geometry_id = hits["geometry_ids"].numpy().reshape(-1)[valid]
        instance = np.array([geometry_to_instance[int(g)] for g in geometry_id], dtype=np.int32)

        sigma = rng.uniform(*config.xyz_noise)
        ray_dir = rays_np[valid, 3:]
        xyz += ray_dir * rng.normal(0.0, sigma, size=(len(xyz), 1))
        barycentric_min = np.min(np.column_stack((primitive_uv, 1 - primitive_uv.sum(axis=1))), axis=1)
        edge = barycentric_min < 0.04
        xyz[edge] += ray_dir[edge] * rng.normal(0.0, config.edge_noise, size=(np.count_nonzero(edge), 1))
        flying = edge & (rng.random(len(xyz)) < config.flying_point_probability)
        xyz[flying] += ray_dir[flying] * rng.uniform(0.002, 0.010, size=(np.count_nonzero(flying), 1))
        normal_sigma = np.deg2rad(rng.uniform(*config.normal_noise_deg))
        primitive_normal += rng.normal(0.0, normal_sigma, size=primitive_normal.shape)
        primitive_normal /= np.linalg.norm(primitive_normal, axis=1, keepdims=True).clip(1e-8)
        incidence = np.abs(np.sum(ray_dir * primitive_normal, axis=1))
        grazing_drop = config.grazing_dropout_strength * (1 - incidence) ** 2
        keep = rng.random(len(xyz)) >= np.maximum(rng.uniform(*config.point_dropout), grazing_drop)
        coords.append(xyz[keep].astype(np.float32))
        normals.append(primitive_normal[keep].astype(np.float32))
        instances.append(instance[keep])
        view_ids.append(np.full(np.count_nonzero(keep), view_id, dtype=np.int16))

    if not coords:
        raise RuntimeError("All raycast views missed the assembly")
    coord = np.concatenate(coords)
    normal = np.concatenate(normals)
    instance = np.concatenate(instances)
    view_id = np.concatenate(view_ids)

    if rng.random() < config.patch_dropout_probability and len(coord) > 100:
        patch_center = coord[rng.integers(len(coord))]
        radius = rng.uniform(0.008, 0.025)
        keep = np.linalg.norm(coord - patch_center, axis=1) > radius
        coord, normal, instance, view_id = coord[keep], normal[keep], instance[keep], view_id[keep]

    raw_coord, raw_normal, raw_instance = coord.copy(), normal.copy(), instance.copy()
    coord, normal, instance = _voxel_reduce(coord, normal, instance, config.voxel_size)
    return {
        "coord": coord,
        "normal": normal,
        "instance_id": instance,
        "raw_coord": raw_coord.astype(np.float32),
        "raw_normal": raw_normal.astype(np.float32),
        "raw_instance_id": raw_instance.astype(np.int32),
        "raw_view_id": view_id.astype(np.int16),
        "camera_origins": np.stack(camera_origins).astype(np.float32),
        "camera_extrinsics": np.stack(camera_extrinsics).astype(np.float32),
        "num_rendered_views": np.array(n_views, dtype=np.int16),
    }
