#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/seamforge3d-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/seamforge3d-cache")
os.environ.setdefault("OPENVINO_TELEMETRY_DISABLED", "1")

import numpy as np
import torch
from tqdm import tqdm

from seamforge3d.config import load_config
from seamforge3d.data import collate_seam_samples
from seamforge3d.metrics import seam_field_metrics, trajectory_metrics
from seamforge3d.models import SeamFieldNet
from seamforge3d.postprocess.dense_refine import refine_trajectory_on_dense_cloud
from seamforge3d.postprocess.reconstruct import ReconstructionConfig, SeamGraphReconstructor
from seamforge3d.runtime import move_batch, resolve_device
from seamforge3d.visualization import visualize_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one checkpoint on a complete val/test split")
    parser.add_argument("--config", required=True, help="Dataset/runtime YAML; its data split is authoritative")
    parser.add_argument("--model", required=True, help="Training checkpoint")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--inference-votes", type=int, help="RandLA-Net repeated deterministic sampling votes")
    parser.add_argument("--probability-threshold", type=float, help="Seam candidate threshold")
    parser.add_argument("--dense-refine", action="store_true", help="Enable raw-cloud two-surface refinement")
    parser.add_argument("--visualizations", type=int, default=4, help="Number of prediction PNGs to retain")
    args = parser.parse_args()

    runtime_cfg = load_config(args.config)
    device = resolve_device(args.device)
    state = torch.load(args.model, map_location="cpu", weights_only=False)
    checkpoint_cfg = state["config"]
    model_cfg = checkpoint_cfg["model"]
    if args.inference_votes is not None and model_cfg["backbone"]["name"] == "randlanet":
        model_cfg["backbone"]["randlanet"]["inference_votes"] = args.inference_votes
    if device.type == "cpu" and model_cfg["backbone"]["name"] == "utonia":
        model_cfg["backbone"]["utonia"]["use_flash_attention"] = False
    model = SeamFieldNet(model_cfg, float(runtime_cfg["data"]["voxel_size"])).to(device)
    model.load_state_dict(state["model"]); model.eval()

    pp = runtime_cfg["postprocess"]
    threshold = float(args.probability_threshold if args.probability_threshold is not None else pp["seam_probability_threshold"])
    reconstructor = SeamGraphReconstructor(ReconstructionConfig(
        probability_threshold=threshold, merge_voxel=float(pp["merge_voxel"]), graph_radius=float(pp["graph_radius"]),
        tangent_angle_deg=float(pp["tangent_angle_deg"]), min_component_length=float(pp["min_component_length"]),
        bridge_gap=float(pp["bridge_gap"]), sample_spacing=float(pp["spline_sample_spacing"]),
    ))
    count = int(runtime_cfg["data"][f"{args.split}_scenes"])
    start = int(runtime_cfg["data"].get(f"{args.split}_start_index", 0))
    files = sorted((Path(runtime_cfg["data"]["root"]) / args.split).glob("*.npz"))[start:start + count]
    if len(files) != count:
        raise FileNotFoundError(f"Expected {count} {args.split} scenes, found {len(files)}")
    output_dir = Path(args.output); output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for scene_index, path in enumerate(tqdm(files, desc=f"evaluate {args.split}")):
        with np.load(path, allow_pickle=False) as archive:
            sample = {key: archive[key] for key in archive.files}
        batch = move_batch(collate_seam_samples([sample]), device)
        with torch.inference_mode():
            tensor_output = model(batch)
        prediction = {key: value.float().cpu().numpy() for key, value in tensor_output.items()}
        trajectories = reconstructor(sample["coord"], prediction)
        if args.dense_refine:
            for trajectory in trajectories:
                trajectory.points, _ = refine_trajectory_on_dense_cloud(
                    trajectory.points, sample["raw_coord"], sample["coord"], prediction["part_embed"]
                )
        gt = [sample["seam_points"][sample["seam_offsets"][i]:sample["seam_offsets"][i + 1]] for i in range(len(sample["seam_offsets"]) - 1)]
        metrics = {**seam_field_metrics(prediction, sample), **trajectory_metrics([item.points for item in trajectories], gt)}
        record = {
            "scene": path.name, "family": str(sample.get("assembly_family", "unknown")),
            "views": int(sample.get("num_rendered_views", -1)), "gt_trajectories": len(gt),
            "predicted_trajectories": len(trajectories), **metrics,
        }
        records.append(record)
        if scene_index < args.visualizations:
            probability = torch.sigmoid(tensor_output["seam_logit"]).cpu().numpy()
            visualize_prediction(sample, probability, trajectories, output_dir / f"{path.stem}_prediction.png")
    with (output_dir / "per_scene.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=True) + "\n")
    metric_keys = sorted({key for record in records for key, value in record.items() if isinstance(value, (int, float)) and key not in ("views", "gt_trajectories", "predicted_trajectories")})
    summary = {"split": args.split, "scenes": len(records), "checkpoint": args.model, "threshold": threshold}
    for key in metric_keys:
        values = np.asarray([record[key] for record in records if key in record], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[f"{key}_mean"] = float(finite.mean()) if len(finite) else float("inf")
        summary[f"{key}_median"] = float(np.median(finite)) if len(finite) else float("inf")
        summary[f"{key}_finite_scenes"] = int(len(finite))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
