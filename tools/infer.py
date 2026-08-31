#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/seamforge3d-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/seamforge3d-cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from seamforge3d.config import load_config
from seamforge3d.data import collate_seam_samples
from seamforge3d.metrics import seam_field_metrics, trajectory_metrics
from seamforge3d.models import SeamFieldNet
from seamforge3d.postprocess.dense_refine import refine_trajectory_on_dense_cloud
from seamforge3d.postprocess.export import export_trajectories
from seamforge3d.postprocess.reconstruct import ReconstructionConfig, SeamGraphReconstructor
from seamforge3d.runtime import move_batch, resolve_device


def plot_prediction(sample, probability, trajectories, output):
    coord = sample["coord"]
    figure = plt.figure(figsize=(12, 9)); axis = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(*coord.T, c=probability, cmap="turbo", s=2, vmin=0, vmax=1, alpha=0.8)
    for trajectory in trajectories:
        axis.plot(*trajectory.points.T, linewidth=3, label=f"pred {trajectory.trajectory_id} ({trajectory.topology})")
    offsets = sample["seam_offsets"]
    for index in range(len(offsets) - 1):
        seam = sample["seam_points"][offsets[index]:offsets[index + 1]]
        axis.plot(*seam.T, "k--", linewidth=1.5, label="GT" if index == 0 else None)
    figure.colorbar(scatter, ax=axis, label="Predicted seam probability")
    axis.legend(); figure.tight_layout(); figure.savefig(output, dpi=180); plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Infer, reconstruct, refine, and visualize weld trajectories")
    parser.add_argument("--config", default="configs/overfit_two_meshes.yaml")
    parser.add_argument("--model", required=True, help="Trained SeamFieldNet checkpoint")
    parser.add_argument("--input", required=True, help="Generated NPZ scene")
    parser.add_argument("--output", default="outputs/inference")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-dense-refine", action="store_true")
    parser.add_argument("--probability-threshold", type=float, help="Override the calibrated seam-field threshold")
    parser.add_argument("--inference-votes", type=int, help="Override RandLA-Net deterministic voting passes")
    args = parser.parse_args()
    cfg = load_config(args.config); device = resolve_device(args.device)
    if device.type == "cpu" and cfg["model"]["backbone"]["name"] == "utonia":
        cfg["model"]["backbone"]["utonia"]["use_flash_attention"] = False
    state = torch.load(args.model, map_location="cpu", weights_only=False)
    cfg = state.get("config", cfg)
    if args.probability_threshold is not None:
        cfg["postprocess"]["seam_probability_threshold"] = args.probability_threshold
    if args.inference_votes is not None:
        cfg["model"]["backbone"]["randlanet"]["inference_votes"] = args.inference_votes
    if device.type == "cpu" and cfg["model"]["backbone"]["name"] == "utonia":
        cfg["model"]["backbone"]["utonia"]["use_flash_attention"] = False
    model = SeamFieldNet(cfg["model"], cfg["data"]["voxel_size"]).to(device)
    model.load_state_dict(state["model"]); model.eval()
    with np.load(args.input, allow_pickle=False) as archive:
        sample = {key: archive[key] for key in archive.files}
    batch = move_batch(collate_seam_samples([sample]), device)
    with torch.inference_mode():
        output = model(batch)
    prediction = {key: value.float().cpu().numpy() for key, value in output.items()}
    pp = cfg["postprocess"]
    reconstructor = SeamGraphReconstructor(ReconstructionConfig(
        probability_threshold=pp["seam_probability_threshold"], merge_voxel=pp["merge_voxel"],
        graph_radius=pp["graph_radius"], tangent_angle_deg=pp["tangent_angle_deg"],
        min_component_length=pp["min_component_length"], bridge_gap=pp["bridge_gap"],
        sample_spacing=pp["spline_sample_spacing"],
    ))
    trajectories = reconstructor(sample["coord"], prediction)
    if not args.no_dense_refine:
        for trajectory in trajectories:
            trajectory.points, refine_confidence = refine_trajectory_on_dense_cloud(
                trajectory.points, sample["raw_coord"], sample["coord"], prediction["part_embed"]
            )
            if np.any(refine_confidence):
                trajectory.confidence *= float(0.5 + 0.5 * refine_confidence.mean())
    output_dir = Path(args.output); export_trajectories(output_dir, trajectories)
    probability = torch.sigmoid(output["seam_logit"]).cpu().numpy()
    plot_prediction(sample, probability, trajectories, output_dir / "prediction.png")
    gt = [sample["seam_points"][sample["seam_offsets"][i]:sample["seam_offsets"][i + 1]] for i in range(len(sample["seam_offsets"]) - 1)]
    metrics = {**seam_field_metrics(prediction, sample), **trajectory_metrics([t.points for t in trajectories], gt)}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
