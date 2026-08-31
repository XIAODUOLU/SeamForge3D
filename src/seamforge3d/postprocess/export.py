from __future__ import annotations

import csv
import json
from pathlib import Path

from .reconstruct import Trajectory


def export_trajectories(output_dir: str | Path, trajectories: list[Trajectory]) -> None:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "trajectories.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trajectory_id", "point_index", "x", "y", "z", "confidence", "topology", "graph_id"])
        for trajectory in trajectories:
            for index, point in enumerate(trajectory.points):
                writer.writerow([trajectory.trajectory_id, index, *map(float, point), trajectory.confidence, trajectory.topology, trajectory.graph_id])
    metadata = [{
        "trajectory_id": t.trajectory_id, "confidence": t.confidence, "topology": t.topology,
        "part_pair": t.part_pair, "point_count": len(t.points), "graph_id": t.graph_id,
        "junction_start_id": t.junction_start_id, "junction_end_id": t.junction_end_id,
    } for t in trajectories]
    (output_dir / "trajectories.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

