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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SeamForge3D JSONL training metrics")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.metrics).read_text(encoding="utf-8").splitlines() if line]
    epochs = [record["epoch"] for record in records]
    keys = ("total", "heat", "offset", "tangent", "embedding", "endpoint", "junction")
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for key in keys:
        values = [record["train"].get(key) for record in records]
        if all(value is not None for value in values):
            axes[0].plot(epochs, values, label=key)
    axes[0].set_yscale("log"); axes[0].set_ylabel("training loss (log)"); axes[0].grid(alpha=0.25); axes[0].legend(ncol=4)
    axes[1].plot(epochs, [record["train"]["total"] for record in records], label="train total")
    axes[1].plot(epochs, [record["val"]["total"] for record in records], label="held-out total")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("total loss"); axes[1].grid(alpha=0.25); axes[1].legend()
    figure.suptitle("SeamFieldNet overfit audit")
    figure.tight_layout()
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180); plt.close(figure)


if __name__ == "__main__":
    main()
