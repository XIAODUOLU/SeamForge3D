from __future__ import annotations

import random
from pathlib import Path

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(requested: str):
    import torch
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device(requested)


def move_batch(batch: dict, device):
    import torch
    return {key: value.to(device, non_blocking=device.type == "cuda") if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def save_checkpoint(path: str | Path, model, optimizer, scheduler, scaler, epoch: int, config: dict, metrics: dict) -> None:
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "config": config, "metrics": metrics,
    }, path)

