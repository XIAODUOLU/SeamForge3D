#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/seamforge3d-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/seamforge3d-cache")
os.environ.setdefault("OPENVINO_TELEMETRY_DISABLED", "1")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from seamforge3d.config import load_config
from seamforge3d.data import SeamDataset, collate_seam_samples
from seamforge3d.losses import SeamFieldLoss
from seamforge3d.models import SeamFieldNet
from seamforge3d.runtime import move_batch, resolve_device, save_checkpoint, seed_everything


def epoch_pass(model, loader, criterion, device, optimizer=None, scaler=None, grad_clip=1.0, amp=True):
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    for batch in tqdm(loader, leave=False, desc="train" if training else "val"):
        batch = move_batch(batch, device)
        context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp and device.type == "cuda" else nullcontext()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), context:
            prediction = model(batch)
            loss, components = criterion(prediction, batch)
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        for key, value in components.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
    return {key: value / max(len(loader), 1) for key, value in totals.items()}


@torch.no_grad()
def calibrate_batch_norm(model, loader, device, batches: int) -> None:
    """Re-estimate official backbone BatchNorm statistics before deployment."""
    batch_norms = [module for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    if not batch_norms or batches <= 0:
        return
    momenta = [module.momentum for module in batch_norms]
    for module in batch_norms:
        module.reset_running_stats()
        module.momentum = None  # exact cumulative average over calibration batches
        module.train()
    model.train()
    # Calibration must not inject dropout noise into non-BN layers.
    for module in model.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            module.eval()
    iterator = iter(loader)
    for _ in tqdm(range(batches), leave=False, desc="calibrate BN"):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        model(move_batch(batch, device))
    for module, momentum in zip(batch_norms, momenta):
        module.momentum = momentum
    model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train full Utonia SeamFieldNet")
    parser.add_argument("--config", default="configs/overfit_two_meshes.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", help="Override the Utonia initialization checkpoint")
    parser.add_argument("--backbone", choices=("randlanet", "utonia"), help="Override model.backbone.name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-scenes", type=int, help="Limit the training split, useful for an overfit audit")
    parser.add_argument("--train-start-index", type=int, help="First offline scene used by an overfit audit")
    parser.add_argument("--val-scenes", type=int, help="Limit the validation split")
    parser.add_argument("--output-dir", help="Override train.output_dir")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.backbone:
        cfg["model"]["backbone"]["name"] = args.backbone
    if args.checkpoint:
        cfg["model"]["backbone"]["utonia"]["checkpoint"] = args.checkpoint
        cfg["model"]["backbone"]["utonia"]["pretrained"] = None
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.train_scenes:
        cfg["data"]["train_scenes"] = args.train_scenes
    if args.train_start_index is not None:
        cfg["data"]["train_start_index"] = args.train_start_index
    if args.val_scenes:
        cfg["data"]["val_scenes"] = args.val_scenes
    if args.output_dir:
        cfg["train"]["output_dir"] = args.output_dir
    device = resolve_device(args.device)
    if device.type == "cpu" and cfg["model"]["backbone"]["name"] == "utonia":
        cfg["model"]["backbone"]["utonia"]["use_flash_attention"] = False
    seed_everything(int(cfg["seed"]))
    output = Path(cfg["train"]["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    train_set = SeamDataset(cfg["data"], "train", int(cfg["seed"]))
    val_set = SeamDataset(cfg["data"], "val", int(cfg["seed"]) + 10_000_000)
    loader_args = dict(batch_size=int(cfg["train"]["batch_size"]), num_workers=int(cfg["train"]["workers"]), collate_fn=collate_seam_samples, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    model = SeamFieldNet(cfg["model"], cfg["data"]["voxel_size"]).to(device)
    criterion = SeamFieldLoss(cfg["loss"], cfg["data"]["supervision_radius"])
    backbone_parameters = list(model.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    task_parameters = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    backbone_lr = (
        float(cfg["train"]["backbone_learning_rate"])
        if cfg["model"]["backbone"]["name"] == "utonia"
        else float(cfg["train"]["learning_rate"])
    )
    optimizer = torch.optim.AdamW([
        {"params": backbone_parameters, "lr": backbone_lr},
        {"params": task_parameters, "lr": float(cfg["train"]["learning_rate"])},
    ], weight_decay=float(cfg["train"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["train"]["epochs"]), eta_min=1e-7)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["amp"] and device.type == "cuda"))
    start_epoch, best = 0, float("inf")
    if cfg["train"].get("resume"):
        state = torch.load(cfg["train"]["resume"], map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
    log_path = output / "metrics.jsonl"
    for epoch in range(start_epoch, int(cfg["train"]["epochs"])):
        train_metrics = epoch_pass(model, train_loader, criterion, device, optimizer, scaler, float(cfg["train"]["grad_clip"]), bool(cfg["train"]["amp"]))
        val_metrics = epoch_pass(model, val_loader, criterion, device, amp=bool(cfg["train"]["amp"]))
        scheduler.step()
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": [group["lr"] for group in optimizer.param_groups]}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record))
        save_checkpoint(output / "last.pt", model, optimizer, scheduler, scaler, epoch, cfg, record)
        if val_metrics["total"] < best:
            best = val_metrics["total"]
            save_checkpoint(output / "best.pt", model, optimizer, scheduler, scaler, epoch, cfg, record)
    calibration_batches = int(cfg["train"].get("bn_calibration_batches", 0))
    if calibration_batches:
        calibrate_batch_norm(model, train_loader, device, calibration_batches)
        calibrated_train = epoch_pass(model, train_loader, criterion, device, amp=bool(cfg["train"]["amp"]))
        calibrated_val = epoch_pass(model, val_loader, criterion, device, amp=bool(cfg["train"]["amp"]))
        calibrated_record = {
            "epoch": int(cfg["train"]["epochs"]) - 1,
            "bn_calibration_batches": calibration_batches,
            "train": calibrated_train,
            "val": calibrated_val,
        }
        (output / "calibrated_metrics.json").write_text(json.dumps(calibrated_record, indent=2), encoding="utf-8")
        save_checkpoint(output / "calibrated.pt", model, optimizer, scheduler, scaler, int(cfg["train"]["epochs"]) - 1, cfg, calibrated_record)
        print(json.dumps({"calibrated": calibrated_record}))


if __name__ == "__main__":
    main()
