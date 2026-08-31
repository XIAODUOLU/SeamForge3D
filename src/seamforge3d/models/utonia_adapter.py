from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .backbone import PointBackbone


class UtoniaBackbone(PointBackbone):
    """Official Utonia encoder initialized from its checkpoint, with full PTv3 decoder.

    Utonia's published checkpoint is encoder-only. For dense seam fields we retain
    every pretrained encoder parameter and instantiate the official four-stage
    PointTransformerV3 decoder. No feature interpolation surrogate is used.
    """

    deployment_family = "cuda-spconv"

    def __init__(
        self,
        pretrained: str | None,
        checkpoint: str | None,
        use_flash_attention: bool,
        freeze_encoder: bool,
        decoder_channels: list[int],
    ) -> None:
        super().__init__()
        try:
            from utonia.model import PointTransformerV3, load
        except ImportError as exc:
            raise RuntimeError("Official Utonia is not installed. Run ./deploy.sh on the CUDA training host.") from exc

        source = checkpoint or "utonia"
        if checkpoint:
            if not Path(checkpoint).is_file():
                raise FileNotFoundError(checkpoint)
            checkpoint_data = load(checkpoint, ckpt_only=True)
        elif pretrained:
            checkpoint_data = load("utonia", repo_id=pretrained, ckpt_only=True)
        else:
            raise ValueError("Utonia needs either model.checkpoint or model.pretrained")

        model_config: dict[str, Any] = dict(checkpoint_data["config"])
        model_config.update(
            enc_mode=False,
            traceable=False,
            freeze_encoder=freeze_encoder,
            enable_flash=bool(use_flash_attention),
            dec_channels=tuple(decoder_channels),
        )
        if not use_flash_attention:
            model_config.update(upcast_attention=True, upcast_softmax=True)
        self.model = PointTransformerV3(**model_config)
        incompatible = self.model.load_state_dict(checkpoint_data["state_dict"], strict=False)
        unexpected = list(incompatible.unexpected_keys)
        invalid_missing = [key for key in incompatible.missing_keys if not key.startswith("dec.")]
        if unexpected or invalid_missing:
            raise RuntimeError(f"Unsafe Utonia checkpoint mismatch: unexpected={unexpected}, missing={invalid_missing}")
        self.in_channels = int(model_config["in_channels"])
        self.output_channels = int(decoder_channels[0])
        self.checkpoint_source = source

    @staticmethod
    def _center_per_batch(coord: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        centered = coord.clone()
        for batch_id in torch.unique(batch):
            mask = batch == batch_id
            centered[mask] -= coord[mask].mean(dim=0, keepdim=True)
        return centered

    def _features(self, coord: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        if self.in_channels == 9:
            color = torch.zeros_like(coord)
            return torch.cat((coord, color, normal), dim=1)
        if self.in_channels == 6:
            return torch.cat((coord, normal), dim=1)
        raise RuntimeError(f"Unsupported official Utonia input width: {self.in_channels}")

    def forward(self, batch_data: dict[str, torch.Tensor], voxel_size: float) -> torch.Tensor:
        coord = self._center_per_batch(batch_data["coord"], batch_data["batch"])
        grid_coord = torch.empty_like(coord, dtype=torch.int32)
        for batch_id in torch.unique(batch_data["batch"]):
            mask = batch_data["batch"] == batch_id
            local = coord[mask]
            grid_coord[mask] = torch.floor((local - local.min(dim=0).values) / voxel_size).to(torch.int32)
        point = self.model({
            "coord": coord,
            "grid_coord": grid_coord,
            "feat": self._features(coord, batch_data["normal"]),
            "batch": batch_data["batch"].to(torch.long),
            "offset": batch_data["offset"].to(torch.long),
        })
        if len(point.feat) != len(coord):
            raise RuntimeError(f"Dense PTv3 decoder returned {len(point.feat)} points for {len(coord)} inputs")
        return point.feat
