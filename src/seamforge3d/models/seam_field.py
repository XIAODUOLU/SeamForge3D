from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import build_backbone


class DenseHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(in_channels), nn.Linear(in_channels, hidden_channels), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class SeamFieldNet(nn.Module):
    def __init__(self, config: dict, voxel_size: float):
        super().__init__()
        self.voxel_size = float(voxel_size)
        self.backbone = build_backbone(config["backbone"])
        width = self.backbone.output_channels
        hidden = int(config["head_hidden_channels"])
        self.seam_head = DenseHead(width, hidden, 1)
        self.offset_head = DenseHead(width, hidden, 3)
        self.tangent_head = DenseHead(width, hidden, 3)
        self.embedding_head = DenseHead(width, hidden, int(config["part_embedding_dim"]))
        self.endpoint_head = DenseHead(width, hidden, 1) if config.get("use_endpoint_head", False) else None
        self.junction_head = DenseHead(width, hidden, 1) if config.get("use_junction_head", False) else None

    def forward(self, batch_data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = self.backbone(batch_data, self.voxel_size)
        tangent = F.normalize(self.tangent_head(features), dim=-1, eps=1e-6)
        output = {
            "seam_logit": self.seam_head(features).squeeze(-1),
            "seam_offset": self.offset_head(features),
            "seam_tangent": tangent,
            "part_embed": self.embedding_head(features),
        }
        if self.endpoint_head is not None:
            output["endpoint_logit"] = self.endpoint_head(features).squeeze(-1)
        if self.junction_head is not None:
            output["junction_logit"] = self.junction_head(features).squeeze(-1)
        return output
