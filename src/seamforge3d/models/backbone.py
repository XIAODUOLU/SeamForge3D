from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class PointBackbone(nn.Module, ABC):
    """Stable contract for interchangeable dense point-cloud backbones."""

    output_channels: int
    deployment_family: str

    @abstractmethod
    def forward(self, batch_data: dict[str, Tensor], voxel_size: float) -> Tensor:
        """Return one feature vector for every input point, preserving input order."""


def build_backbone(config: dict) -> PointBackbone:
    name = config.get("name", "randlanet")
    if name == "utonia":
        from .utonia_adapter import UtoniaBackbone
        options = config["utonia"]
        return UtoniaBackbone(
            pretrained=options.get("pretrained"), checkpoint=options.get("checkpoint"),
            use_flash_attention=bool(options["use_flash_attention"]), freeze_encoder=bool(options["freeze_encoder"]),
            decoder_channels=list(options["decoder_channels"]),
        )
    if name == "randlanet":
        from .randlanet_adapter import RandLANetBackbone
        return RandLANetBackbone(**config["randlanet"])
    raise ValueError(f"Unknown backbone '{name}'. Available: randlanet, utonia")
