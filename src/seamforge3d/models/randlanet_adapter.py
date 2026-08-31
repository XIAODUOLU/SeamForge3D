from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from .backbone import PointBackbone


def _knn(support: np.ndarray, query: np.ndarray, neighbors: int) -> np.ndarray:
    actual = min(neighbors, len(support))
    index = cKDTree(support).query(query, k=actual)[1]
    if actual == 1:
        index = index[:, None]
    if actual < neighbors:
        index = np.concatenate((index, np.repeat(index[:, -1:], neighbors - actual, axis=1)), axis=1)
    return index.astype(np.int64)


class RandLANetBackbone(PointBackbone):
    """Adapter around the unmodified Open3D-ML RandLA-Net implementation."""

    deployment_family = "open3d-cpu"

    def __init__(
        self, feature_channels: int = 64, num_neighbors: int = 16, num_layers: int = 4,
        subsampling_ratio: list[int] | tuple[int, ...] = (4, 4, 4, 4),
        encoder_channels: list[int] | tuple[int, ...] = (16, 64, 128, 256), checkpoint: str | None = None,
        use_batch_stats_in_eval: bool = True, inference_votes: int = 4,
    ):
        super().__init__()
        try:
            from open3d.ml.torch.models import RandLANet
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "Open3D-ML RandLA-Net is unavailable or ABI-incompatible. "
                "Use './deploy.sh --profile intel' (Open3D 0.19 + PyTorch 2.2)."
            ) from exc
        self.model = RandLANet(
            num_neighbors=num_neighbors, num_layers=num_layers, num_points=1,
            num_classes=feature_channels, ignored_label_inds=[], subsampling_ratio=list(subsampling_ratio),
            in_channels=6, dim_features=8, dim_output=list(encoder_channels), grid_size=1.0,
        )
        self.num_neighbors = int(num_neighbors)
        self.subsampling_ratio = tuple(map(int, subsampling_ratio))
        self.output_channels = int(feature_channels)
        self.use_batch_stats_in_eval = bool(use_batch_stats_in_eval)
        self.inference_votes = max(1, int(inference_votes))
        if checkpoint:
            state = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
            state = state.get("model", state.get("state_dict", state))
            self.model.load_state_dict(state, strict=True)

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode and self.use_batch_stats_in_eval:
            # RandLA-Net is normally trained with many iterations and momentum
            # 0.01. For single-scene industrial inference, per-scan statistics
            # avoid stale running-state drift while all Dropout layers stay off.
            for module in self.model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.train()
        return self

    def _hierarchy(self, coord: np.ndarray) -> dict[str, list[torch.Tensor]]:
        coords, neighbors, pools, interpolation = [], [], [], []
        current = coord.astype(np.float32)
        for ratio in self.subsampling_ratio:
            neighbor = _knn(current, current, self.num_neighbors)
            sub_count = max(1, len(current) // ratio)
            sub = current[:sub_count]
            coords.append(torch.from_numpy(current[None]))
            neighbors.append(torch.from_numpy(neighbor[None]))
            pools.append(torch.from_numpy(neighbor[:sub_count][None]))
            interpolation.append(torch.from_numpy(_knn(sub, current, 1)[None]))
            current = sub
        return {"coords": coords, "neighbor_indices": neighbors, "sub_idx": pools, "interp_idx": interpolation}

    def forward(self, batch_data: dict[str, torch.Tensor], voxel_size: float) -> torch.Tensor:
        del voxel_size
        outputs = torch.empty(
            (len(batch_data["coord"]), self.output_channels),
            device=batch_data["coord"].device, dtype=batch_data["coord"].dtype,
        )
        self.model.device = batch_data["coord"].device
        for batch_id in torch.unique(batch_data["batch"]):
            mask = batch_data["batch"] == batch_id
            original_index = torch.nonzero(mask, as_tuple=False).flatten()
            count = len(original_index)
            votes = 1 if self.training else self.inference_votes
            accumulated = torch.zeros((count, self.output_channels), device=outputs.device, dtype=outputs.dtype)
            for vote in range(votes):
                if self.training:
                    permutation = torch.randperm(count, device=original_index.device)
                else:
                    # Official RandLA-Net inference aggregates repeated random
                    # samples. Fixed seeds make the same coverage reproducible.
                    order = np.random.default_rng(vote).permutation(count).astype(np.int64)
                    permutation = torch.from_numpy(order).to(original_index.device)
                selected = original_index[permutation]
                coord = batch_data["coord"][selected]
                coord = coord - coord.mean(dim=0, keepdim=True)
                features = torch.cat((coord, batch_data["normal"][selected]), dim=1)
                hierarchy = self._hierarchy(coord.detach().cpu().numpy())
                inputs = {key: [item.to(coord.device) for item in value] for key, value in hierarchy.items()}
                inputs["features"] = features.unsqueeze(0)
                feature = self.model(inputs).squeeze(0)
                if feature.shape != (count, self.output_channels):
                    raise RuntimeError(f"Unexpected RandLA-Net output shape {tuple(feature.shape)}")
                accumulated[permutation] += feature
            outputs[original_index] = accumulated / votes
        return outputs
