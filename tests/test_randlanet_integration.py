import pytest
import torch

pytest.importorskip("open3d")

from seamforge3d.models.randlanet_adapter import RandLANetBackbone


def test_official_randlanet_dense_forward_and_backward():
    torch.manual_seed(3)
    count = 512
    coord = torch.rand(count, 3)
    normal = torch.nn.functional.normalize(torch.randn(count, 3), dim=1)
    batch = {"coord": coord, "normal": normal, "batch": torch.zeros(count, dtype=torch.long)}
    model = RandLANetBackbone(
        feature_channels=16, num_neighbors=8, num_layers=3,
        subsampling_ratio=(4, 4, 4), encoder_channels=(8, 16, 32),
        inference_votes=2,
    )
    output = model(batch, voxel_size=0.003)
    assert output.shape == (count, 16)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
