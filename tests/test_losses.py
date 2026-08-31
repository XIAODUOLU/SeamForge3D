import torch

from seamforge3d.losses import SeamFieldLoss


def test_full_multitask_loss_is_finite_and_differentiable():
    count = 32
    prediction = {
        "seam_logit": torch.randn(count, requires_grad=True),
        "seam_offset": torch.randn(count, 3, requires_grad=True) * 0.01,
        "seam_tangent": torch.nn.functional.normalize(torch.randn(count, 3, requires_grad=True), dim=-1),
        "part_embed": torch.randn(count, 8, requires_grad=True),
    }
    target = {
        "seam_heat": torch.rand(count), "seam_distance": torch.linspace(0, 0.02, count),
        "seam_offset": torch.zeros(count, 3), "seam_tangent": torch.nn.functional.normalize(torch.randn(count, 3), dim=-1),
        "instance_id": torch.tensor([0] * 16 + [1] * 16), "batch": torch.zeros(count, dtype=torch.long),
        "endpoint_heat": torch.zeros(count), "junction_heat": torch.zeros(count),
    }
    config = {"heat": 2.0, "offset": 1.0, "tangent": 0.2, "embedding": 0.5, "focal_alpha": 0.25, "focal_gamma": 2.0, "embedding_delta_var": 0.5, "embedding_delta_dist": 1.5}
    loss, components = SeamFieldLoss(config, 0.012)(prediction, target)
    assert torch.isfinite(loss)
    assert {"heat", "offset", "tangent", "embedding", "total"}.issubset(components)
    loss.backward()

