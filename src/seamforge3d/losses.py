from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def continuous_focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    """Class-balanced focal loss for a continuous Gaussian target field.

    Positive and negative masses are normalized independently. This prevents the
    all-background solution when a seam occupies only a few hundred of tens of
    thousands of scanned points, while retaining the full Gaussian supervision.
    """
    probability = torch.sigmoid(logits)
    eps = torch.finfo(probability.dtype).eps
    positive_mass = target
    # Down-weight ambiguous samples in the shoulder of the Gaussian for the
    # negative term, following the heatmap-focal treatment used by keypoint nets.
    negative_mass = (1 - target).pow(4)
    positive = -(1 - probability).pow(gamma) * torch.log(probability.clamp_min(eps)) * positive_mass
    negative = -probability.pow(gamma) * torch.log((1 - probability).clamp_min(eps)) * negative_mass
    positive = positive.sum() / positive_mass.sum().clamp_min(1.0)
    negative = negative.sum() / negative_mass.sum().clamp_min(1.0)
    return alpha * positive + (1 - alpha) * negative


def discriminative_embedding_loss(
    embedding: torch.Tensor, instance_id: torch.Tensor, batch: torch.Tensor,
    delta_var: float, delta_dist: float, regularization: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pull_terms, push_terms, reg_terms = [], [], []
    for batch_id in torch.unique(batch):
        scene_mask = batch == batch_id
        scene_embedding = embedding[scene_mask]
        scene_instances = instance_id[scene_mask]
        labels = torch.unique(scene_instances[scene_instances >= 0])
        if len(labels) == 0:
            continue
        means = []
        for label in labels:
            vectors = scene_embedding[scene_instances == label]
            mean = vectors.mean(dim=0)
            means.append(mean)
            pull_terms.append(F.relu(torch.linalg.vector_norm(vectors - mean, dim=1) - delta_var).pow(2).mean())
        means = torch.stack(means)
        reg_terms.append(torch.linalg.vector_norm(means, dim=1).mean())
        if len(means) > 1:
            pair_distance = torch.cdist(means, means)
            upper = torch.triu(torch.ones_like(pair_distance, dtype=torch.bool), diagonal=1)
            push_terms.append(F.relu(2 * delta_dist - pair_distance[upper]).pow(2).mean())
    zero = embedding.sum() * 0.0
    pull = torch.stack(pull_terms).mean() if pull_terms else zero
    push = torch.stack(push_terms).mean() if push_terms else zero
    reg = torch.stack(reg_terms).mean() if reg_terms else zero
    return pull + push + regularization * reg, {"embedding_pull": pull, "embedding_push": push, "embedding_reg": reg}


class SeamFieldLoss(nn.Module):
    def __init__(self, config: dict, supervision_radius: float):
        super().__init__()
        self.config = config
        self.supervision_radius = float(supervision_radius)

    def forward(self, prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        heat = continuous_focal_loss(
            prediction["seam_logit"], target["seam_heat"], float(self.config["focal_alpha"]), float(self.config["focal_gamma"])
        )
        active = target["seam_distance"] < self.supervision_radius
        if active.any():
            weights = target["seam_heat"][active].clamp_min(0.05)
            # Normalize metric offsets so millimetre-scale errors have useful
            # gradients instead of vanishing in Smooth-L1's quadratic region.
            offset_pointwise = F.smooth_l1_loss(
                prediction["seam_offset"][active] / self.supervision_radius,
                target["seam_offset"][active] / self.supervision_radius,
                reduction="none",
            ).mean(dim=1)
            offset = (offset_pointwise * weights).sum() / weights.sum()
            cosine = torch.sum(prediction["seam_tangent"][active] * target["seam_tangent"][active], dim=1).abs()
            tangent = ((1 - cosine) * weights).sum() / weights.sum()
        else:
            offset = prediction["seam_offset"].sum() * 0.0
            tangent = prediction["seam_tangent"].sum() * 0.0
        embedding, embedding_parts = discriminative_embedding_loss(
            prediction["part_embed"], target["instance_id"], target["batch"],
            float(self.config["embedding_delta_var"]), float(self.config["embedding_delta_dist"]),
        )
        components = {"heat": heat, "offset": offset, "tangent": tangent, "embedding": embedding, **embedding_parts}
        total = (
            float(self.config["heat"]) * heat + float(self.config["offset"]) * offset
            + float(self.config["tangent"]) * tangent + float(self.config["embedding"]) * embedding
        )
        if "endpoint_logit" in prediction:
            endpoint = continuous_focal_loss(prediction["endpoint_logit"], target["endpoint_heat"], 0.25, 2.0)
            components["endpoint"] = endpoint
            total = total + 0.2 * endpoint
        if "junction_logit" in prediction:
            junction = continuous_focal_loss(prediction["junction_logit"], target["junction_heat"], 0.25, 2.0)
            components["junction"] = junction
            total = total + 0.2 * junction
        components["total"] = total
        return total, components
