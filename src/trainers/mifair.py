from __future__ import annotations

import torch


class MIFairOAELossComputer:
    """Count-based minibatch MIFair regularizer for Overall Accuracy Equality."""

    def __init__(
        self,
        n_groups: int,
        eta: float,
        strength: float = 1.0,
        eps: float = 1e-12,
    ):
        if n_groups <= 0:
            raise ValueError("MIFair requires at least one observed group.")
        if eta < 0:
            raise ValueError("mifair_eta must be non-negative.")
        if strength < 0 or strength > 1:
            raise ValueError("mifair_strength must be in [0, 1].")
        if eps <= 0:
            raise ValueError("mifair_eps must be strictly positive.")

        self.n_groups = int(n_groups)
        self.eta = float(eta)
        self.strength = float(strength)
        self.eps = float(eps)

    @property
    def effective_eta(self) -> float:
        return self.eta * self.strength

    def compute_mi_regularizer(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        group_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        probs = torch.softmax(pred, dim=1)
        target = target.long().view(-1)
        group_idx = group_idx.long().view(-1)

        if probs.shape[0] != target.shape[0] or target.shape[0] != group_idx.shape[0]:
            raise ValueError(
                "pred, target, and group_idx must share the same batch dimension."
            )

        p_correct = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        batch_size = p_correct.shape[0]
        if batch_size == 0:
            raise ValueError("MIFair regularization received an empty batch.")

        device = pred.device
        group_map = (
            group_idx.unsqueeze(0)
            == torch.arange(self.n_groups, device=device).unsqueeze(1)
        ).float()
        group_count = group_map.sum(dim=1)
        observed_groups = group_count > 0

        p_a = group_count / float(batch_size)
        p_b1 = p_correct.mean()
        p_b = torch.stack([p_b1, 1.0 - p_b1], dim=0).clamp_min(self.eps)

        joint_b1 = (group_map @ p_correct) / float(batch_size)
        joint_b0 = (group_map @ (1.0 - p_correct)) / float(batch_size)
        joint = torch.stack([joint_b1, joint_b0], dim=1)

        safe_joint = joint.clamp_min(self.eps)
        safe_p_a = p_a.clamp_min(self.eps).unsqueeze(1)
        mi_terms = safe_joint * (
            torch.log(safe_joint)
            - torch.log(safe_p_a)
            - torch.log(p_b.unsqueeze(0))
        )
        mi_regularizer = (mi_terms * observed_groups.unsqueeze(1)).sum()
        stats = {
            "mi_regularizer": float(mi_regularizer.detach().item()),
            "effective_eta": self.effective_eta,
            "observed_group_count": int(observed_groups.sum().item()),
            "soft_accuracy": float(p_b1.detach().item()),
        }
        return mi_regularizer, stats
