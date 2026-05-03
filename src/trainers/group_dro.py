from __future__ import annotations

from typing import Optional

import torch


class GroupDROLossComputer:
    """Batch-wise Group DRO loss with exponentiated-gradient adversarial weights."""

    def __init__(
        self,
        group_counts: torch.Tensor,
        step_size: float = 0.01,
        adjustment: Optional[torch.Tensor] = None,
        normalize_loss: bool = False,
        strength: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        if group_counts.numel() == 0:
            raise ValueError("Group DRO requires at least one observed group.")

        self.step_size = float(step_size)
        self.normalize_loss = bool(normalize_loss)
        self.strength = float(strength)
        self.group_counts = group_counts.detach().clone().float()
        self.adjustment = (
            adjustment.detach().clone().float()
            if adjustment is not None
            else torch.zeros_like(self.group_counts)
        )
        if self.adjustment.shape != self.group_counts.shape:
            raise ValueError(
                "Group DRO adjustment must have the same shape as group_counts."
            )

        if device is not None:
            self.to(device)
        else:
            self.adv_probs = torch.ones_like(self.group_counts) / self.group_counts.numel()

    @property
    def n_groups(self) -> int:
        return int(self.group_counts.numel())

    def to(self, device: torch.device | str) -> "GroupDROLossComputer":
        self.group_counts = self.group_counts.to(device)
        self.adjustment = self.adjustment.to(device)
        if hasattr(self, "adv_probs"):
            self.adv_probs = self.adv_probs.to(device)
        else:
            self.adv_probs = torch.ones_like(self.group_counts) / self.group_counts.numel()
        return self

    def state_dict(self) -> dict:
        return {
            "adv_probs": self.adv_probs.detach().cpu(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if "adv_probs" in state_dict:
            self.adv_probs = state_dict["adv_probs"].to(self.group_counts.device)

    def compute_group_average(
        self,
        values: torch.Tensor,
        group_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group_sum, group_count = self.compute_group_sum(values, group_idx, self.n_groups)
        denom = group_count.clamp_min(1.0)
        return group_sum / denom, group_count

    @staticmethod
    def compute_group_sum(
        values: torch.Tensor,
        group_idx: torch.Tensor,
        n_groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 1:
            values = values.view(-1)

        group_idx = group_idx.long().view(-1)
        if values.shape[0] != group_idx.shape[0]:
            raise ValueError("values and group_idx must have the same batch dimension.")

        device = values.device
        group_map = (
            group_idx.unsqueeze(0)
            == torch.arange(n_groups, device=device).unsqueeze(1)
        ).float()
        group_count = group_map.sum(dim=1)
        group_sum = group_map @ values.float()
        return group_sum, group_count

    def compute_robust_loss(
        self,
        per_sample_losses: torch.Tensor,
        group_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        group_loss, group_count = self.compute_group_average(per_sample_losses, group_idx)
        standard_training_loss = per_sample_losses.mean()
        observed_groups = (group_count > 0).float()

        # Keep absent groups neutral within the batch: they should not receive
        # adjustment-driven upweighting without any batch evidence.
        adjusted_loss = group_loss + (self.adjustment * observed_groups)
        if self.normalize_loss:
            adjusted_loss_sum = (adjusted_loss * observed_groups).sum().clamp_min(1e-12)
            adjusted_loss = adjusted_loss / adjusted_loss_sum
            adjusted_loss = adjusted_loss * observed_groups

        # adversarial group weights update
        effective_step_size = self.step_size * self.strength
        self.adv_probs = self.adv_probs * torch.exp(
            effective_step_size * adjusted_loss.detach()
        )
        # normalizing
        self.adv_probs = self.adv_probs / self.adv_probs.sum().clamp_min(1e-12)

        full_group_dro_loss = torch.dot(group_loss, self.adv_probs)
        robust_loss = (
            (1.0 - self.strength) * standard_training_loss
            + self.strength * full_group_dro_loss
        )
        return robust_loss, group_loss, group_count, self.adv_probs.detach().clone()
