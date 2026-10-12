"""Fixed k: commit the k positions whose drawn tokens are most probable."""

from __future__ import annotations

import torch

from .base import CommitRule, StepView, rank_positions


class FixedK(CommitRule):
    name = "fixed_k"

    def __init__(self, k: int = 4):
        if int(k) < 1:
            raise ValueError("k must be >= 1")
        self.k = int(k)

    def select(self, view: StepView) -> torch.Tensor:
        take = torch.zeros_like(view.eligible)
        for row in range(take.shape[0]):
            positions = torch.nonzero(view.eligible[row], as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            order = rank_positions(view.written_prob[row], positions)
            take[row, order[: min(self.k, int(positions.numel()))]] = True
        return take
