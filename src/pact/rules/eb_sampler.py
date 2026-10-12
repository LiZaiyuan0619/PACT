"""EB-Sampler: commit the longest prefix whose entropy bound stays below gamma.

Re-implemented following the paper "Accelerated Sampling from Masked Diffusion
Models via Entropy Bounded Unmasking" (Eq. 2, Algorithm 1).  Positions are
sorted by the probability of their most likely
token.  For a prefix, the error proxy is the sum of the entropies minus the
largest one; the rule commits the longest prefix whose proxy is at most
``gamma``.  The first position always passes (its proxy is zero).
"""

from __future__ import annotations

import torch

from .base import CommitRule, StepView, rank_positions


class EBSampler(CommitRule):
    name = "eb_sampler"

    def __init__(self, gamma: float = 0.5):
        if float(gamma) < 0.0:
            raise ValueError("gamma must be >= 0")
        self.gamma = float(gamma)

    def select(self, view: StepView) -> torch.Tensor:
        take = torch.zeros_like(view.eligible)
        for row in range(take.shape[0]):
            positions = torch.nonzero(view.eligible[row], as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            order = rank_positions(view.top1_prob[row], positions)
            entropy = view.entropy[row, order].to(view.top1_prob.dtype)
            bound = torch.cumsum(entropy, dim=0) - torch.cummax(entropy, dim=0).values
            valid = torch.nonzero(bound <= self.gamma, as_tuple=False)
            k = int(valid[-1].item()) + 1
            take[row, order[:k]] = True
        return take
