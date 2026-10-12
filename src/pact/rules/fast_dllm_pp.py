"""Fast-dLLM++: commit the longest prefix allowed by a Frechet profile.

Re-implemented following the paper "Fast-dLLM++: Frechet Profile Decoding for
Faster Diffusion LLM Inference" (Section 3).  Positions are sorted by the
probability ``c_j`` of their most likely token.  For the prefix of length
``k`` the rule compares the Frechet lower bound
``L_k = max(0, sum_{j<=k} c_j - (k - 1))`` with ``U_k = 1 - c_k`` and commits
the longest prefix with ``L_k - U_k > delta``, or one position if none
qualifies.
"""

from __future__ import annotations

import torch

from .base import CommitRule, StepView, rank_positions


class FastDLLMPlusPlus(CommitRule):
    name = "fast_dllm_pp"

    def __init__(self, delta: float = 0.0):
        self.delta = float(delta)

    def select(self, view: StepView) -> torch.Tensor:
        take = torch.zeros_like(view.eligible)
        for row in range(take.shape[0]):
            positions = torch.nonzero(view.eligible[row], as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            order = rank_positions(view.top1_prob[row], positions)
            profile = view.top1_prob[row, order]
            prefix_sum = torch.cumsum(profile, dim=0)
            offset = torch.arange(profile.numel(), device=profile.device, dtype=profile.dtype)
            lower = torch.clamp(prefix_sum - offset, min=0.0)
            upper = 1.0 - profile
            valid = torch.nonzero((lower - upper) > self.delta, as_tuple=False)
            k = int(valid[-1].item()) + 1 if valid.numel() else 1
            take[row, order[:k]] = True
        return take
