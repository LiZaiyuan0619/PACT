"""Fast-dLLM: commit every position whose most likely token clears a threshold.

Re-implemented following the paper "Fast-dLLM: Training-free Acceleration of
Diffusion LLM by Enabling KV Cache and Parallel Decoding" (Algorithm 1).
A position is scored by the probability of its most likely token, and every
masked position of the block with ``max_v p_i(v) >= threshold`` commits its
drawn token.  When no position clears the threshold, the position whose drawn
token is most probable commits, so that every step makes progress.
"""

from __future__ import annotations

import torch

from .base import CommitRule, StepView, rank_positions


class FastDLLM(CommitRule):
    name = "fast_dllm"

    def __init__(self, threshold: float = 0.9):
        self.threshold = float(threshold)

    def select(self, view: StepView) -> torch.Tensor:
        take = view.eligible & (view.top1_prob >= self.threshold)
        for row in range(take.shape[0]):
            positions = torch.nonzero(view.eligible[row], as_tuple=False).flatten()
            if positions.numel() and not bool(take[row].any()):
                order = rank_positions(view.written_prob[row], positions)
                take[row, order[0]] = True
        return take
