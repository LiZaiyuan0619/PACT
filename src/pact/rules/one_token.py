"""One token per step: the reference sampler of LLaDA with as many steps as positions.

Each step commits the single position whose drawn token is most probable
("low-confidence remasking" in the LLaDA reference code).  An answer of 256
positions therefore takes 256 forward passes, which defines a speed of 1.0x.
"""

from __future__ import annotations

import torch

from .base import CommitRule, StepView


class OneTokenPerStep(CommitRule):
    name = "one_token_per_step"

    def select(self, view: StepView) -> torch.Tensor:
        take = torch.zeros_like(view.eligible)
        score = torch.where(view.eligible, view.written_prob,
                            torch.full_like(view.written_prob, float("-inf")))
        for row in range(take.shape[0]):
            if bool(view.eligible[row].any()):
                take[row, torch.topk(score[row], k=1).indices] = True
        return take
