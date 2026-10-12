"""KLASS: commit positions whose distribution has stopped changing.

Re-implemented following the paper "KLASS: KL-Guided Fast Inference in Masked
Diffusion Models" (Section 4.2).  After every forward pass the rule computes,
for each position, the KL divergence of the current distribution from the one
of the previous step.  A masked position is ready when its last
``history_length`` divergences are all below ``kl_threshold`` and its most
likely token has probability above ``confidence``.  All ready positions
commit; when none is ready, the ``fallback`` positions with the most likely
top tokens commit.  The first step of a problem has no previous distribution,
so it always falls back.
"""

from __future__ import annotations

import torch

from .base import CommitRule, StepView, rank_positions


class KLASS(CommitRule):
    name = "klass"

    def __init__(self, confidence: float = 0.9, kl_threshold: float = 0.001,
                 history_length: int = 2, fallback: int = 1):
        if int(history_length) < 1 or int(fallback) < 1:
            raise ValueError("history_length and fallback must be >= 1")
        self.confidence = float(confidence)
        self.kl_threshold = float(kl_threshold)
        self.history_length = int(history_length)
        self.fallback = int(fallback)
        self.reset()

    def reset(self) -> None:
        self._previous = None      # (n, L, V) distributions of the previous step
        self._history = None       # (n, L, history_length) recent KL divergences
        self._count = None         # (n, L) steps observed while eligible
        self._ready = None         # (n, L) positions ready to commit

    def _observe(self, view: StepView) -> None:
        probs = view.probs.detach().to(torch.float32)
        shape = probs.shape[:-1]
        if self._previous is None or self._previous.shape != probs.shape:
            self._previous = probs.clone()
            self._history = torch.full((*shape, self.history_length), float("inf"),
                                       dtype=torch.float32, device=probs.device)
            self._count = torch.zeros(shape, dtype=torch.int32, device=probs.device)
            self._ready = torch.zeros(shape, dtype=torch.bool, device=probs.device)
            return
        previous = self._previous.clamp_min(1e-12)
        current = probs.clamp_min(1e-12)
        kl = (current * (current.log() - previous.log())).sum(dim=-1)
        history = torch.roll(self._history, shifts=-1, dims=-1)
        history[..., -1] = kl
        counts = self._count
        eligible = view.eligible
        counts[eligible] = torch.clamp(counts[eligible] + 1, max=self.history_length)
        stable = (counts >= self.history_length) & torch.all(history < self.kl_threshold, dim=-1)
        top1 = probs.max(dim=-1).values
        self._history = history
        self._ready = eligible & stable & (top1 > self.confidence)
        self._previous = probs.clone()

    def select(self, view: StepView) -> torch.Tensor:
        self._observe(view)
        take = torch.zeros_like(view.eligible)
        for row in range(take.shape[0]):
            positions = torch.nonzero(view.eligible[row], as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            ready = view.eligible[row] & self._ready[row]
            if bool(ready.any()):
                take[row] = ready
            else:
                order = rank_positions(view.top1_prob[row], positions)
                take[row, order[: min(self.fallback, int(order.numel()))]] = True
        return take
