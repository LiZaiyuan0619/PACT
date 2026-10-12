"""PACT: certify the set of tokens that a step writes (Section 3, Algorithm 1).

One step of PACT, for each answer:

1. Score every masked position by its written-word score
   ``s_i = log p_i(x_i)``: the log probability of the drawn token ``x_i`` under
   the model distribution ``p_i`` *before* temperature.
2. Sort the positions by score, highest first.  Call this order ``pi``.
3. Set-level cut: ``k*`` is the length of the longest prefix of ``pi`` whose
   scores sum to at least ``log tau``.  If no prefix passes, ``k* = 1``.
4. Spacing: scan ``pi`` and keep a position only if it lies at least ``g`` away
   from every position kept so far, until ``k*`` positions are kept.  If the
   scan ends early, lower the gap by one and scan again.
5. List the kept positions in the order of ``pi`` and cut again: commit the
   longest prefix (at most ``k*`` positions) whose scores sum to at least
   ``log tau``, or the first position if none passes.

With ``g = 1`` spacing keeps the first ``k*`` positions of ``pi`` and the step
reduces to the set-level cut alone (Proposition 3).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .base import CommitRule, StepView, rank_positions


def passing_prefix(probs_in_order: torch.Tensor, log_tau: float) -> int:
    """Length of the longest prefix whose log probabilities sum to >= ``log_tau``.

    The running sum only decreases, so this is the number of passing prefixes.
    Returns 0 when even the first position fails.
    """
    cumulative = torch.cumsum(torch.log(probs_in_order.clamp_min(1e-12)), dim=0)
    valid = torch.nonzero(cumulative >= log_tau, as_tuple=False)
    return int(valid[-1].item()) + 1 if valid.numel() else 0


def spaced_scan(order: np.ndarray, k: int, gap: int) -> np.ndarray:
    """Spacing (Algorithm 1, lines 5-12).

    ``order`` lists the candidate positions from the highest score down.  The
    scan repeatedly takes the first remaining candidate that lies at least
    ``h`` away from every kept position; when none qualifies, ``h`` drops by
    one.  Returns the ``k`` kept positions in the order they were kept.
    """
    chosen = np.array([], dtype=np.int64)
    candidates = np.asarray(order, dtype=np.int64)
    for h in range(int(gap), 0, -1):
        if chosen.size == k:
            break
        while chosen.size < k and candidates.size:
            if chosen.size:
                far = np.all(np.abs(candidates[:, None] - chosen[None, :]) >= h, axis=1)
            else:
                far = np.ones(candidates.size, dtype=bool)
            first = int(np.argmax(far))
            if not far[first]:
                break
            chosen = np.append(chosen, candidates[first])
            candidates = np.delete(candidates, first)
    if chosen.size != k:
        raise RuntimeError("spacing could not keep k positions")
    return chosen


class PACT(CommitRule):
    """PACT with threshold ``tau`` and minimum gap ``gap`` (``gap = 1``: no spacing)."""

    name = "pact"

    def __init__(self, tau: float = 0.5, gap: int = 8):
        if not 0.0 < float(tau) <= 1.0:
            raise ValueError("tau must lie in (0, 1]")
        if int(gap) < 1:
            raise ValueError("gap must be >= 1")
        self.tau = float(tau)
        self.gap = int(gap)

    def select_row(self, score: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
        """Committed positions of one answer.

        ``score`` holds ``p_i(x_i)`` for every position and ``eligible`` marks
        the masked positions of the current block.
        """
        positions = torch.nonzero(eligible, as_tuple=False).flatten()
        if positions.numel() == 0:
            return positions
        log_tau = math.log(self.tau)
        order = rank_positions(score, positions)                      # pi
        k_star = passing_prefix(score[order], log_tau)                # set-level cut
        if self.gap <= 1:
            chosen = order[:k_star]
        elif k_star:
            kept = spaced_scan(order.cpu().numpy(), k_star, self.gap)
            kept = torch.as_tensor(np.sort(kept), device=score.device)
            kept_in_order = rank_positions(score, kept)               # pi^g
            k = passing_prefix(score[kept_in_order], log_tau)         # cut again
            chosen = kept_in_order[:k]
            if chosen.numel() == 0:
                # Unreachable: the first position of pi^g is pi_1, which passed above.
                chosen = kept[:1]
        else:
            chosen = order[:0]
        if chosen.numel() == 0:
            chosen = order[:1]                                         # no prefix passes
        return chosen

    def select(self, view: StepView) -> torch.Tensor:
        take = torch.zeros_like(view.eligible)
        for row in range(view.eligible.shape[0]):
            chosen = self.select_row(view.written_prob[row], view.eligible[row])
            take[row, chosen] = True
        return take
