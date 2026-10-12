"""The interface shared by every commit rule.

At each step the decoder runs one forward pass, draws a token for every masked
position at the sampling temperature, and hands a :class:`StepView` to the
commit rule.  The rule returns the positions whose drawn tokens are written;
the other positions stay masked.  A written token is never revised.

Every rule writes the drawn tokens.  The rules differ only in which positions
they commit, so all of them see exactly the same proposals at a given step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StepView:
    """What a commit rule sees at one step.

    All tensors cover the answer span: ``n`` answers (rows) by ``L`` positions.
    """

    #: (n, L, V) model distributions ``p_i``, before any temperature (fp32).
    probs: torch.Tensor
    #: (n, L) tokens drawn from ``q_i ∝ p_i^(1/T)``.
    tokens: torch.Tensor
    #: (n, L) ``p_i(x_i)``: the probability of the drawn token under ``p_i``.
    written_prob: torch.Tensor
    #: (n, L) ``max_v p_i(v)``: the probability of the most likely token.
    top1_prob: torch.Tensor
    #: (n, L) the most likely token at each position.
    top1_token: torch.Tensor
    #: (n, L) entropy of ``p_i``.
    entropy: torch.Tensor
    #: (n, L) masked positions of the current block, i.e. the candidates.
    eligible: torch.Tensor


def rank_positions(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """``positions`` sorted by ``values`` from high to low (ties keep index order)."""
    return positions[torch.argsort(values[positions], descending=True, stable=True)]


class CommitRule:
    """Base class.  Subclasses implement :meth:`select`."""

    #: name used on the command line and in the configs
    name: str = ""

    def reset(self) -> None:
        """Forget all state.  Called once per problem, before the first step."""

    def select(self, view: StepView) -> torch.Tensor:
        """Return an (n, L) boolean mask of the positions to commit.

        Every row that still has eligible positions commits at least one.
        """
        raise NotImplementedError

    def describe(self) -> dict:
        """The rule and its parameters, as stored with the outputs."""
        params = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        return {"rule": self.name, **params}
