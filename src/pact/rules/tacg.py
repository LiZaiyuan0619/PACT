"""TACG: trajectory-aware commit gating.

Re-implemented following the paper "TACG: Trajectory-Aware Commit Gating for
Diffusion Language Model Decoding" (Eq. 7, 8, 12, 15-18 and Algorithm 1).
TACG changes *when* a position may commit; like every rule here it writes the
drawn token.  Three pieces decide readiness:

* TILG (temporal implicit logits guidance) keeps an exponential moving average
  of past log probabilities as a reference and measures the gain of the most
  likely token against it: ``b_i = [q_t(x_hat_i) - p_ref(x_hat_i)]_+`` with the
  guided logits ``(1 + w) z_t - w z_ref``.
* The history gate requires the most likely token of a position to stay the
  same for ``history_min_streak`` steps, unless its probability reaches
  ``confidence_escape``.
* Capped extra promotion admits a few positions just below the threshold,
  ranked by ``c_i + rerank_lambda * b_i``; their number is at most
  ``min(extra_max, ceil(extra_ratio * base))`` for ``base`` positions above
  the threshold.

Working on log probabilities instead of raw logits is exact: the guidance is
invariant to adding a per-position constant, which softmax discards.
"""

from __future__ import annotations

import math

import torch

from .base import CommitRule, StepView, rank_positions


def logp_ema(probabilities: torch.Tensor, ema: torch.Tensor | None, *, decay: float) -> torch.Tensor:
    """``decay * ema + (1 - decay) * log p``, seeded with the first observation."""
    logp = torch.log(probabilities.clamp_min(1e-12))
    if ema is None:
        return logp.clone()
    return float(decay) * ema + (1.0 - float(decay)) * logp


def tilg_scores(probabilities: torch.Tensor, reference_logp: torch.Tensor, *,
                guidance_weight: float, rerank_lambda: float) -> dict:
    """Readiness scores ``s_i = c_i + lambda * b_i`` of the most likely token ``x_hat_i``."""
    logp = torch.log(probabilities.clamp_min(1e-12))
    x_hat = probabilities.argmax(dim=-1, keepdim=True)
    guided_logp = reference_logp + (1.0 + float(guidance_weight)) * (logp - reference_logp)
    guided_probs = torch.softmax(guided_logp, dim=-1)
    reference_probs = torch.softmax(reference_logp, dim=-1)
    confidence = probabilities.gather(-1, x_hat).squeeze(-1)
    bonus = torch.clamp(
        guided_probs.gather(-1, x_hat) - reference_probs.gather(-1, x_hat), min=0.0,
    ).squeeze(-1)
    return {"scores": confidence + float(rerank_lambda) * bonus,
            "confidence": confidence, "bonus": bonus}


def history_streaks(previous_top1: torch.Tensor | None, current_top1: torch.Tensor,
                    streaks: torch.Tensor | None) -> torch.Tensor:
    """Number of consecutive steps the most likely token of each position has stayed the same."""
    if previous_top1 is None or streaks is None:
        return torch.ones_like(current_top1)
    same = current_top1 == previous_top1
    return torch.where(same, streaks + 1, torch.ones_like(streaks))


def extra_budget(base_count: int, *, ratio: float, extra_max: int) -> int:
    """Promotion budget ``min(extra_max, ceil(ratio * max(base, 1)))``."""
    base = max(int(base_count), 0)
    raw = int(math.ceil(float(ratio) * max(base, 1)))
    return int(min(int(extra_max), max(raw, 0)))


class TACG(CommitRule):
    name = "tacg"

    def __init__(self, threshold: float = 0.9, guidance_weight: float = 0.3,
                 ema_decay: float = 0.95, rerank_lambda: float = 0.05,
                 extra_conf_floor: float = 0.8, extra_ratio: float = 0.25,
                 extra_max: int = 2, extra_allow_empty_base: bool = True,
                 history_min_streak: int = 3, confidence_escape: float = 0.95,
                 fallback: int = 1):
        self.threshold = float(threshold)
        self.guidance_weight = float(guidance_weight)
        self.ema_decay = float(ema_decay)
        self.rerank_lambda = float(rerank_lambda)
        self.extra_conf_floor = float(extra_conf_floor)
        self.extra_ratio = float(extra_ratio)
        self.extra_max = int(extra_max)
        self.extra_allow_empty_base = bool(extra_allow_empty_base)
        self.history_min_streak = int(history_min_streak)
        self.confidence_escape = float(confidence_escape)
        self.fallback = int(fallback)
        self.reset()

    def reset(self) -> None:
        self._logp_ema = None      # (n, L, V) reference log probabilities
        self._prev_top1 = None     # (n, L) most likely tokens of the previous step
        self._streak = None        # (n, L) history-gate counters
        self._ready = None         # (n, L) positions ready to commit

    def _observe(self, view: StepView) -> None:
        probs = view.probs.detach().to(torch.float32)
        eligible = view.eligible
        reference = self._logp_ema
        fresh = reference is None or reference.shape != probs.shape
        if fresh:
            # First step of a problem: the reference is the current belief, so
            # the bonus is zero and the step reduces to confidence gating.
            reference = logp_ema(probs, None, decay=self.ema_decay)
        scores = tilg_scores(probs, reference, guidance_weight=self.guidance_weight,
                             rerank_lambda=self.rerank_lambda)
        confidence = scores["confidence"]
        streaks = history_streaks(None if fresh else self._prev_top1, view.top1_token,
                                  None if fresh else self._streak)
        base_ready = eligible & (confidence > self.threshold)
        near_threshold = (eligible & (confidence > self.extra_conf_floor)
                          & (confidence <= self.threshold))
        ready = base_ready.clone()
        for row in range(probs.shape[0]):
            base_count = int(base_ready[row].sum().item())
            if base_count <= 0 and not self.extra_allow_empty_base:
                continue
            budget = extra_budget(base_count, ratio=self.extra_ratio, extra_max=self.extra_max)
            candidates = torch.nonzero(near_threshold[row], as_tuple=False).flatten()
            if budget <= 0 or candidates.numel() == 0:
                continue
            k_extra = min(budget, int(candidates.numel()))
            _, top = torch.topk(scores["scores"][row, candidates], k=k_extra)
            ready[row, candidates[top]] = True
        # The history gate filters the base and the promoted positions alike.
        persistent = streaks >= self.history_min_streak
        escape = confidence >= self.confidence_escape
        self._ready = ready & (persistent | escape) & eligible
        self._logp_ema = logp_ema(probs, reference, decay=self.ema_decay)
        self._prev_top1 = view.top1_token.detach().clone()
        self._streak = streaks

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
                # The history gate can refuse every position; the rule still has
                # to make progress, so the most confident positions commit.
                order = rank_positions(view.top1_prob[row], positions)
                take[row, order[: min(self.fallback, int(order.numel()))]] = True
        return take
