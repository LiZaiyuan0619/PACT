"""Commit rules.  Each rule decides which drawn tokens a step writes.

======================  ========================  ==========================================
name                    class                     setting in Table 1 (Appendix Table 3)
======================  ========================  ==========================================
``pact``                :class:`PACT`             ``tau = 0.5``, ``gap = 8``
``fast_dllm``           :class:`FastDLLM`         ``threshold = 0.9``
``fast_dllm_pp``        :class:`FastDLLMPlusPlus` ``delta = 0``
``eb_sampler``          :class:`EBSampler`        ``gamma = 0.5``
``klass``               :class:`KLASS`            ``confidence = 0.9``, ``kl_threshold = 0.001``
``tacg``                :class:`TACG`             ``threshold = 0.9``
``fixed_k``             :class:`FixedK`           ``k = 4``
``one_token_per_step``  :class:`OneTokenPerStep`  --
======================  ========================  ==========================================
"""

from __future__ import annotations

from .base import CommitRule, StepView, rank_positions
from .eb_sampler import EBSampler
from .fast_dllm import FastDLLM
from .fast_dllm_pp import FastDLLMPlusPlus
from .fixed_k import FixedK
from .klass import KLASS
from .one_token import OneTokenPerStep
from .pact import PACT
from .tacg import TACG

RULES = {cls.name: cls for cls in (PACT, FastDLLM, FastDLLMPlusPlus, EBSampler,
                                   KLASS, TACG, FixedK, OneTokenPerStep)}


def make_rule(name: str, **params) -> CommitRule:
    """Build a rule from its name and parameters, e.g. ``make_rule("pact", tau=0.5, gap=8)``."""
    if name not in RULES:
        raise ValueError(f"unknown rule {name!r}; choose from {sorted(RULES)}")
    return RULES[name](**params)


__all__ = ["CommitRule", "StepView", "rank_positions", "RULES", "make_rule",
           "PACT", "FastDLLM", "FastDLLMPlusPlus", "EBSampler", "KLASS", "TACG",
           "FixedK", "OneTokenPerStep"]
