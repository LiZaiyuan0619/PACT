"""The published rules, fixed k and one token per step on small hand-made steps."""

import math

import pytest
import torch
from conftest import make_view, random_view

from pact.rules import (EBSampler, FastDLLM, FastDLLMPlusPlus, FixedK, KLASS, OneTokenPerStep,
                        RULES, TACG, make_rule)


def dist(top1: list[float], vocab: int = 4) -> torch.Tensor:
    """(1, L, V) distributions whose most likely token (id 0) has probability ``top1``."""
    rows = []
    for p in top1:
        rest = (1.0 - p) / (vocab - 1)
        rows.append([p] + [rest] * (vocab - 1))
    return torch.tensor([rows], dtype=torch.float32)


def committed(rule, view, row=0):
    return set(torch.nonzero(rule.select(view)[row]).flatten().tolist())


def test_fast_dllm_commits_every_position_above_the_threshold():
    view = make_view(dist([0.95, 0.5, 0.91, 0.89]))
    assert committed(FastDLLM(threshold=0.9), view) == {0, 2}


def test_fast_dllm_falls_back_to_the_most_probable_drawn_token():
    probs = dist([0.6, 0.8, 0.7])
    tokens = torch.tensor([[0, 1, 0]])          # position 1 drew an unlikely token
    view = make_view(probs, tokens)
    assert committed(FastDLLM(threshold=0.9), view) == {2}


def test_fast_dllm_pp_takes_the_longest_prefix_with_a_positive_margin():
    view = make_view(dist([0.99, 0.98, 0.9, 0.6]))
    # prefix 1: L = 0.99, U = 0.01; prefix 2: L = 0.97, U = 0.02;
    # prefix 3: L = 0.87, U = 0.10; prefix 4: L = 0.47, U = 0.40 -> all margins > 0
    assert committed(FastDLLMPlusPlus(delta=0.0), view) == {0, 1, 2, 3}
    assert committed(FastDLLMPlusPlus(delta=0.5), view) == {0, 1, 2}


def test_eb_sampler_bounds_the_entropy_of_the_prefix():
    view = make_view(dist([0.99, 0.9, 0.6, 0.3]))
    entropies = view.entropy[0].tolist()
    gamma = entropies[1] + 1e-4              # admits positions 0 and 1 (max is dropped)
    assert committed(EBSampler(gamma=gamma), view) == {0, 1}
    assert committed(EBSampler(gamma=0.0), view) == {0}


def test_fixed_k_commits_the_k_most_probable_drawn_tokens():
    probs = dist([0.9, 0.8, 0.7, 0.6, 0.5])
    tokens = torch.tensor([[0, 0, 1, 0, 0]])
    assert committed(FixedK(k=2), make_view(probs, tokens)) == {0, 1}
    assert committed(FixedK(k=3), make_view(probs, tokens)) == {0, 1, 3}


def test_one_token_per_step_commits_exactly_one_position(generator):
    for _ in range(20):
        view = random_view(generator)
        take = OneTokenPerStep().select(view)
        assert take.sum(dim=1).tolist() == [1] * take.shape[0]
        for row in range(take.shape[0]):
            score = torch.where(view.eligible[row], view.written_prob[row],
                                torch.tensor(float("-inf")))
            assert take[row, int(score.argmax())]


def test_klass_waits_for_stable_distributions():
    rule = KLASS(confidence=0.9, kl_threshold=0.001, history_length=2)
    rule.reset()
    view = make_view(dist([0.95, 0.95, 0.5]))
    assert committed(rule, view) == {0}          # first step: no history, fall back to one
    assert committed(rule, view) == {0}          # one stable step is not enough
    assert committed(rule, view) == {0, 1}       # two stable steps and confident


def test_klass_rejects_a_changing_distribution():
    rule = KLASS(confidence=0.9, kl_threshold=0.001, history_length=2)
    rule.reset()
    for p in (0.95, 0.97, 0.99):                 # position 0 keeps moving
        view = make_view(dist([p, 0.95]))
        chosen = committed(rule, view)
    assert chosen == {1}                         # only the stable position is ready


def test_tacg_history_gate_needs_a_persistent_top_token():
    rule = TACG(threshold=0.9, history_min_streak=3, confidence_escape=0.99)
    rule.reset()
    view = make_view(dist([0.95, 0.93, 0.5]))
    assert committed(rule, view) == {0}          # streak 1: gated, fall back to one
    assert committed(rule, view) == {0}          # streak 2: gated
    assert committed(rule, view) == {0, 1}       # streak 3: both pass


def test_tacg_confidence_escape_skips_the_history_gate():
    rule = TACG(threshold=0.9, history_min_streak=3, confidence_escape=0.95)
    rule.reset()
    view = make_view(dist([0.97, 0.96, 0.5]))
    assert committed(rule, view) == {0, 1}


@pytest.mark.parametrize("name,params", [
    ("pact", {"tau": 0.5, "gap": 8}), ("pact", {"tau": 0.05, "gap": 1}),
    ("fast_dllm", {"threshold": 0.9}), ("fast_dllm_pp", {"delta": 0.0}),
    ("eb_sampler", {"gamma": 0.5}), ("klass", {"confidence": 0.9, "kl_threshold": 0.001}),
    ("tacg", {"threshold": 0.9}), ("fixed_k", {"k": 4}), ("one_token_per_step", {}),
])
def test_every_rule_commits_at_least_one_eligible_position(name, params, generator):
    rule = make_rule(name, **params)
    rule.reset()
    for _ in range(30):
        view = random_view(generator, n=3, length=40)
        take = rule.select(view)
        assert not bool((take & ~view.eligible).any())
        assert bool((take.sum(dim=1) >= 1).all())


def test_registry_lists_every_rule():
    assert set(RULES) == {"pact", "fast_dllm", "fast_dllm_pp", "eb_sampler", "klass", "tacg",
                          "fixed_k", "one_token_per_step"}
    with pytest.raises(ValueError):
        make_rule("unknown")
    assert math.isclose(make_rule("pact").tau, 0.5) and make_rule("pact").gap == 8
