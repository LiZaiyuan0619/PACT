"""PACT (Algorithm 1) and the properties stated in Proposition 3."""

import itertools
import math

import numpy as np
import torch
from conftest import make_view, random_view

from pact.rules import PACT
from pact.rules.pact import passing_prefix, spaced_scan


def one_row(scores, eligible=None):
    """A view with one answer whose drawn tokens have probabilities ``scores``."""
    scores = torch.tensor(scores, dtype=torch.float32)
    probs = torch.stack([scores, 1.0 - scores], dim=-1).unsqueeze(0)
    tokens = torch.zeros(1, len(scores), dtype=torch.long)
    if eligible is not None:
        eligible = torch.tensor([eligible])
    return make_view(probs, tokens, eligible)


def committed(rule, view, row=0):
    return set(torch.nonzero(rule.select(view)[row]).flatten().tolist())


def test_commits_only_the_top_position_when_no_prefix_passes():
    view = one_row([0.30, 0.45, 0.20, 0.40])
    assert committed(PACT(tau=0.5, gap=1), view) == {1}
    assert committed(PACT(tau=0.5, gap=8), view) == {1}


def test_committed_set_clears_the_threshold_at_every_temperature(generator):
    for tau, gap, temperature in itertools.product([0.05, 0.3, 0.5, 0.9], [1, 4, 8],
                                                   [0.0, 0.3, 0.6, 1.0, 1.5]):
        rule = PACT(tau=tau, gap=gap)
        for _ in range(10):
            view = random_view(generator, temperature=temperature)
            take = rule.select(view)
            for row in range(take.shape[0]):
                chosen = torch.nonzero(take[row]).flatten()
                assert len(chosen) >= 1
                assert bool(view.eligible[row, chosen].all())
                total = torch.log(view.written_prob[row, chosen]).sum().item()
                if total < math.log(tau) - 1e-5:
                    # the only exception: no prefix passes and the top position commits alone
                    top = view.written_prob[row][view.eligible[row]].max().item()
                    assert len(chosen) == 1
                    assert math.isclose(total, math.log(top), abs_tol=1e-6)


def test_without_spacing_commits_the_largest_set_that_clears_the_threshold(generator):
    """No set of masked positions that clears log(tau) is larger than the one PACT commits."""
    for tau in (0.1, 0.3, 0.5, 0.8):
        rule = PACT(tau=tau, gap=1)
        for _ in range(100):
            view = random_view(generator, n=1, length=8)
            positions = torch.nonzero(view.eligible[0]).flatten().tolist()
            logs = {p: math.log(view.written_prob[0, p].item()) for p in positions}
            largest = max((len(s) for r in range(len(positions) + 1)
                           for s in itertools.combinations(positions, r)
                           if sum(logs[p] for p in s) >= math.log(tau)), default=0)
            assert len(committed(rule, view)) == max(largest, 1)


def test_tau_near_one_commits_one_position_per_step(generator):
    for gap in (1, 8):
        rule = PACT(tau=1.0 - 1e-6, gap=gap)
        for _ in range(50):
            view = random_view(generator, sharpness=1.0)
            assert rule.select(view).sum(dim=1).tolist() == [1] * view.eligible.shape[0]


def test_gap_one_keeps_the_order():
    order = np.array([5, 6, 2, 9, 0, 1])
    for k in range(1, len(order) + 1):
        assert spaced_scan(order, k, 1).tolist() == order[:k].tolist()


def test_spacing_scans_in_score_order_and_lowers_the_gap():
    # gap 8 keeps 10 and 20; nothing else is 8 away from both, so the gap drops
    # to 7, where 3 qualifies.
    assert spaced_scan(np.array([10, 11, 12, 3, 20, 4]), 3, 8).tolist() == [10, 20, 3]
    # four neighbours: one position per gap until the gap reaches 3, then 1
    assert spaced_scan(np.array([0, 1, 2, 3]), 2, 8).tolist() == [0, 3]
    assert spaced_scan(np.array([0, 1, 2, 3]), 4, 8).tolist() == [0, 3, 1, 2]


def test_spacing_spreads_the_commit_and_the_second_cut_drops_fillers():
    scores = [0.5] * 32
    for p in (0, 1, 2, 3):
        scores[p] = 0.999
    for p in (10, 20, 30):
        scores[p] = 0.99
    view = one_row(scores)
    # set-level cut: 0.999^4 * 0.99^3 >= 0.95 > 0.999^4 * 0.99^3 * 0.5, so k* = 7
    assert committed(PACT(tau=0.95, gap=1), view) == {0, 1, 2, 3, 10, 20, 30}
    # spacing keeps 0, 10, 20, 30 at gap 8 and fills with 5, 15, 25 at gap 5;
    # the second cut keeps the four positions that clear the threshold
    assert committed(PACT(tau=0.95, gap=8), view) == {0, 10, 20, 30}


def test_spacing_never_commits_more_than_the_set_level_cut(generator):
    for _ in range(200):
        view = random_view(generator, n=1, length=32)
        positions = torch.nonzero(view.eligible[0]).flatten()
        order = positions[torch.argsort(view.written_prob[0, positions], descending=True,
                                        stable=True)]
        k_star = max(passing_prefix(view.written_prob[0, order], math.log(0.3)), 1)
        assert len(committed(PACT(tau=0.3, gap=8), view)) <= k_star


def test_only_eligible_positions_commit():
    view = one_row([0.99, 0.98, 0.97, 0.96], eligible=[False, True, False, True])
    assert committed(PACT(tau=0.5, gap=1), view) == {1, 3}
