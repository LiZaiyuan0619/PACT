"""The decoding loop with a small random model on CPU."""

import pytest
import torch

from pact.decoding import add_gumbel_noise, decode
from pact.rules import make_rule

MASK = 9
VOCAB = 9          # ids 0..8 are words, 9 is the mask (outside the logits)


class TinyModel:
    """Logits that depend on the position and on the tokens already written."""

    def __init__(self, seed: int = 0, sharpness: float = 4.0):
        g = torch.Generator().manual_seed(seed)
        self.position = sharpness * torch.randn(64, VOCAB, generator=g)
        self.context = torch.randn(VOCAB + 1, VOCAB, generator=g)
        self.calls = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        n, length = x.shape
        written = (x != MASK).float().unsqueeze(-1)
        context = (self.context[x] * written).mean(dim=1, keepdim=True)
        return self.position[:length].unsqueeze(0).expand(n, -1, -1) + context


PROMPT = torch.tensor([1, 2, 3])


def run(rule, **kwargs):
    model = TinyModel()
    settings = dict(mask_id=MASK, num_samples=4, gen_length=16, block_length=16,
                    temperature=0.6, seed=113)
    settings.update(kwargs)
    result = decode(model, PROMPT, rule, **settings)
    return model, result


@pytest.mark.parametrize("name,params", [
    ("pact", {"tau": 0.5, "gap": 8}), ("pact", {"tau": 0.05, "gap": 1}),
    ("fast_dllm", {"threshold": 0.9}), ("fast_dllm_pp", {"delta": 0.0}),
    ("eb_sampler", {"gamma": 0.5}), ("klass", {"confidence": 0.9, "kl_threshold": 0.001}),
    ("tacg", {"threshold": 0.9}), ("fixed_k", {"k": 4}), ("one_token_per_step", {}),
])
def test_every_rule_fills_the_answer(name, params):
    model, result = run(make_rule(name, **params))
    assert result.answer_ids.shape == (4, 16)
    assert not bool((result.answer_ids == MASK).any())
    assert result.forward_passes == model.calls
    assert 1 <= result.forward_passes <= 16


def test_one_token_per_step_spends_one_forward_pass_per_position():
    _, result = run(make_rule("one_token_per_step"))
    assert result.forward_passes == 16


def test_fixed_k_spends_length_over_k_forward_passes():
    _, result = run(make_rule("fixed_k", k=4))
    assert result.forward_passes == 4


def test_decoding_is_reproducible_under_a_seed():
    rule = make_rule("pact", tau=0.3, gap=4)
    _, first = run(rule)
    _, second = run(rule)
    _, other = run(rule, seed=7)
    assert torch.equal(first.answer_ids, second.answer_ids)
    assert first.forward_passes == second.forward_passes
    assert not torch.equal(first.answer_ids, other.answer_ids)


def test_greedy_decoding_writes_the_most_likely_tokens():
    """At temperature 0 one token per step reduces to the greedy argmax decoder."""
    model = TinyModel()
    result = decode(model, PROMPT, make_rule("one_token_per_step"), mask_id=MASK,
                    num_samples=1, gen_length=8, block_length=8, temperature=0.0)
    x = torch.full((1, 3 + 8), MASK, dtype=torch.long)
    x[0, :3] = PROMPT
    for _ in range(8):                         # replay: commit the most confident position
        logits = model(x)[:, 3:]
        probs = torch.softmax(logits, dim=-1)
        conf, tok = probs.max(dim=-1)
        conf = torch.where(x[:, 3:] == MASK, conf, torch.tensor(float("-inf")))
        pos = int(conf.argmax())
        x[0, 3 + pos] = tok[0, pos]
    assert torch.equal(result.answer_ids, x[:, 3:])


def test_blocks_are_filled_left_to_right():
    order = []

    class Recorder(TinyModel):
        def __call__(self, x):
            order.append((x[:, 3:] != MASK).sum(dim=0).tolist())
            return super().__call__(x)

    decode(Recorder(), PROMPT, make_rule("fixed_k", k=2), mask_id=MASK, num_samples=2,
           gen_length=8, block_length=4, temperature=0.6, seed=1)
    # while the first block (positions 0-3) is unfinished, nothing in the second is written
    for written in order:
        if min(written[:4]) < 2:
            assert sum(written[4:]) == 0


def test_gumbel_sampling_matches_the_tempered_distribution():
    logits = torch.tensor([[[2.0, 1.0, 0.0]]]).expand(20000, 1, 3)
    g = torch.Generator().manual_seed(0)
    draws = torch.argmax(add_gumbel_noise(logits, 0.5, g), dim=-1).flatten()
    freq = torch.bincount(draws, minlength=3).float() / draws.numel()
    expected = torch.softmax(torch.tensor([2.0, 1.0, 0.0]) / 0.5, dim=0)
    assert torch.allclose(freq, expected, atol=0.01)
