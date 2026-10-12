import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pact.rules import StepView  # noqa: E402


def make_view(probs: torch.Tensor, tokens: torch.Tensor | None = None,
              eligible: torch.Tensor | None = None) -> StepView:
    """A StepView built from distributions ``probs`` of shape (n, L, V)."""
    probs = probs.to(torch.float32)
    if tokens is None:
        tokens = probs.argmax(dim=-1)
    if eligible is None:
        eligible = torch.ones(probs.shape[:2], dtype=torch.bool)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    return StepView(probs=probs, tokens=tokens,
                    written_prob=torch.gather(probs, -1, tokens.unsqueeze(-1)).squeeze(-1),
                    top1_prob=probs.max(dim=-1).values, top1_token=probs.argmax(dim=-1),
                    entropy=entropy, eligible=eligible)


def random_view(generator: torch.Generator, n: int = 2, length: int = 24, vocab: int = 6,
                sharpness: float = 3.0, temperature: float = 0.6,
                p_eligible: float = 0.7) -> StepView:
    logits = sharpness * torch.randn(n, length, vocab, generator=generator)
    probs = torch.softmax(logits, dim=-1)
    gumbel = -torch.log(-torch.log(torch.rand(n, length, vocab, generator=generator)
                                   .clamp(1e-9, 1 - 1e-7)))
    tokens = torch.argmax(logits + temperature * gumbel, dim=-1)
    eligible = torch.rand(n, length, generator=generator) < p_eligible
    eligible[:, 0] = True
    return make_view(probs, tokens, eligible)


@pytest.fixture
def generator():
    g = torch.Generator()
    g.manual_seed(0)
    return g
