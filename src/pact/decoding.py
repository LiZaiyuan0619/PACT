"""The decoding loop shared by every commit rule (Section 2 and Appendix C.4).

An answer of ``gen_length`` positions starts fully masked.  The answer is cut
into blocks of ``block_length`` positions that are filled left to right; the
experiments in the paper use a single block of 256 positions, so the commit
rule alone decides the order in which positions are filled.  Each step

1. runs one forward pass of the model on the current sequence,
2. draws a token ``x_i`` from ``q_i ∝ p_i^(1/T)`` at every position
   (Gumbel-max sampling, as in the LLaDA reference sampler),
3. asks the commit rule which masked positions of the block to write, and
4. writes the drawn tokens at those positions.  Written tokens are never revised.

The answers of one problem are decoded together as a batch.  A step runs while
any answer of the batch still has a masked position, so the cost of a problem
is the number of forward passes until all its answers are filled, padding
after the end of an answer included.  No key-value cache is used.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from .rules.base import CommitRule, StepView


@dataclass
class DecodeResult:
    #: (n, gen_length) token ids of the answers
    answer_ids: torch.Tensor
    #: forward passes spent on the batch of answers
    forward_passes: int
    #: wall-clock seconds of the decoding loop
    seconds: float


def add_gumbel_noise(logits: torch.Tensor, temperature: float,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """Gumbel-max sampling: ``argmax`` of the result is a draw from ``softmax(logits / T)``.

    This is the log-space form of ``exp(logits) / (-log u) ** T`` used by the
    LLaDA reference sampler; both have the same argmax.  ``T = 0`` is greedy.
    """
    t = torch.as_tensor(temperature, device=logits.device, dtype=torch.float32)
    if bool((t == 0).all()):
        return logits
    u = torch.rand(logits.shape, device=logits.device, dtype=torch.float32,
                   generator=generator).clamp_(1e-9, 1.0 - 1e-7)
    return logits.to(torch.float32) - t * torch.log(-torch.log(u))


def predictive_stats(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(p, entropy, top1_prob)`` of the untempered distributions, in fp32."""
    p = F.softmax(logits.to(torch.float32), dim=-1)
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)
    top1_prob = p.max(dim=-1).values
    return p, entropy, top1_prob


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def decode(forward: Callable[[torch.Tensor], torch.Tensor], prompt_ids: torch.Tensor,
           rule: CommitRule, *, mask_id: int, valid_vocab: int | None = None,
           num_samples: int = 8, gen_length: int = 256, block_length: int = 256,
           temperature: float = 0.6, seed: int | None = None) -> DecodeResult:
    """Decode ``num_samples`` answers to one prompt with a commit rule.

    ``forward`` maps token ids ``(n, length)`` to aligned logits
    ``(n, length, V)``; :meth:`pact.models.DiffusionLM.logits` is one.
    ``seed`` seeds the generator that draws the tokens.
    """
    if gen_length % block_length:
        raise ValueError("gen_length must be a multiple of block_length")
    prompt = prompt_ids.reshape(1, -1)
    device = prompt.device
    n, plen = num_samples, prompt.shape[1]
    x = torch.full((n, plen + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :plen] = prompt
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
    rule.reset()
    span = slice(plen, plen + gen_length)
    forward_passes = 0
    _synchronize(device)
    started = time.perf_counter()
    for block in range(gen_length // block_length):
        lo, hi = plen + block * block_length, plen + (block + 1) * block_length
        while bool((x[:, lo:hi] == mask_id).any()):
            logits = forward(x)[:, span]
            forward_passes += 1
            if valid_vocab is not None and logits.shape[-1] > valid_vocab:
                logits = logits.clone()
                logits[..., valid_vocab:] = float("-inf")
            probs, entropy, top1_prob = predictive_stats(logits)
            tokens = torch.argmax(add_gumbel_noise(logits, temperature, generator), dim=-1)
            eligible = torch.zeros((n, gen_length), dtype=torch.bool, device=device)
            eligible[:, lo - plen:hi - plen] = x[:, lo:hi] == mask_id
            view = StepView(
                probs=probs, tokens=tokens,
                written_prob=torch.gather(probs, -1, tokens.unsqueeze(-1)).squeeze(-1),
                top1_prob=top1_prob, top1_token=logits.argmax(dim=-1),
                entropy=entropy, eligible=eligible)
            selected = rule.select(view) & eligible
            if bool((eligible.any(dim=1) & ~selected.any(dim=1)).any()):
                raise RuntimeError(f"rule {rule.name!r} committed nothing for an unfinished answer")
            answer = x[:, span]
            answer[selected] = tokens[selected]
    _synchronize(device)
    return DecodeResult(answer_ids=x[:, span].clone(), forward_passes=forward_passes,
                        seconds=time.perf_counter() - started)


def decode_question(lm, question: str, rule: CommitRule, **kwargs) -> tuple[list[str], DecodeResult]:
    """Decode the answers to a question with a loaded :class:`pact.models.DiffusionLM`."""
    result = decode(lm.logits, lm.encode_prompt(question), rule, mask_id=lm.mask_id,
                    valid_vocab=lm.valid_vocab, **kwargs)
    return lm.decode_answers(result.answer_ids), result
