"""Loading the four diffusion language models and running one forward pass.

All four checkpoints are the public releases on Hugging Face and load with
``trust_remote_code``.  They differ in three details that the decoder must
respect:

* the mask token (LLaDA-8B and Dream use fixed ids; LLaDA-MoE and LLaDA2.1-mini
  declare it in the tokenizer);
* Dream predicts position ``i`` from the output at ``i - 1``, so its logits are
  shifted right by one;
* LLaDA2.1-mini attends with its block-causal mask over blocks of 32 tokens,
  as in its reference code.

Some checkpoints emit logits over more ids than the tokenizer knows; those ids
can never be decoded and are masked out before any probability is read.
"""

from __future__ import annotations

from typing import Any

import torch

#: Short names used in the configs, and the checkpoints they stand for.
MODELS = {
    "llada-8b": "GSAI-ML/LLaDA-8B-Instruct",
    "llada-moe": "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
    "llada2.1-mini": "inclusionAI/LLaDA2.1-mini",
    "dream-7b": "Dream-org/Dream-v0-Instruct-7B",
}

LLADA_MASK_ID = 126336   # <|mdm_mask|>
DREAM_MASK_ID = 151666   # <|mask|>
LLADA2_ATTENTION_BLOCK = 32


def resolve_model(name: str) -> str:
    """Map a short name to its checkpoint; other strings (local paths) pass through."""
    return MODELS.get(name, name)


def detect_family(config: Any) -> str:
    """``llada`` (LLaDA-8B, LLaDA-MoE), ``llada2`` (LLaDA2.1-mini) or ``dream``."""
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = " ".join(getattr(config, "architectures", []) or []).lower()
    text = f"{model_type} {architectures}"
    if "llada2" in text:
        return "llada2"
    if "dream" in text:
        return "dream"
    return "llada"


def decoding_ids(config: Any, tokenizer, family: str) -> tuple[int, int | None]:
    """``(mask_id, valid_vocab)``.  ``valid_vocab`` is None when no logit needs masking."""
    if family == "dream":
        return DREAM_MASK_ID, len(tokenizer)
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        mask_token = getattr(tokenizer, "mask_token", None)
        if mask_token:
            mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    if not isinstance(mask_id, int) or int(mask_id) < 0:
        mask_id = LLADA_MASK_ID
    vocab_size = int(getattr(config, "vocab_size", len(tokenizer)))
    valid_vocab = len(tokenizer) if vocab_size > len(tokenizer) else None
    return int(mask_id), valid_vocab


class DiffusionLM:
    """A loaded checkpoint with everything the decoder needs."""

    def __init__(self, model, tokenizer, family: str, mask_id: int, valid_vocab: int | None):
        self.model = model
        self.tokenizer = tokenizer
        self.family = family
        self.mask_id = mask_id
        self.valid_vocab = valid_vocab

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Logits aligned so that ``logits[:, i]`` predicts position ``i``."""
        return forward_logits(self.model, self.family, input_ids)

    def encode_prompt(self, question: str) -> torch.Tensor:
        """The question as a single user turn under the model's chat template."""
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], add_generation_prompt=True, tokenize=False)
        return self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)

    def decode_answers(self, answer_ids: torch.Tensor) -> list[str]:
        """Text of each answer; special tokens (end of text, padding) are dropped."""
        if self.valid_vocab is not None:
            answer_ids = answer_ids.masked_fill(answer_ids >= self.valid_vocab,
                                                self.tokenizer.eos_token_id)
        return self.tokenizer.batch_decode(answer_ids, skip_special_tokens=True)


def load_model(name: str, device: str = "cuda") -> DiffusionLM:
    """Load a checkpoint in bfloat16 from Hugging Face (or a local copy of it)."""
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    path = resolve_model(name)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    family = detect_family(config)
    model_cls = AutoModelForCausalLM if family == "llada2" else AutoModel
    model = (model_cls.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16)
             .eval().requires_grad_(False).to(device))
    mask_id, valid_vocab = decoding_ids(model.config, tokenizer, family)
    return DiffusionLM(model, tokenizer, family, mask_id, valid_vocab)


def llada2_block_inputs(batch: int, length: int, *, device, dtype,
                        block_length: int = LLADA2_ATTENTION_BLOCK):
    """Block-causal attention mask and position ids of LLaDA2, and the padded length."""
    blocks = (length + block_length - 1) // block_length
    padded = blocks * block_length
    block_mask = torch.tril(torch.ones(blocks, blocks, device=device, dtype=dtype))
    attention_mask = (block_mask.repeat_interleave(block_length, dim=0)
                      .repeat_interleave(block_length, dim=1).unsqueeze(0).unsqueeze(0))
    attention_mask = attention_mask.expand(batch, -1, -1, -1)
    position_ids = torch.arange(padded, device=device, dtype=torch.long).unsqueeze(0)
    return attention_mask, position_ids.expand(batch, -1), padded


def forward_logits(model, family: str, input_ids: torch.Tensor) -> torch.Tensor:
    """One forward pass with position-aligned logits for every family."""
    if family == "dream":
        logits = model(input_ids, "full", None).logits
        return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    if family == "llada":
        return model(input_ids).logits
    if family == "llada2":
        batch, length = input_ids.shape
        attention_mask, position_ids, padded = llada2_block_inputs(
            batch, length, device=input_ids.device, dtype=torch.bfloat16)
        if padded != length:
            mask_id = int(getattr(model.config, "mask_token_id", 156895))
            padded_ids = torch.full((batch, padded), mask_id, device=input_ids.device,
                                    dtype=input_ids.dtype)
            padded_ids[:, :length] = input_ids
            input_ids = padded_ids
        output = model(input_ids=input_ids, attention_mask=attention_mask,
                       position_ids=position_ids)
        return output.logits[:, :length]
    raise ValueError(f"unsupported model family: {family!r}")
