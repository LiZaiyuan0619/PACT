"""PACT: Certifying What Is Written in Parallel Decoding of Diffusion Language Models.

* :mod:`pact.rules`     -- PACT and the commit rules it is compared with
* :mod:`pact.decoding`  -- the decoding loop shared by all rules
* :mod:`pact.models`    -- loading LLaDA-8B, LLaDA-MoE, LLaDA2.1-mini and Dream-7B
* :mod:`pact.eval`      -- benchmarks, grading, pass@k and speed
"""

from .decoding import DecodeResult, decode, decode_question
from .rules import RULES, make_rule

__version__ = "0.1.0"

__all__ = ["DecodeResult", "decode", "decode_question", "RULES", "make_rule", "__version__"]
