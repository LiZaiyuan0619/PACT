"""pass@k and speed (Section 4.1 and Appendix C.4).

* pass@k is the unbiased estimator of Chen et al. (2021) over the ``n``
  answers drawn for each problem, averaged over problems.  With ``n = 8``,
  pass@1 is the fraction of correct answers.
* Speed is the number of tokens written per forward pass:
  ``gen_length / (forward passes per problem)``.  Every answer fills all
  ``gen_length`` positions, so this is also the speed-up over one token per
  step.  The answers of a problem are decoded together, and a problem's
  forward passes are counted until all of its answers are filled.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """``1 - C(n - c, k) / C(n, k)`` for one problem with ``c`` of ``n`` answers correct."""
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod([(n - c - i) / (n - i) for i in range(k)]))


def mean_pass_at_k(passed: Iterable[Sequence[bool]], k: int) -> float:
    """Mean pass@k over problems; ``passed`` holds one list of booleans per problem."""
    values = [pass_at_k(len(p), int(sum(bool(v) for v in p)), k) for p in passed]
    return float(np.mean(values))


def speed(forward_passes: Iterable[int], gen_length: int = 256) -> float:
    """Tokens written per forward pass: ``gen_length`` over the mean forward passes."""
    passes = np.asarray(list(forward_passes), dtype=np.float64)
    return float(gen_length / passes.mean())
