"""Grading answers (Appendix C.2).

* Code: the first code block of the answer is run against the tests of the
  benchmark (:mod:`pact.eval.code_exec`).
* Math: the final answer is extracted (the last ``\\boxed{}``, an "answer
  is" phrase, or else the last number; :mod:`pact.eval.math_parser`) and
  compared with the reference for equivalence (:mod:`pact.eval.math_grader`).  Symbolic checks run under a 1-second limit,
  so a heavily loaded machine can very rarely grade an answer differently.
* IFEval: the official strict check; an answer is correct when it follows
  every instruction of its prompt (:mod:`pact.eval.ifeval`).
"""

from __future__ import annotations

from .benchmarks import BENCHMARKS


def score_math(gold: str, answers: list[str]) -> list[bool]:
    from .math_grader import math_equal
    from .math_parser import extract_answer
    return [bool(math_equal(gold, extract_answer(answer), timeout=True)) for answer in answers]


def score_answers(benchmark: str, problems: dict, answers: dict, *,
                  code_workers: int = 8) -> dict:
    """``{problem_id: [correct, ...]}`` for ``answers = {problem_id: [text, ...]}``."""
    kind = BENCHMARKS[benchmark]
    if kind == "code":
        from .code_exec import score_code
        return score_code(benchmark, problems, answers, n_workers=code_workers)
    if kind == "math":
        return {pid: score_math(problems[pid]["gold"], texts) for pid, texts in answers.items()}
    from .ifeval import score_ifeval
    return {pid: [row["strict_prompt"] for row in score_ifeval(problems[pid], texts)]
            for pid, texts in answers.items()}
