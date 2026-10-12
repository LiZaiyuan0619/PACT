"""Scoring IFEval answers with the official checker (strict, prompt level).

The official code of Google Research is vendored unchanged under
``third_party/instruction_following_eval`` (Apache-2.0).  An answer counts as
correct when it follows every instruction of its prompt under the strict
check.  The checker needs the NLTK ``punkt`` tokenizer; see README.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

THIRD_PARTY = Path(__file__).resolve().parents[3] / "third_party"


def _official():
    if str(THIRD_PARTY) not in sys.path:
        sys.path.insert(0, str(THIRD_PARTY))
    from instruction_following_eval import evaluation_lib
    return evaluation_lib


def score_ifeval(problem: dict, answers: Sequence[str]) -> list[dict]:
    """Strict and loose results of each answer to one IFEval prompt."""
    evaluation_lib = _official()
    example = evaluation_lib.InputExample(
        key=int(problem["id"]),
        instruction_id_list=list(problem["instruction_id_list"]),
        prompt=str(problem["question"]),
        kwargs=list(problem["kwargs"]),
    )
    results = []
    for answer in answers:
        mapping = {example.prompt: str(answer)}
        strict = evaluation_lib.test_instruction_following_strict(example, mapping)
        loose = evaluation_lib.test_instruction_following_loose(example, mapping)
        results.append({
            "strict_prompt": bool(strict.follow_all_instructions),
            "strict_instructions": [bool(v) for v in strict.follow_instruction_list],
            "loose_prompt": bool(loose.follow_all_instructions),
            "loose_instructions": [bool(v) for v in loose.follow_instruction_list],
        })
    return results
