"""The five benchmarks: which problems, and the question each prompt carries.

=========  =======================================  ==========================
benchmark  problems (Appendix Table 2)              graded by
=========  =======================================  ==========================
humaneval  all 164                                  tests of the benchmark
mbpp       the 500 test problems (task ids 11-510)  tests of the benchmark
gsm8k      first 500 of the test split shuffled     final answer
           once with seed 113
math500    all 500                                  final answer
ifeval     first 500 of the 541 prompts             official strict check
=========  =======================================  ==========================

The question is placed in a single user turn under the model's chat template,
with no system message and no examples (:meth:`pact.models.DiffusionLM.encode_prompt`).
For the code benchmarks the question states the function to complete, as
below; MBPP also lists its unit tests.  MATH500 questions end with an
instruction to box the final answer.  GSM8K and IFEval questions are used as
they are.

The problem files under ``data/`` are written by ``data/prepare_data.py``
with the functions of this module.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

#: benchmark -> kind of grading
BENCHMARKS = {"humaneval": "code", "mbpp": "code", "gsm8k": "math",
              "math500": "math", "ifeval": "ifeval"}

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

GSM8K_SHUFFLE_SEED = 113
GSM8K_PROBLEMS = 500
IFEVAL_PROBLEMS = 500
MATH_INSTRUCT = r"(Please put the final answer in \boxed{} tag, i.e. $\boxed{answer here}$)"


def load_problems(benchmark: str, data_dir: str | Path | None = None,
                  num_problems: int = 0) -> list[dict]:
    """Problems of a benchmark in their fixed order (``num_problems <= 0``: all)."""
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {benchmark!r}; choose from {sorted(BENCHMARKS)}")
    path = Path(data_dir or DATA_DIR) / f"{benchmark}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run data/prepare_data.py")
    with path.open(encoding="utf-8") as handle:
        problems = [json.loads(line) for line in handle if line.strip()]
    return problems if num_problems <= 0 else problems[:num_problems]


def shard(problems: list, shard_id: int = 0, num_shards: int = 1) -> list:
    """Every ``num_shards``-th problem starting at ``shard_id`` (order kept)."""
    if not 0 <= shard_id < num_shards:
        raise ValueError("shard_id must lie in [0, num_shards)")
    return [p for i, p in enumerate(problems) if i % num_shards == shard_id]


# ---------------------------------------------------------------- questions

def humaneval_question(problem: dict) -> str:
    return ("You are an expert Python programmer. Your task is to complete the "
            f"implementation of a function named `{problem['entry_point']}`.\n\n"
            f"Here is the function to complete:\n```python\n{problem['prompt'].rstrip()}\n```\n")


def mbpp_entry_point(example: dict) -> str:
    return example["test_list"][0].split(" ")[1].split("(")[0]


def mbpp_question(example: dict) -> str:
    name = mbpp_entry_point(example)
    tests = "\n".join(f"    {test}" for test in example["test_list"])
    try:
        tree = ast.parse(example["test_list"][0].strip())
        n_args = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == name:
                n_args = len(node.args)
                break
    except Exception:  # noqa: BLE001
        n_args = 2
    params = ", ".join(f"input_param_{i + 1}" for i in range(n_args))
    return ("You are an expert Python programmer. Your task is to complete the "
            f"implementation of a function named `{name}`.\n\n"
            f"** TARGET FUNCTION **\n{example['text']}\n\n"
            "** UNIT TESTS **\n"
            f"Your code should pass unit tests like:\n{tests}\n\n"
            "Here is the function to complete:\n"
            f"```python\ndef {name}({params}):\n"
            f"    \"\"\"{example['text']}\n    \"\"\"\n```\n")


def math500_question(problem: str) -> str:
    return problem + " " + MATH_INSTRUCT


def gsm8k_gold(answer: str) -> str:
    """Reference answer of GSM8K: the text after ``####``."""
    return answer.split("####")[-1].strip()
