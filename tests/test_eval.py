"""Benchmarks, grading, pass@k, speed and configs."""

import math
from pathlib import Path

import pytest

from pact.config import expand, load_config
from pact.eval import BENCHMARKS, load_problems, mean_pass_at_k, pass_at_k, speed
from pact.eval.code_exec import build_test_program, check_program, extract_code
from pact.eval.scoring import score_math
from pact.rules import make_rule

ROOT = Path(__file__).resolve().parents[1]


def test_problem_files_hold_the_problems_of_the_paper():
    sizes = {"humaneval": 164, "mbpp": 500, "gsm8k": 500, "math500": 500, "ifeval": 500}
    for name in BENCHMARKS:
        problems = load_problems(name)
        assert len(problems) == sizes[name]
        assert len({str(p["id"]) for p in problems}) == sizes[name]
        assert all(p["question"] for p in problems)
    assert [p["id"] for p in load_problems("mbpp")][:2] == [11, 12]
    assert load_problems("gsm8k")[0]["id"].startswith("test:")


def test_pass_at_k():
    assert pass_at_k(8, 0, 1) == 0.0
    assert pass_at_k(8, 8, 8) == 1.0
    assert math.isclose(pass_at_k(8, 3, 1), 3 / 8)
    assert math.isclose(pass_at_k(4, 1, 2), 0.5)
    assert math.isclose(mean_pass_at_k([[True, False], [False, False]], 1), 0.25)
    assert math.isclose(mean_pass_at_k([[True] + [False] * 7, [False] * 8], 8), 0.5)


def test_speed_is_positions_over_forward_passes():
    assert math.isclose(speed([256, 256]), 1.0)
    assert math.isclose(speed([50, 60], gen_length=256), 256 / 55)


def test_math_grading():
    assert score_math("10", ["So he can make \\boxed{10} packs.", "The answer is 12."]) == [True, False]
    assert score_math("\\frac{1}{2}", ["$\\boxed{\\dfrac{1}{2}}$"]) == [True]
    assert score_math("18", ["\\boxed{18.0}"]) == [True]
    assert score_math("(\\frac{27}{8},\\frac{15}{8})",
                      ["\\boxed{\\left(\\frac{27}{8}, \\frac{15}{8}\\right)}"]) == [True]


def test_code_grading_runs_the_first_code_block():
    problem = {"prompt": "def add(a, b):\n    \"\"\"Return a + b.\"\"\"\n", "entry_point": "add",
               "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n"}
    good = "Here it is:\n```python\ndef add(a, b):\n    return a + b\n```\n```python\nprint(1)\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    assert extract_code(good).startswith("def add")
    assert check_program(build_test_program(problem, good, "humaneval"))
    assert not check_program(build_test_program(problem, bad, "humaneval"))
    mbpp = {"test_list": ["assert add(2, 2) == 4"]}
    assert check_program(build_test_program(mbpp, good, "mbpp"))


def test_ifeval_strict_check():
    pytest.importorskip("langdetect")
    pytest.importorskip("immutabledict")
    from pact.eval.ifeval import score_ifeval
    problem = {"id": 0, "question": "Describe a cat without using any commas.",
               "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}
    results = score_ifeval(problem, ["A cat is a small animal.", "A cat, a small animal."])
    assert [r["strict_prompt"] for r in results] == [True, False]


def test_table1_config_expands_to_every_cell():
    runs = expand(load_config(ROOT / "configs/table1.yaml"))
    assert len(runs) == 4 * 5 * 8
    pact = {r.model: r.params for r in runs if r.name == "pact" and r.benchmark == "gsm8k"}
    assert pact["llada-8b"] == {"tau": 0.5, "gap": 8}
    assert pact["llada2.1-mini"] == {"tau": 0.5, "gap": 1}
    mbpp = {r.num_shards for r in runs if r.benchmark == "mbpp"}
    assert mbpp == {2}
    assert all(r.decoding["temperature"] == 0.6 and r.decoding["block_length"] == 256 for r in runs)


def test_every_config_expands():
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        runs = expand(load_config(path))
        assert runs, path
        for run in runs:
            make_rule(run.rule, **run.params)
